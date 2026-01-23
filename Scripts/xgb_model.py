import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score, confusion_matrix
from imblearn.metrics import specificity_score
import xgboost as xgb
import h5py

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

protein = ['ProtBert']
molecule = ['MolFormer']

for i in range (5):

    for prot in protein:
        for mol in molecule:

            prot_feat = h5py.File('../../{}_features.h5'.format(prot))
            mol_feat = h5py.File('../../{}_features.h5'.format(mol))

            if prot == 'ESM':
                target_dim = 2560
            else:
                target_dim = 1024

            if mol == 'MolFormer':
                drug_dim = 768
            else:
                drug_dim = 2048

            def extract_prot(seq):
                prot_dataset = prot_feat[seq]
                return np.squeeze(prot_dataset[()])

            def extract_mol(smiles):
                mol_dataset = mol_feat[smiles]
                return np.squeeze(mol_dataset[()])
            
            class DTIDataset(Dataset):
                def __init__(self, dataframe):
                    self.proteins = dataframe['Target Sequence'].tolist()
                    self.smiles = dataframe['SMILES'].tolist()
                    self.labels = dataframe['Label'].tolist()
                def __len__(self):
                    return len(self.labels)
                def __getitem__(self, idx):
                    protein_seq = self.proteins[idx]
                    smiles_str = self.smiles[idx]
                    label = self.labels[idx]
                    protein_inputs = extract_prot(protein_seq)
                    mol_inputs = extract_mol(smiles_str)
                    return {
                        'protein_embeddings': torch.tensor(protein_inputs, dtype=torch.float32),
                        'mol_embeddings': torch.tensor(mol_inputs, dtype=torch.float32),
                        'label': torch.tensor(label, dtype=torch.float32)
                    }

            class DTIFeatureExtractor(nn.Module):
                def __init__(self, latent_activation=nn.ReLU):
                    super(DTIFeatureExtractor, self).__init__()
                    self.latent_activation = latent_activation
                    self.drug_projector = nn.Sequential(nn.Linear(drug_dim, 1024), self.latent_activation())
                    nn.init.xavier_normal_(self.drug_projector[0].weight)
                    self.target_projector = nn.Sequential(nn.Linear(target_dim, 1024), self.latent_activation())
                    nn.init.xavier_normal_(self.target_projector[0].weight)
                def forward(self, drug, target):
                    drug_projection = self.drug_projector(drug)
                    target_projection = self.target_projector(target)
                    combined_features = torch.cat([drug_projection, target_projection], dim=1)
                    return combined_features

            def extract_features_for_xgb(model, dataloader, device):
                model.eval()
                all_features, all_labels = [], []
                with torch.no_grad():
                    for batch in dataloader:
                        mol_embeddings = batch['mol_embeddings'].to(device)
                        protein_embeddings = batch['protein_embeddings'].to(device)
                        labels = batch['label'].to(device)
                        features = model(mol_embeddings, protein_embeddings)
                        all_features.append(features.cpu().numpy())
                        all_labels.append(labels.cpu().numpy())
                return np.vstack(all_features), np.concatenate(all_labels)

            def evaluate_xgb_model(xgb_model, features, labels):
                preds_proba = xgb_model.predict_proba(features)[:, 1]
                preds = (preds_proba > 0.5).astype(int)   
                accuracy = accuracy_score(labels, preds)
                precision = precision_score(labels, preds, zero_division=0)
                recall = recall_score(labels, preds, zero_division=0)
                specificity = specificity_score(labels, preds, average='binary', pos_label=1)
                try:
                    auc = roc_auc_score(labels, preds_proba)
                    mcc = matthews_corrcoef(labels, preds)
                except ValueError:
                    auc = 0.5
                    mcc = 0
                    print("Warning: AUC/MCC could not be calculated.")
                return preds, preds_proba, accuracy, precision, recall, specificity, mcc, auc
            

            train_df = pd.read_csv('../../training_set.csv', index_col=0).reset_index(drop=True)
            val_df = pd.read_csv('../../validation_set.csv', index_col=0).reset_index(drop=True)
            test_df_unseen_prot = pd.read_csv('../../test_set_unseen_protein.csv', index_col=0).dropna().reset_index(drop=True)
            test_df_unseen_lig = pd.read_csv('../../test_set_unseen_ligands.csv', index_col=0).dropna().reset_index(drop=True)

            val_df[['Predictions', 'Predictions_Proba']] = np.nan
            test_df_unseen_prot[['Predictions', 'Predictions_Proba']] = np.nan
            test_df_unseen_lig[['Predictions', 'Predictions_Proba']] = np.nan

            for df in [train_df, val_df, test_df_unseen_prot, test_df_unseen_lig]:
                df['Prot_feat'] = df['Target Sequence'].map(extract_prot)
                df['Mol_feat'] = df['SMILES'].map(extract_mol)

            train_dataset = DTIDataset(train_df)
            val_dataset = DTIDataset(val_df)
            test_dataset_unseen_prot = DTIDataset(test_df_unseen_prot)
            test_dataset_unseen_lig = DTIDataset(test_df_unseen_lig)

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)  # Fixed: Changed to False
            test_loader_unseen_prot = DataLoader(test_dataset_unseen_prot, batch_size=32, shuffle=False)
            test_loader_unseen_lig = DataLoader(test_dataset_unseen_lig, batch_size=32, shuffle=False)

            feature_extractor = DTIFeatureExtractor().to(DEVICE)

            print("Extracting features for XGBoost...")
            train_features, train_labels = extract_features_for_xgb(feature_extractor, train_loader, DEVICE)
            val_features, val_labels = extract_features_for_xgb(feature_extractor, val_loader, DEVICE)
            test_unseen_prot_features, test_unseen_prot_labels = extract_features_for_xgb(feature_extractor, test_loader_unseen_prot, DEVICE)
            test_unseen_lig_features, test_unseen_lig_labels = extract_features_for_xgb(feature_extractor, test_loader_unseen_lig, DEVICE)

            print("Training XGBoost model...")
            xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

            xgb_model.fit(
                train_features, train_labels,
                eval_set=[(val_features, val_labels)],
                verbose=True
            )

            print("\nEvaluating on validation set...")
            val_preds, val_preds_proba, val_acc, val_precision, val_recall, val_spec, val_mcc, val_auc = evaluate_xgb_model(xgb_model, val_features, val_labels)
            print(f"Val Acc: {val_acc:.3f}, Val Sensitivity: {val_recall:.3f}, Val Specificity: {val_spec:.3f}, Val MCC: {val_mcc:.3f}, Val AUC: {val_auc:.3f}")
            val_df['Predictions'] = val_preds
            val_df['Predictions_Proba'] = val_preds_proba
            val_df[['UniProt', 'SMILES', 'Label', 'Predictions']].to_csv('val_pred_xgb_{}_{}_{}.csv'.format(prot, mol, i))

            print("\nEvaluating on test set (unseen protein)...")
            test_preds, test_preds_proba, test_acc, test_precision, test_recall, test_spec, test_mcc, test_auc = evaluate_xgb_model(xgb_model, test_unseen_prot_features, test_unseen_prot_labels)
            print(f"Test Acc: {test_acc:.3f}, Test Sensitivity: {test_recall:.3f}, Test Specificity: {test_spec:.3f}, Test MCC: {test_mcc:.3f}, Test AUC: {test_auc:.3f}")
            test_df_unseen_prot['Predictions'] = test_preds
            test_df_unseen_prot['Predictions_Proba'] = test_preds_proba
            test_df_unseen_prot[['UniProt', 'SMILES', 'Label', 'Predictions']].to_csv('test_pred_unseen_prot_xgb_{}_{}_{}.csv'.format(prot, mol, i))


            print("\nEvaluating on test set (unseen ligand)...")
            test_preds, test_preds_proba, test_acc, test_precision, test_recall, test_spec, test_mcc, test_auc = evaluate_xgb_model(xgb_model, test_unseen_lig_features, test_unseen_lig_labels)
            print(f"Test Acc: {test_acc:.3f}, Test Sensitivity: {test_recall:.3f}, Test Specificity: {test_spec:.3f}, Test MCC: {test_mcc:.3f}, Test AUC: {test_auc:.3f}")
            test_df_unseen_lig['Predictions'] = test_preds
            test_df_unseen_lig['Predictions_Proba'] = test_preds_proba
            test_df_unseen_lig[['UniProt', 'SMILES', 'Label', 'Predictions']].to_csv('test_pred_unseen_lig_xgb_{}_{}_{}.csv'.format(prot, mol, i))


            xgb_model.save_model('dti_model_xgb_{}_{}_{}.pth'.format(prot, mol, i))
            print("Training finished.")

            def cal_metrics(df):
                pred = list(df['Predictions'])
                pred_proba = list(df['Predictions_Proba'])
                label = list(df['Label'])
                tn, fp, fn, tp = confusion_matrix(label, pred, labels=[0,1]).ravel()
                acc = np.round(accuracy_score(label, pred), 3)
                mcc = np.round(matthews_corrcoef(label, pred), 3)
                roc = np.round(roc_auc_score(label, pred_proba), 3)
                sen = np.round(tp/(tp + fn), 3)
                spec = np.round(tn/(tn + fp), 3)
                return acc, spec, sen, mcc, roc

            accuracy, specificity, sensitivity, mcc, roc = cal_metrics(val_df)
            print('Val', accuracy, specificity, sensitivity, mcc, roc)

            accuracy, specificity, sensitivity, mcc, roc = cal_metrics(test_df_unseen_prot)
            print('Unseen protein', accuracy, specificity, sensitivity, mcc, roc)

            accuracy, specificity, sensitivity, mcc, roc = cal_metrics(test_df_unseen_lig)
            print('Unseen ligand', accuracy, specificity, sensitivity, mcc, roc)
