import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score
from imblearn.metrics import specificity_score
import h5py

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 50

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


            class Cosine(nn.Module):
                def forward(self, x1, x2):
                    return nn.CosineSimilarity()(x1, x2)


            class DTIProfilerCosine(nn.Module):
                def __init__(self, latent_activation=nn.ReLU):
                    super(DTIProfilerCosine, self).__init__()
                    self.latent_activation = latent_activation
                    self.drug_projector = nn.Sequential(nn.Linear(drug_dim, 1024), self.latent_activation())
                    nn.init.xavier_normal_(self.drug_projector[0].weight)
                    self.target_projector = nn.Sequential(nn.Linear(target_dim, 1024), self.latent_activation())
                    nn.init.xavier_normal_(self.target_projector[0].weight)
                    self.activator = Cosine()
                def forward(self, drug, target):
                    drug_projection = self.drug_projector(drug)
                    target_projection = self.target_projector(target)
                    distance = self.activator(drug_projection, target_projection)
                    return distance.unsqueeze(1)


            def train_model(model, dataloader, criterion, optimizer, device):
                model.train()
                total_loss, all_preds, all_labels, all_probs = 0, [], [], []
                for batch in dataloader:
                    mol_embeddings = batch['mol_embeddings'].to(device)
                    protein_embeddings = batch['protein_embeddings'].to(device)
                    labels = batch['label'].unsqueeze(1).to(device)
                    optimizer.zero_grad()
                    logits = torch.clamp(model(mol_embeddings, protein_embeddings), max=1)
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    preds = (logits > 0.5).float()
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(logits.cpu().detach().numpy())
                avg_loss = total_loss / len(dataloader)
                accuracy = accuracy_score(all_labels, all_preds)
                precision = precision_score(all_labels, all_preds, zero_division=0)
                recall = recall_score(all_labels, all_preds, zero_division=0)
                specificity = specificity_score(all_labels, all_preds, average='binary', pos_label=1)
                try:
                    auc = roc_auc_score(all_labels, all_probs)
                    mcc = matthews_corrcoef(all_labels, all_preds)
                except ValueError:
                    auc = 0.5
                    mcc = 0
                    print("Warning: AUC/MCC could not be calculated (likely only one class in predictions/labels during this eval).")
                return avg_loss, accuracy, precision, recall, specificity, mcc, auc


            def evaluate_model(model, dataloader, criterion, device):
                model.eval()
                total_loss, all_preds, all_labels, all_probs = 0, [], [], []
                with torch.no_grad():
                    for batch in dataloader:
                        mol_embeddings = batch['mol_embeddings'].to(device)
                        protein_embeddings = batch['protein_embeddings'].to(device)
                        labels = batch['label'].unsqueeze(1).to(device)
                        logits = torch.clamp(model(mol_embeddings, protein_embeddings), max=1)
                        loss = criterion(logits, labels)
                        total_loss += loss.item()
                        preds = (logits > 0.5).float()
                        all_preds.extend(preds.cpu().numpy().flatten())
                        all_labels.extend(labels.cpu().numpy().flatten())
                        all_probs.extend(logits.cpu().numpy().flatten())
                avg_loss = total_loss / len(dataloader)
                accuracy = accuracy_score(all_labels, all_preds)
                precision = precision_score(all_labels, all_preds, zero_division=0)
                recall = recall_score(all_labels, all_preds, zero_division=0)
                specificity = specificity_score(all_labels, all_preds, average='binary', pos_label=1)
                try:
                    auc = roc_auc_score(all_labels, all_probs)
                    mcc = matthews_corrcoef(all_labels, all_preds)
                except ValueError:
                    auc = 0.5
                    mcc = 0
                    print("Warning: AUC/MCC could not be calculated (likely only one class in predictions/labels during this eval).")
                return all_preds, all_probs, avg_loss, accuracy, precision, recall, specificity, mcc, auc
            

            train_df = pd.read_csv('../../training_set.csv', index_col=0).dropna().reset_index(drop=True)
            val_df = pd.read_csv('../../validation_set.csv', index_col=0).dropna().reset_index(drop=True)
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

            model = DTIProfilerCosine().to(DEVICE)
            optimizer = optim.AdamW(model.parameters(), lr=1e-4)
            criterion = nn.BCELoss()

            best_val_auc = 0
            best_val_loss = float('inf')

            for epoch in range(EPOCHS):
                print("----Protein Model: {}----".format(prot))
                print("----Molecule Model: {}----".format(mol))
                train_loss, train_acc, train_precision, train_recall, train_spec, train_mcc, train_auc = train_model(model, train_loader, criterion, optimizer, DEVICE)
                val_preds, val_probs, val_loss, val_acc, val_precision, val_recall, val_spec, val_mcc, val_auc = evaluate_model(model, val_loader, criterion, DEVICE)
                print(f"Epoch {epoch+1}/{EPOCHS}:")
                print(f"  Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f}, Train AUC: {train_auc:.3f}, Train Sensitivity: {train_recall:.3f}, Train Specificity: {train_spec:.3f}, Train MCC: {train_mcc:.3f}")
                print(f"  Val Loss: {val_loss:.3f}, Val Acc: {val_acc:.3f}, Val AUC: {val_auc:.3f}, Val Sensitivity: {val_recall:.3f}, Val Specificity: {val_spec:.3f}, Val MCC: {val_mcc:.3f}")
                if val_auc > best_val_auc: # Or use val_loss < best_val_loss
                    best_val_auc = val_auc
                    torch.save(model.state_dict(), 'best_dti_model_cosine_{}_{}_{}.pth'.format(prot, mol, i))
                    print(f"  New best model saved with Val AUC: {best_val_auc:.3f}")

            print("Training finished.")

            model.load_state_dict(torch.load('best_dti_model_cosine_{}_{}_{}.pth'.format(prot, mol, i)))

            print("Final evaluation on validation set...")
            val_preds, val_probs, val_loss, val_acc, val_precision, val_recall, val_spec, val_mcc, val_auc = evaluate_model(model, val_loader, criterion, DEVICE)
            print(f"  Val Loss: {val_loss:.3f}, Val Acc: {val_acc:.3f}, Val Sensitivity: {val_recall:.3f}, Val Specificity: {val_spec:.3f}, Val MCC: {val_mcc:.3f}, Val AUC: {val_auc:.3f}")
            val_df['Predictions'] = val_preds
            val_df['Predictions_Proba'] = val_probs
            val_df[['UniProt', 'SMILES', 'Label', 'Predictions']].to_csv('val_pred_cosine_{}_{}_{}.csv'.format(prot, mol, i))

            print("Final evaluation on test set (unseen protein)...")
            test_preds, test_probs, test_loss, test_acc, test_precision, test_recall, test_spec, test_mcc, test_auc = evaluate_model(model, test_loader_unseen_prot, criterion, DEVICE)
            print(f"  Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.3f}, Test Sensitivity: {test_recall:.3f}, Test Specificity: {test_spec:.3f}, Test MCC: {test_mcc:.3f}, Test AUC: {test_auc:.3f}")
            test_df_unseen_prot['Predictions'] = test_preds
            test_df_unseen_prot['Predictions_Proba'] = test_probs
            test_df_unseen_prot[['UniProt', 'SMILES', 'Label', 'Predictions']].to_csv('test_pred_unseen_prot_cosine_{}_{}_{}.csv'.format(prot, mol, i))

            print("Final evaluation on test set (unseen ligand)...")
            test_preds, test_probs, test_loss, test_acc, test_precision, test_recall, test_spec, test_mcc, test_auc = evaluate_model(model, test_loader_unseen_lig, criterion, DEVICE)
            print(f"  Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.3f}, Test Sensitivity: {test_recall:.3f}, Test Specificity: {test_spec:.3f}, Test MCC: {test_mcc:.3f}, Test AUC: {test_auc:.3f}")
            test_df_unseen_lig['Predictions'] = test_preds
            test_df_unseen_lig['Predictions_Proba'] = test_probs
            test_df_unseen_lig[['UniProt', 'SMILES', 'Label', 'Predictions']].to_csv('test_pred_unseen_lig_cosine_{}_{}_{}.csv'.format(prot, mol, i))


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
