import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score
from imblearn.metrics import specificity_score
import h5py
from typing import Tuple, Optional, List
import warnings
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import re

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

for i in range (5):

    # Load your datasets
    train_df = pd.read_csv('../../training_set.csv', index_col=0).dropna().reset_index(drop=True)
    val_df = pd.read_csv('../../validation_set.csv', index_col=0).dropna().reset_index(drop=True)
    test_df_unseen_prot = pd.read_csv('../../test_set_unseen_protein.csv', index_col=0).dropna().reset_index(drop=True)
    test_df_unseen_lig = pd.read_csv('../../test_set_unseen_ligands.csv', index_col=0).dropna().reset_index(drop=True)

    val_df[['Predictions']] = np.nan
    test_df_unseen_prot[['Predictions']] = np.nan
    test_df_unseen_lig[['Predictions']] = np.nan

    print(f"Loaded datasets:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val: {len(val_df)} samples")
    print(f"  Test unseen protein: {len(test_df_unseen_prot)} samples")
    print(f"  Test unseen ligands: {len(test_df_unseen_lig)} samples")

    class DTIDatasetWithTokenization(Dataset):
        """
        Dataset that tokenizes sequences on-the-fly instead of using pre-extracted features
        """
        def __init__(self, dataframe, protein_tokenizer, mol_tokenizer, 
                    max_protein_len: int = 1024, max_mol_len: int = 512):
            self.proteins = dataframe['Target Sequence'].tolist()
            self.smiles = dataframe['SMILES'].tolist()
            self.labels = dataframe['Label'].tolist()
            self.protein_tokenizer = protein_tokenizer
            self.mol_tokenizer = mol_tokenizer
            self.max_protein_len = max_protein_len
            self.max_mol_len = max_mol_len

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            protein_seq = self.proteins[idx]
            smiles_str = self.smiles[idx]
            label = self.labels[idx]

            # Tokenize protein sequence (add spaces between amino acids for ProtBert)
            spaced_protein = ' '.join(list(protein_seq))
            protein_tokens = self.protein_tokenizer(
                spaced_protein,
                return_tensors="pt",
                padding=False,  # We'll pad in collate_fn
                truncation=True,
                max_length=self.max_protein_len
            )

            # Tokenize SMILES
            mol_tokens = self.mol_tokenizer(
                smiles_str,
                return_tensors="pt",
                padding=False,  # We'll pad in collate_fn
                truncation=True,
                max_length=self.max_mol_len
            )

            return {
                'protein_input_ids': protein_tokens['input_ids'].squeeze(0),
                'protein_attention_mask': protein_tokens['attention_mask'].squeeze(0),
                'mol_input_ids': mol_tokens['input_ids'].squeeze(0),
                'mol_attention_mask': mol_tokens['attention_mask'].squeeze(0),
                'label': torch.tensor(label, dtype=torch.float32),
                'protein_seq': protein_seq,
                'smiles': smiles_str
            }


    def collate_fn_with_tokenization(batch):
        """Custom collate function for tokenized data"""
        
        # Extract tokenized sequences
        protein_input_ids = [item['protein_input_ids'] for item in batch]
        protein_attention_masks = [item['protein_attention_mask'] for item in batch]
        mol_input_ids = [item['mol_input_ids'] for item in batch]
        mol_attention_masks = [item['mol_attention_mask'] for item in batch]
        
        labels = torch.stack([item['label'] for item in batch])
        protein_seqs = [item['protein_seq'] for item in batch]
        smiles = [item['smiles'] for item in batch]
        
        # Pad sequences
        protein_input_ids_padded = pad_sequence(protein_input_ids, batch_first=True, padding_value=0)
        protein_attention_masks_padded = pad_sequence(protein_attention_masks, batch_first=True, padding_value=0)
        mol_input_ids_padded = pad_sequence(mol_input_ids, batch_first=True, padding_value=0)
        mol_attention_masks_padded = pad_sequence(mol_attention_masks, batch_first=True, padding_value=0)
        
        return {
            'protein_input_ids': protein_input_ids_padded,
            'protein_attention_mask': protein_attention_masks_padded,
            'mol_input_ids': mol_input_ids_padded,
            'mol_attention_mask': mol_attention_masks_padded,
            'labels': labels,
            'protein_seqs': protein_seqs,
            'smiles': smiles
        }


    class CrossAttentionLayer(nn.Module):
        """Cross-attention layer for protein-ligand interaction with masking support"""
        
        def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
            super(CrossAttentionLayer, self).__init__()
            self.d_model = d_model
            self.n_heads = n_heads
            self.d_k = d_model // n_heads
            
            # Query, Key, Value projections
            self.W_q = nn.Linear(d_model, d_model)
            self.W_k = nn.Linear(d_model, d_model)
            self.W_v = nn.Linear(d_model, d_model)
            self.W_o = nn.Linear(d_model, d_model)
            
            self.dropout = nn.Dropout(dropout)
            self.layer_norm = nn.LayerNorm(d_model)
            
        def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                    key_mask: Optional[torch.Tensor] = None, 
                    return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            batch_size, seq_len_q, _ = query.size()
            seq_len_k = key.size(1)
            
            # Residual connection
            residual = query
            
            # Linear projections
            Q = self.W_q(query).view(batch_size, seq_len_q, self.n_heads, self.d_k).transpose(1, 2)
            K = self.W_k(key).view(batch_size, seq_len_k, self.n_heads, self.d_k).transpose(1, 2)
            V = self.W_v(value).view(batch_size, seq_len_k, self.n_heads, self.d_k).transpose(1, 2)
            
            # Scaled dot-product attention
            scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
            
            # Apply key mask if provided
            if key_mask is not None:
                key_mask_expanded = key_mask.unsqueeze(1).unsqueeze(2)
                scores = scores.masked_fill(~key_mask_expanded.bool(), -1e9)
            
            attention_weights = torch.softmax(scores, dim=-1)
            attention_weights = self.dropout(attention_weights)
            
            # Apply attention to values
            context = torch.matmul(attention_weights, V)
            context = context.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
            
            # Output projection and residual connection
            output = self.W_o(context)
            output = self.layer_norm(output + residual)
            
            if return_attention:
                return output, attention_weights
            return output, None


    class DTIProfilerWithFineTuning(nn.Module):
        """
        DTI model with unfrozen feature extractors for end-to-end training
        """
        def __init__(self, d_model: int = 512, n_heads: int = 4, 
                    freeze_protein_encoder: bool = False,
                    freeze_mol_encoder: bool = False):
            super(DTIProfilerWithFineTuning, self).__init__()

            self.d_model = d_model
            self.freeze_protein_encoder = freeze_protein_encoder
            self.freeze_mol_encoder = freeze_mol_encoder

            # Load pre-trained models
            print("Loading ProtBert...")
            self.protein_tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
            self.protein_encoder = AutoModel.from_pretrained("Rostlab/prot_bert")
            self.protein_dim = self.protein_encoder.config.hidden_size  # 1024 for ProtBert
        
            print("Loading MolFormer...")
            self.mol_tokenizer = AutoTokenizer.from_pretrained("ibm/MolFormer-XL-both-10pct", trust_remote_code=True)
            self.mol_encoder = AutoModel.from_pretrained("ibm/MolFormer-XL-both-10pct", trust_remote_code=True)
            self.mol_dim = self.mol_encoder.config.hidden_size  # 768 for MolFormer
            
            # Freeze encoders if requested
            if self.freeze_protein_encoder:
                print("Freezing protein encoder weights...")
                for param in self.protein_encoder.parameters():
                    param.requires_grad = False
            else:
                print("Protein encoder weights are UNFROZEN and trainable")
                
            if self.freeze_mol_encoder:
                print("Freezing molecular encoder weights...")
                for param in self.mol_encoder.parameters():
                    param.requires_grad = False
            else:
                print("Molecular encoder weights are UNFROZEN and trainable")
            
            # Input projections to common dimension
            self.protein_projector = nn.Linear(self.protein_dim, d_model)
            self.mol_projector = nn.Linear(self.mol_dim, d_model)
            
            # Self-attention layers
            self.protein_self_attention = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model*2, 
                dropout=0.1, batch_first=True
            )
            self.mol_self_attention = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model*2, 
                dropout=0.1, batch_first=True
            )

            # Cross-attention layers
            self.protein_to_mol_attention = CrossAttentionLayer(d_model, n_heads)
            self.mol_to_protein_attention = CrossAttentionLayer(d_model, n_heads)
            
            # Final classifier
            self.classifier = nn.Sequential(
                nn.Linear(d_model * 2, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )

        def forward(self, protein_input_ids, protein_attention_mask,
                    mol_input_ids, mol_attention_mask, return_attention: bool = False):
            
            # Encode protein sequence
            if not self.freeze_protein_encoder:
                protein_outputs = self.protein_encoder(
                    input_ids=protein_input_ids,
                    attention_mask=protein_attention_mask
                )
            else:
                with torch.no_grad():
                    protein_outputs = self.protein_encoder(
                        input_ids=protein_input_ids,
                        attention_mask=protein_attention_mask
                    )
            
            protein_embeddings = protein_outputs.last_hidden_state
            
            # Encode molecular sequence
            if not self.freeze_mol_encoder:
                mol_outputs = self.mol_encoder(
                    input_ids=mol_input_ids,
                    attention_mask=mol_attention_mask
                    )
            else:
                with torch.no_grad():
                    mol_outputs = self.mol_encoder(
                        input_ids=mol_input_ids,
                        attention_mask=mol_attention_mask
                        )

            mol_embeddings = mol_outputs.last_hidden_state
            
            # Project to common dimension
            protein_proj = self.protein_projector(protein_embeddings)
            mol_proj = self.mol_projector(mol_embeddings)
            
            # Self-attention with masking
            protein_self_att = self.protein_self_attention(
                protein_proj, src_key_padding_mask=~protein_attention_mask.bool()
            )
            mol_self_att = self.mol_self_attention(
                mol_proj, src_key_padding_mask=~mol_attention_mask.bool()
            )
            
            # Cross-attention
            protein_cross_att, prot_to_mol_att = self.protein_to_mol_attention(
                protein_self_att, mol_self_att, mol_self_att, 
                key_mask=mol_attention_mask, return_attention=return_attention
            )
            mol_cross_att, mol_to_prot_att = self.mol_to_protein_attention(
                mol_self_att, protein_self_att, protein_self_att, 
                key_mask=protein_attention_mask, return_attention=return_attention
            )
            
            # Masked pooling
            protein_lengths = protein_attention_mask.sum(dim=1, keepdim=True).float()
            protein_masked = protein_cross_att * protein_attention_mask.unsqueeze(-1).float()
            protein_pooled = protein_masked.sum(dim=1) / protein_lengths
            
            mol_lengths = mol_attention_mask.sum(dim=1, keepdim=True).float()
            mol_masked = mol_cross_att * mol_attention_mask.unsqueeze(-1).float()
            mol_pooled = mol_masked.sum(dim=1) / mol_lengths
            
            # Combine features
            combined_features = torch.cat([protein_pooled, mol_pooled], dim=1)
            
            # Classification
            logits = self.classifier(combined_features)
            
            if return_attention:
                attention_weights = {
                    'protein_to_mol': prot_to_mol_att,
                    'mol_to_protein': mol_to_prot_att
                }
                return logits, attention_weights
            
            return logits


    def create_datasets_with_tokenization(train_df, val_df, test_df_unseen_prot, test_df_unseen_lig):
        """Create datasets with tokenization instead of pre-extracted features"""
        
        # Load tokenizers
        protein_tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        
        mol_tokenizer = AutoTokenizer.from_pretrained("ibm/MolFormer-XL-both-10pct", trust_remote_code=True)
        
        # Create datasets
        train_dataset = DTIDatasetWithTokenization(
            train_df, protein_tokenizer, mol_tokenizer, 
            max_protein_len=1024, max_mol_len=512
        )
        val_dataset = DTIDatasetWithTokenization(
            val_df, protein_tokenizer, mol_tokenizer, 
            max_protein_len=1024, max_mol_len=512
        )
        test_unseen_prot_dataset = DTIDatasetWithTokenization(
            test_df_unseen_prot, protein_tokenizer, mol_tokenizer, 
            max_protein_len=1024, max_mol_len=512
        )
        test_unseen_lig_dataset = DTIDatasetWithTokenization(
            test_df_unseen_lig, protein_tokenizer, mol_tokenizer,
            max_protein_len=1024, max_mol_len=512
        )

        return train_dataset, val_dataset, test_unseen_prot_dataset, test_unseen_lig_dataset


    def train_model_end_to_end(model, dataloader, criterion, optimizer, device):
        """Training function for end-to-end model"""
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        for batch in tqdm(dataloader, desc="Training"):
            protein_input_ids = batch['protein_input_ids'].to(device)
            protein_attention_mask = batch['protein_attention_mask'].to(device)
            mol_input_ids = batch['mol_input_ids'].to(device)
            mol_attention_mask = batch['mol_attention_mask'].to(device)
            labels = batch['labels'].unsqueeze(1).to(device)

            optimizer.zero_grad()
            logits = model(protein_input_ids, protein_attention_mask,
                        mol_input_ids, mol_attention_mask)
            logits = torch.clamp(logits, min=1e-7, max=1-1e-7)  # Numerical stability
            loss = criterion(logits, labels)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            total_loss += loss.item()
            preds = (logits > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        specificity = specificity_score(all_labels, all_preds, average='binary', pos_label=1)

        try:
            auc = roc_auc_score(all_labels, all_preds)
            mcc = matthews_corrcoef(all_labels, all_preds)
        except ValueError:
            auc = 0.5
            mcc = 0

        return avg_loss, accuracy, precision, recall, specificity, mcc, auc


    def evaluate_model_end_to_end(model, dataloader, criterion, device):
        """Evaluation function for end-to-end model"""
        model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                protein_input_ids = batch['protein_input_ids'].to(device)
                protein_attention_mask = batch['protein_attention_mask'].to(device)
                mol_input_ids = batch['mol_input_ids'].to(device)
                mol_attention_mask = batch['mol_attention_mask'].to(device)
                labels = batch['labels'].unsqueeze(1).to(device)
                
                logits = model(protein_input_ids, protein_attention_mask,
                            mol_input_ids, mol_attention_mask)
                logits = torch.clamp(logits, min=1e-7, max=1-1e-7)
                loss = criterion(logits, labels)

                total_loss += loss.item()
                preds = (logits > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        specificity = specificity_score(all_labels, all_preds, average='binary', pos_label=1)
        
        try:
            auc = roc_auc_score(all_labels, all_preds)
            mcc = matthews_corrcoef(all_labels, all_preds)
        except ValueError:
            auc = 0.5
            mcc = 0

        return all_preds, avg_loss, accuracy, precision, recall, specificity, mcc, auc


    # ============================================================================
    # MAIN EXECUTION
    # ============================================================================

    if __name__ == "__main__":
        print("Creating datasets with tokenization...")
        train_dataset, val_dataset, test_unseen_prot_dataset, test_unseen_lig_dataset = create_datasets_with_tokenization(
            train_df, val_df, test_df_unseen_prot, test_df_unseen_lig
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, 
                                collate_fn=collate_fn_with_tokenization)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, 
                            collate_fn=collate_fn_with_tokenization)
        test_unseen_prot_loader = DataLoader(test_unseen_prot_dataset, batch_size=8, shuffle=False, 
                                collate_fn=collate_fn_with_tokenization)
        test_unseen_lig_loader = DataLoader(test_unseen_lig_dataset, batch_size=8, shuffle=False,
                                collate_fn=collate_fn_with_tokenization)

        print("Creating model with unfrozen weights...")
        model = DTIProfilerWithFineTuning(
            d_model=512,
            n_heads=4,
            freeze_protein_encoder=True,  # SET TO TRUE TO FREEZE
            freeze_mol_encoder=True       # SET TO TRUE TO FREEZE
        ).to(DEVICE)
        
        # Count trainable parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"\nModel Parameter Summary:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Frozen parameters: {frozen_params:,}")
        print(f"  Percentage trainable: {100 * trainable_params / total_params:.1f}%")
        
        # Use a lower learning rate for fine-tuning
        optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)
        criterion = nn.BCELoss()
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.1, patience=3
        )
        
        best_val_auc = 0
        EPOCHS = 10  # Fewer epochs for fine-tuning
        
        print(f"\nStarting end-to-end training with unfrozen weights...")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        
        for epoch in range(EPOCHS):
            print(f"\nEpoch {epoch+1}/{EPOCHS}")
            print("-" * 50)
            
            # Training
            train_loss, train_acc, train_precision, train_recall, train_spec, train_mcc, train_auc = train_model_end_to_end(
                model, train_loader, criterion, optimizer, DEVICE
            )
            
            # Validation
            val_preds, val_loss, val_acc, val_precision, val_recall, val_spec, val_mcc, val_auc = evaluate_model_end_to_end(
                model, val_loader, criterion, DEVICE
            )
            
            # Learning rate scheduling
            scheduler.step(val_mcc)
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"Epoch {epoch+1}/{EPOCHS}:")
            print(f"  Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f}, Train AUC: {train_auc:.3f}, Train Sensitivity: {train_recall:.3f}, Train Specificity: {train_spec:.3f}, Train MCC: {train_mcc:.3f}")
            print(f"  Val Loss: {val_loss:.3f}, Val Acc: {val_acc:.3f}, Val AUC: {val_auc:.3f}, Val Sensitivity: {val_recall:.3f}, Val Specificity: {val_spec:.3f}, Val MCC: {val_mcc:.3f}")
            print(f"Learning Rate: {current_lr:.2e}")
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save(model.state_dict(), 'best_dti_end_to_end_unfrozen_none_ProtBert_MolFormer_{}.pth'.format(i))
                print(f"*** New best model saved! Val AUC: {best_val_auc:.4f} ***")
        
        print(f"\nTraining completed! Best Val AUC: {best_val_auc:.4f}")

        model.load_state_dict(torch.load('best_dti_end_to_end_unfrozen_none_ProtBert_MolFormer_{}.pth'.format(i)))
        
        print("\nFinal evaluation on validation set...")
        val_preds, val_loss, val_acc, val_precision, val_recall, val_spec, val_mcc, val_auc = evaluate_model_end_to_end(model, val_loader, criterion, DEVICE)
        print(f"  Val Loss: {val_loss:.3f}, Val Acc: {val_acc:.3f}, Val Sensitivity: {val_recall:.3f}, Val Specificity: {val_spec:.3f}, Val MCC: {val_mcc:.3f}, Val AUC: {val_auc:.3f}")
        val_df['Predictions'] = val_preds
        val_df.to_csv('val_pred_fine_tune_none_ProtBert_MolFormer_{}.csv'.format(i))

        print("\nFinal evaluation on test set (unseen protein)...")
        test_preds, test_loss, test_acc, test_precision, test_recall, test_spec, test_mcc, test_auc = evaluate_model_end_to_end(model, test_unseen_prot_loader, criterion, DEVICE)       
        print(f"  Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.3f}, Test Sensitivity: {test_recall:.3f}, Test Specificity: {test_spec:.3f}, Test MCC: {test_mcc:.3f}, Test AUC: {test_auc:.3f}")
        test_df_unseen_prot['Predictions'] = test_preds
        test_df_unseen_prot.to_csv('test_pred_unseen_prot_fine_tune_none_ProtBert_MolFormer_{}.csv'.format(i))

        print("\nFinal evaluation on test set (unseen ligand)...")
        test_preds, test_loss, test_acc, test_precision, test_recall, test_spec, test_mcc, test_auc = evaluate_model_end_to_end(model, test_unseen_lig_loader, criterion, DEVICE)
        print(f"  Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.3f}, Test Sensitivity: {test_recall:.3f}, Test Specificity: {test_spec:.3f}, Test MCC: {test_mcc:.3f}, Test AUC: {test_auc:.3f}")
        test_df_unseen_lig['Predictions'] = test_preds
        test_df_unseen_lig.to_csv('test_pred_unseen_lig_fine_tune_none_ProtBert_MolFormer_{}.csv'.format(i))

'''        
        def cal_metrics(df):
                pred = list(df['Predictions'].apply(lambda x: [int(el) for el in x.strip("[]").split(".")[0]]))
                label = list(df['Label'])
                tn, fp, fn, tp = confusion_matrix(label, pred, labels=[0,1]).ravel()
                acc = np.round(accuracy_score(label, pred), 3)
                mcc = np.round(matthews_corrcoef(label, pred), 3)
                roc = np.round(roc_auc_score(label, pred), 3)
                sen = np.round(tp/(tp + fn), 3)
                spec = np.round(tn/(tn + fp), 3)
                return acc, spec, sen, mcc, roc

        accuracy, specificity, sensitivity, mcc, roc = cal_metrics(val_df)
        print('Val', accuracy, specificity, sensitivity, mcc, roc)

        accuracy, specificity, sensitivity, mcc, roc = cal_metrics(test_df_unseen_prot)
        print('Unseen protein', accuracy, specificity, sensitivity, mcc, roc)

        accuracy, specificity, sensitivity, mcc, roc = cal_metrics(test_df_unseen_lig)
        print('Unseen ligand', accuracy, specificity, sensitivity, mcc, roc)

'''
