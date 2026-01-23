"""Quick start example for training a DTI prediction model."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim

from src.models import DTIProfilerTransformer
from src.utils.data_loader import load_datasets, extract_features, DTIDataset, create_dataloaders
from src.utils.training import train_model

def main():
    """Quick start training example."""

    print("="*60)
    print("neuroGPCRs Quick Start Example")
    print("="*60)

    # Configuration
    config = {
        'data_dir': 'data',
        'protein_feat': 'path/to/ProtBert_features.h5',  # Update this
        'mol_feat': 'path/to/MolFormer_features.h5',      # Update this
        'protein_encoder': 'ProtBert',
        'molecule_encoder': 'MolFormer',
        'batch_size': 32,
        'learning_rate': 0.0001,
        'num_epochs': 10,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    print(f"\nUsing device: {config['device']}")

    # Load data
    print("\nLoading datasets...")
    train_df, val_df, test_unseen_prot_df, test_unseen_lig_df = load_datasets(
        data_dir=config['data_dir']
    )

    # Load features
    print("\nLoading pre-computed features...")
    try:
        prot_feat, mol_feat, target_dim, drug_dim = extract_features(
            prot_feat_file=config['protein_feat'],
            mol_feat_file=config['mol_feat'],
            protein_encoder=config['protein_encoder'],
            molecule_encoder=config['molecule_encoder']
        )
    except FileNotFoundError:
        print("\nERROR: Feature files not found!")
        print("Please update the paths in this script to point to your feature files.")
        print("You'll need to generate embeddings using ProtBert and MolFormer first.")
        return

    # Create datasets and dataloaders
    print("\nCreating datasets...")
    train_dataset = DTIDataset(train_df, prot_feat, mol_feat)
    val_dataset = DTIDataset(val_df, prot_feat, mol_feat)

    train_loader, val_loader, _, _ = create_dataloaders(
        train_dataset, val_dataset, val_dataset, val_dataset,
        batch_size=config['batch_size']
    )

    # Create model
    print("\nInitializing model...")
    device = torch.device(config['device'])
    model = DTIProfilerTransformer(
        drug_dim=drug_dim,
        target_dim=target_dim,
        latent_dim=1024,
        n_heads=4,
        n_layers=2,
        dropout=0.1
    ).to(device)

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup training
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
    criterion = nn.BCELoss()

    # Train
    print("\nStarting training...")
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=config['num_epochs'],
        save_path='quick_start_model.pth',
        verbose=True
    )

    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    print(f"\nModel saved to: quick_start_model.pth")
    print(f"Best validation AUC: {max([h['val_auc'] for h in history]):.4f}")

if __name__ == "__main__":
    main()
