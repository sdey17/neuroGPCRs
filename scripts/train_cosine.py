"""Training script for Cosine Similarity DTI model."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import yaml
import argparse
from pathlib import Path

from src.models import DTIProfilerCosine
from src.utils.data_loader import load_datasets, extract_features, DTIDataset, create_dataloaders
from src.utils.training import train_model, evaluate_model
from src.utils.metrics import print_metrics, evaluate_predictions


def main(config_path: str = "config.yaml", protein_feat: str = None, mol_feat: str = None):
    """Main training function."""

    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and config['device']['use_cuda'] else "cpu")
    print(f"Using device: {device}")

    # Set random seed
    torch.manual_seed(config['seed'])

    # Load datasets
    print("\nLoading datasets...")
    train_df, val_df, test_unseen_prot_df, test_unseen_lig_df = load_datasets(
        data_dir=config['data']['data_dir'],
        train_file=config['data']['train_file'],
        val_file=config['data']['val_file'],
        test_unseen_prot_file=config['data']['test_unseen_protein'],
        test_unseen_lig_file=config['data']['test_unseen_ligand']
    )

    # Load features
    print("\nChecking feature files...")
    protein_feat_path = Path(protein_feat)
    mol_feat_path = Path(mol_feat)

    if not protein_feat_path.exists() or not mol_feat_path.exists():
        print("Feature files not found. Running embedding generation...")
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from generate_embeddings import generate_protein_embeddings, generate_molecule_embeddings

        all_dfs = [train_df, val_df, test_unseen_prot_df, test_unseen_lig_df]
        all_sequences = sorted(set(seq for df in all_dfs for seq in df['Target Sequence'].dropna().unique()))
        all_smiles = sorted(set(smi for df in all_dfs for smi in df['SMILES'].dropna().unique()))

        if not protein_feat_path.exists():
            print(f"  Generating protein embeddings -> {protein_feat}")
            generate_protein_embeddings(all_sequences, output_file=protein_feat)

        if not mol_feat_path.exists():
            print(f"  Generating molecule embeddings -> {mol_feat}")
            generate_molecule_embeddings(all_smiles, output_file=mol_feat)

    print("\nLoading pre-computed features...")

    prot_feat, mol_feat, target_dim, drug_dim = extract_features(
        prot_feat_file=protein_feat,
        mol_feat_file=mol_feat,
        protein_encoder=config['model']['protein_encoder'],
        molecule_encoder=config['model']['molecule_encoder']
    )

    # Create datasets
    train_dataset = DTIDataset(train_df, prot_feat, mol_feat)
    val_dataset = DTIDataset(val_df, prot_feat, mol_feat)
    test_unseen_prot_dataset = DTIDataset(test_unseen_prot_df, prot_feat, mol_feat)
    test_unseen_lig_dataset = DTIDataset(test_unseen_lig_df, prot_feat, mol_feat)

    # Create dataloaders
    train_loader, val_loader, test_unseen_prot_loader, test_unseen_lig_loader = create_dataloaders(
        train_dataset, val_dataset, test_unseen_prot_dataset, test_unseen_lig_dataset,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers']
    )

    # Create model
    print("\nInitializing Cosine Similarity Model...")
    model = DTIProfilerCosine(
        drug_dim=drug_dim,
        target_dim=target_dim,
        latent_dim=config['model']['latent_dim']
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Setup optimizer and criterion
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    criterion = nn.BCELoss()

    # Training
    print("\nStarting Training...")

    results_dir = Path(config['output']['results_dir'])
    results_dir.mkdir(exist_ok=True)

    save_path = results_dir / f"cosine_{config['model']['protein_encoder']}_{config['model']['molecule_encoder']}.pth"

    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=config['training']['num_epochs'],
        save_path=str(save_path),
        verbose=True
    )

    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(results_dir / f"history_cosine.csv", index=False)

    # Final evaluation
    print("\nFinal Evaluation")

    # Validation set
    val_preds, val_probs, val_loss, val_acc, val_prec, val_rec, val_spec, val_mcc, val_auc = evaluate_model(
        model, val_loader, criterion, device, verbose=False
    )
    val_df['Predictions'] = val_preds
    val_df['Predictions_Proba'] = val_probs
    val_df.to_csv(results_dir / "val_predictions_cosine.csv")

    print("\nValidation Set:")
    val_metrics = evaluate_predictions(val_df)
    print_metrics(val_metrics)

    # Test set (unseen protein)
    test_preds, test_probs, test_loss, test_acc, test_prec, test_rec, test_spec, test_mcc, test_auc = evaluate_model(
        model, test_unseen_prot_loader, criterion, device, verbose=False
    )
    test_unseen_prot_df['Predictions'] = test_preds
    test_unseen_prot_df['Predictions_Proba'] = test_probs
    test_unseen_prot_df.to_csv(results_dir / "test_unseen_protein_predictions_cosine.csv")

    print("\nTest Set (Unseen Protein):")
    test_prot_metrics = evaluate_predictions(test_unseen_prot_df)
    print_metrics(test_prot_metrics)

    # Test set (unseen ligand)
    test_preds, test_probs, test_loss, test_acc, test_prec, test_rec, test_spec, test_mcc, test_auc = evaluate_model(
        model, test_unseen_lig_loader, criterion, device, verbose=False
    )
    test_unseen_lig_df['Predictions'] = test_preds
    test_unseen_lig_df['Predictions_Proba'] = test_probs
    test_unseen_lig_df.to_csv(results_dir / "test_unseen_ligand_predictions_cosine.csv")

    print("\nTest Set (Unseen Ligand):")
    test_lig_metrics = evaluate_predictions(test_unseen_lig_df)
    print_metrics(test_lig_metrics)

    print("\nTraining Complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Cosine Similarity DTI model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--protein_feat", type=str, default="ProtBert_features.h5", help="Path to protein features h5 file")
    parser.add_argument("--mol_feat", type=str, default="MolFormer_features.h5", help="Path to molecule features h5 file")

    args = parser.parse_args()
    main(args.config, args.protein_feat, args.mol_feat)
