"""Unified training script for Cross-Attention DTI model with flexible fine-tuning options."""

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
from transformers import AutoTokenizer

from src.models.cross_attention_finetune import DTIFineTuneCrossAttention
from src.utils.finetune_data_loader import (
    load_datasets_for_finetuning,
    DTIFineTuneDataset,
    create_finetune_dataloaders
)
from src.utils.finetune_training import train_model_finetune, evaluate_model_finetune
from src.utils.metrics import print_metrics, evaluate_predictions


def main(args):
    """Main training function with flexible freezing options."""

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override config with command line arguments
    freeze_protein = args.freeze_protein
    freeze_molecule = args.freeze_molecule

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and config['device']['use_cuda'] else "cpu")
    print(f"Using device: {device}")

    # Set random seed
    torch.manual_seed(config['seed'])

    # Load tokenizers
    print("\n" + "="*60)
    print("Loading tokenizers...")
    print("="*60)

    protein_model_name = config['finetune']['protein_model']
    molecule_model_name = config['finetune']['molecule_model']

    print(f"Protein tokenizer: {protein_model_name}")
    protein_tokenizer = AutoTokenizer.from_pretrained(protein_model_name, do_lower_case=False)

    print(f"Molecule tokenizer: {molecule_model_name}")
    molecule_tokenizer = AutoTokenizer.from_pretrained(molecule_model_name, trust_remote_code=True)

    # Load datasets
    print("\n" + "="*60)
    print("Loading datasets...")
    print("="*60)
    train_df, val_df, test_unseen_prot_df, test_unseen_lig_df = load_datasets_for_finetuning(
        data_dir=config['data']['data_dir'],
        train_file=config['data']['train_file'],
        val_file=config['data']['val_file'],
        test_unseen_prot_file=config['data']['test_unseen_protein'],
        test_unseen_lig_file=config['data']['test_unseen_ligand']
    )

    # Create datasets
    train_dataset = DTIFineTuneDataset(
        train_df,
        protein_tokenizer,
        molecule_tokenizer,
        max_protein_len=config['finetune']['max_protein_len'],
        max_molecule_len=config['finetune']['max_molecule_len']
    )
    val_dataset = DTIFineTuneDataset(
        val_df,
        protein_tokenizer,
        molecule_tokenizer,
        max_protein_len=config['finetune']['max_protein_len'],
        max_molecule_len=config['finetune']['max_molecule_len']
    )
    test_unseen_prot_dataset = DTIFineTuneDataset(
        test_unseen_prot_df,
        protein_tokenizer,
        molecule_tokenizer,
        max_protein_len=config['finetune']['max_protein_len'],
        max_molecule_len=config['finetune']['max_molecule_len']
    )
    test_unseen_lig_dataset = DTIFineTuneDataset(
        test_unseen_lig_df,
        protein_tokenizer,
        molecule_tokenizer,
        max_protein_len=config['finetune']['max_protein_len'],
        max_molecule_len=config['finetune']['max_molecule_len']
    )

    # Create dataloaders
    train_loader, val_loader, test_unseen_prot_loader, test_unseen_lig_loader = create_finetune_dataloaders(
        train_dataset, val_dataset, test_unseen_prot_dataset, test_unseen_lig_dataset,
        batch_size=config['finetune']['batch_size'],
        num_workers=config['training']['num_workers']
    )

    # Create model with specified freezing configuration
    print("\n" + "="*60)
    print("Initializing Cross-Attention Model...")
    print(f"Protein Encoder: {'FROZEN' if freeze_protein else 'TRAINABLE'}")
    print(f"Molecule Encoder: {'FROZEN' if freeze_molecule else 'TRAINABLE'}")
    print("="*60)

    # Determine if both are frozen (use single freeze_encoders flag)
    freeze_both = freeze_protein and freeze_molecule

    model = DTIFineTuneCrossAttention(
        protein_model_name=protein_model_name,
        molecule_model_name=molecule_model_name,
        d_model=config['finetune']['d_model'],
        n_heads=config['finetune']['n_heads'],
        dropout=config['finetune']['dropout'],
        freeze_encoders=freeze_both
    ).to(device)

    # Apply individual freezing if not both frozen
    if not freeze_both:
        if freeze_protein:
            print("Freezing protein encoder weights...")
            for param in model.protein_encoder.parameters():
                param.requires_grad = False
        if freeze_molecule:
            print("Freezing molecule encoder weights...")
            for param in model.molecule_encoder.parameters():
                param.requires_grad = False

    # Print model info
    params_dict = model.get_num_params()
    print("\nModel Parameters:")
    print(f"  Protein Encoder: {params_dict['protein_encoder']:,}")
    print(f"  Molecule Encoder: {params_dict['molecule_encoder']:,}")
    print(f"  Self-Attention: {params_dict['self_attention']:,}")
    print(f"  Cross-Attention: {params_dict['cross_attention']:,}")
    print(f"  Classifier: {params_dict['classifier']:,}")
    print(f"  Total: {params_dict['total']:,}")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable: {trainable_params:,}")

    # Setup optimizer with different learning rates based on freezing configuration
    param_groups = []

    if not freeze_protein:
        param_groups.append({
            'params': model.protein_encoder.parameters(),
            'lr': config['finetune']['encoder_lr']
        })

    if not freeze_molecule:
        param_groups.append({
            'params': model.molecule_encoder.parameters(),
            'lr': config['finetune']['encoder_lr']
        })

    # Always train task-specific layers
    param_groups.extend([
        {'params': model.protein_projector.parameters()},
        {'params': model.molecule_projector.parameters()},
        {'params': model.protein_self_attention.parameters()},
        {'params': model.molecule_self_attention.parameters()},
        {'params': model.protein_to_mol_attention.parameters()},
        {'params': model.mol_to_protein_attention.parameters()},
        {'params': model.classifier.parameters()},
    ])

    optimizer = optim.AdamW(
        param_groups,
        lr=config['finetune']['learning_rate'],
        weight_decay=config['finetune']['weight_decay']
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=2,
        verbose=True
    )

    criterion = nn.BCELoss()

    # Training
    print("\n" + "="*60)
    print("Starting Training...")
    print("="*60)

    results_dir = Path(config['output']['results_dir'])
    results_dir.mkdir(exist_ok=True)

    # Create descriptive model name based on freezing configuration
    freeze_config = f"prot{'_frozen' if freeze_protein else '_train'}_mol{'_frozen' if freeze_molecule else '_train'}"
    save_path = results_dir / f"cross_attention_{freeze_config}.pth"

    model, history = train_model_finetune(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=config['finetune']['num_epochs'],
        save_path=str(save_path),
        verbose=True,
        early_stopping_patience=config['finetune']['early_stopping_patience']
    )

    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(results_dir / f"history_{freeze_config}.csv", index=False)

    # Final evaluation
    print("\n" + "="*60)
    print("Final Evaluation")
    print("="*60)

    # Validation set
    val_preds, val_probs, val_loss, val_acc, val_prec, val_rec, val_spec, val_mcc, val_auc = evaluate_model_finetune(
        model, val_loader, criterion, device, verbose=False
    )
    val_df['Predictions'] = val_preds
    val_df['Predictions_Proba'] = val_probs
    val_df.to_csv(results_dir / f"val_predictions_{freeze_config}.csv")

    print("\nValidation Set:")
    val_metrics = evaluate_predictions(val_df)
    print_metrics(val_metrics)

    # Test set (unseen protein)
    test_preds, test_probs, test_loss, test_acc, test_prec, test_rec, test_spec, test_mcc, test_auc = evaluate_model_finetune(
        model, test_unseen_prot_loader, criterion, device, verbose=False
    )
    test_unseen_prot_df['Predictions'] = test_preds
    test_unseen_prot_df['Predictions_Proba'] = test_probs
    test_unseen_prot_df.to_csv(results_dir / f"test_unseen_protein_{freeze_config}.csv")

    print("\nTest Set (Unseen Protein):")
    test_prot_metrics = evaluate_predictions(test_unseen_prot_df)
    print_metrics(test_prot_metrics)

    # Test set (unseen ligand)
    test_preds, test_probs, test_loss, test_acc, test_prec, test_rec, test_spec, test_mcc, test_auc = evaluate_model_finetune(
        model, test_unseen_lig_loader, criterion, device, verbose=False
    )
    test_unseen_lig_df['Predictions'] = test_preds
    test_unseen_lig_df['Predictions_Proba'] = test_probs
    test_unseen_lig_df.to_csv(results_dir / f"test_unseen_ligand_{freeze_config}.csv")

    print("\nTest Set (Unseen Ligand):")
    test_lig_metrics = evaluate_predictions(test_unseen_lig_df)
    print_metrics(test_lig_metrics)

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Model saved to: {save_path}")
    print(f"Configuration: {freeze_config}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Cross-Attention DTI model with flexible encoder freezing options"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--freeze_protein",
        action="store_true",
        help="Freeze protein encoder weights (ProtBert)"
    )
    parser.add_argument(
        "--freeze_molecule",
        action="store_true",
        help="Freeze molecule encoder weights (MolFormer)"
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("CROSS-ATTENTION DTI MODEL - UNIFIED TRAINING")
    print("="*60)
    print(f"Configuration:")
    print(f"  Freeze Protein Encoder: {args.freeze_protein}")
    print(f"  Freeze Molecule Encoder: {args.freeze_molecule}")
    print("="*60)

    main(args)
