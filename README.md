# neuroGPCRs: Evaluation of Deep Learning Architectures for Predicting GPCR-Mediated Neurotoxicity

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Abstract

G protein-coupled receptors (GPCRs) are therapeutic targets for over 30% of approved drugs, yet specific GPCR subtypes act as molecular initiating events in several neurotoxic adverse outcome pathways. This repository implements **seven deep learning architectures** for predicting GPCR-ligand binding interactions, systematically evaluating the impact of encoder fine-tuning strategies.

### Models Implemented

1. **CosSim** - Dual-Projection Cosine Similarity Networks
2. **Transformers** - Transformer Encoder Networks
3. **CA-Base** - Cross-Attention with frozen protein and ligand encoders
4. **CA-Lig** - Cross-Attention with fine-tuned ligand encoder
5. **CA-Prot** - Cross-Attention with fine-tuned protein encoder
6. **CA-Full** - Cross-Attention with both encoders fine-tuned
7. **XGB** - XGBoost baseline

All models leverage pre-trained language models (ProtBert for proteins, MolFormer for ligands) and are evaluated on multiple challenging scenarios including unseen proteins and unseen ligands.


**Key Finding**: CA-Base (frozen encoders) achieves the best performance, suggesting that task-specific training of pre-trained encoders may lead to overfitting for this dataset.

## Dataset

(Will be made available following the manuscript publication)
- **Training Set**: 119 GPCRs with binding data
- **Validation Set**: Random split from training distribution
- **Test Set (Unseen Proteins)**: 9 GPCRs not seen during training
- **Test Set (Unseen Ligands)**: Ligands not seen during training

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/neuroGPCRs.git
cd neuroGPCRs

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

Place your data files in the `data/` directory:
- `training_set.csv`
- `validation_set.csv`
- `test_set_unseen_protein.csv`
- `test_set_unseen_ligands.csv`

**Data Format:**
Each CSV should contain the following columns:
- `UniProt`: UniProt ID of the protein
- `Target Sequence`: Amino acid sequence of the protein
- `SMILES`: SMILES string of the molecule
- `Label`: Binary label (0 or 1) indicating binding

### Pre-computing Embeddings

CosSim, Transformer, and XGBoost models require pre-computed embeddings. These are generated automatically if the feature files are missing when you run the training scripts. You can also generate them explicitly:

```bash
python scripts/generate_embeddings.py \
    --data_files data/training_set.csv data/validation_set.csv \
                data/test_set_unseen_protein.csv data/test_set_unseen_ligands.csv
```

**Note:** Cross-attention models encode sequences on-the-fly and don't need pre-computed embeddings.

### Training Models

#### 1. Cosine Similarity Model
```bash
# Embeddings are generated automatically if not already present
python scripts/train_cosine.py --config config.yaml
```

#### 2. Transformer Encoder Model
```bash
python scripts/train_transformer.py --config config.yaml
```

#### 3. XGBoost Model
```bash
python scripts/train_xgb.py --config config.yaml
```

#### 4. Cross-Attention Models (Unified Script)

The repository provides a **unified training script** that supports all four cross-attention variants:

```bash
# CA-Base: Both encoders frozen (best performance)
python scripts/train_cross_attention_unified.py \
    --freeze_protein --freeze_molecule

# CA-Lig: Fine-tune ligand encoder only
python scripts/train_cross_attention_unified.py \
    --freeze_protein

# CA-Prot: Fine-tune protein encoder only
python scripts/train_cross_attention_unified.py \
    --freeze_molecule

# CA-Full: Fine-tune both encoders
python scripts/train_cross_attention_unified.py
```

**Configuration Options:**
- `--freeze_protein`: Freeze ProtBert encoder weights
- `--freeze_molecule`: Freeze MolFormer encoder weights
- `--config`: Path to configuration file (default: `config.yaml`)

### Making Predictions (Inference)

Once you have a trained model, you can predict GPCR-ligand interactions for new compounds using the `predict_interactions.py` script:

#### Predict for a single SMILES:
```bash
python scripts/predict_interactions.py \
    --smiles "CCOCc1sc(NC(=O)c2ccco2)nc1-c1ccccc1" \
    --model results/cross_attention_prot_frozen_mol_frozen_run1.pth \
    --output my_predictions.csv \
    --top_k 10
```

#### Predict for multiple SMILES from a file:
```bash
# Create a file with SMILES (one per line)
cat > compounds.txt << EOF
CCOCc1sc(NC(=O)c2ccco2)nc1-c1ccccc1
COc1cc(N(C)CCN(C)C)c2nc(C(=O)Nc3ccc(N4CCOCC4)cc3)cc(O)c2c1
COc1ccccc1OCCNCCCc1c[nH]c2ccccc12
EOF

# Run predictions
python scripts/predict_interactions.py \
    --smiles_file compounds.txt \
    --model results/cross_attention_prot_frozen_mol_frozen_run1.pth \
    --output batch_predictions.csv \
    --threshold 0.5
```

**Prediction Options:**
- `--smiles`: Single SMILES string to predict
- `--smiles_file`: File containing SMILES strings (one per line)
- `--model`: Path to trained model checkpoint (.pth file)
- `--data`: Path to training CSV (to extract GPCR sequences, default: `data/training_set.csv`)
- `--output`: Output CSV file for predictions
- `--top_k`: Only save top K predictions per compound
- `--threshold`: Only save predictions above this probability threshold
- `--batch_size`: Batch size for inference (default: 16)

**Output Format:**

The predictions are saved as a CSV file with the following columns:
- `SMILES`: Input compound SMILES
- `UniProt`: UniProt ID of the GPCR
- `Target_Sequence`: Amino acid sequence of the GPCR
- `Binding_Probability`: Predicted binding probability (0-1)

Results are sorted by binding probability (highest first).

**Example Output:**
```
SMILES,UniProt,Target_Sequence,Binding_Probability
CCOCc1sc...,P29274,MPIMGSSVYITVELAIAVLAILGNVLVCWAVWLNS...,0.8523
CCOCc1sc...,P30542,MPPSISAFQAAYIGIEVLIALVSVPGNVLVIWAVK...,0.7891
CCOCc1sc...,P28222,MEEPGAQCAPPPPAGSETWVPQANLSSAPSQNCSA...,0.7234
```

## 🔧 Configuration

Edit `config.yaml` to customize:
- Data paths and file names
- Model architecture parameters
- Training hyperparameters (learning rate, batch size, etc.)
- Device settings (CPU/GPU)
- Output directories

Example configuration for cross-attention models:
```yaml
finetune:
  protein_model: "Rostlab/prot_bert"
  molecule_model: "ibm/MolFormer-XL-both-10pct"
  d_model: 512
  n_heads: 4
  dropout: 0.1
  num_epochs: 10
  batch_size: 8
  learning_rate: 0.00001
  weight_decay: 0.00001
```

## 📚 Model Architectures

### 1. Cosine Similarity Model (CosSim)
Projects protein and ligand embeddings to a shared 1024-dimensional space using linear layers, then computes cosine similarity as the binding score.

**Key Features:**
- Simple and interpretable
- Low computational cost
- Good baseline performance
- Requires pre-computed embeddings

### 2. Transformer Encoder Model
Concatenates projected protein and ligand embeddings, processes through a multi-head transformer encoder, and classifies with an MLP.

**Key Features:**
- Self-attention mechanism
- Captures complex interactions
- Moderate computational cost
- Requires pre-computed embeddings

### 3. Cross-Attention Models (4 Variants)

Uses bidirectional cross-attention layers to allow proteins and ligands to attend to each other, capturing fine-grained interaction patterns.

**Architecture Flow:**
1. Encoders (ProtBert 1024-dim, MolFormer 768-dim)
2. Projectors to 512-dim shared space
3. Self-attention on both modalities
4. Bidirectional cross-attention (protein ↔ ligand)
5. Masked mean pooling
6. Concatenation (1024-dim)
7. MLP classifier (1024 → 512 → 256 → 1)

**Four Variants:**
- **CA-Base**: Both encoders frozen (★ **Best Performance**)
- **CA-Lig**: Fine-tune ligand encoder, freeze protein encoder
- **CA-Prot**: Fine-tune protein encoder, freeze ligand encoder
- **CA-Full**: Fine-tune both encoders

**Key Features:**
- Bidirectional attention mechanism
- Most expressive architecture
- Can compute embeddings on-the-fly
- Supports flexible encoder freezing strategies

### 4. XGBoost Baseline (XGB)
Projects protein and ligand embeddings to a shared latent space (Linear + ReLU, Xavier-initialised), concatenates them into a 2048-dim feature vector, and trains an XGBoost classifier on top.  The projection network is kept frozen — all learning happens inside XGBoost.

**Key Features:**
- Traditional ML baseline for comparison
- Random projection + gradient-boosted trees
- Same pre-computed embeddings as CosSim / Transformer
- Hyperparameters (`n_estimators`, `max_depth`, `learning_rate`) tunable in `config.yaml`

## Evaluation

Models are evaluated on three test scenarios:
1. **Random Split Validation**: Standard validation set from training distribution
2. **Unseen Proteins**: 9 GPCRs not present in training data
3. **Unseen Ligands**: Ligands not seen during training

**Metrics Computed:**
- Accuracy
- Precision
- Sensitivity (Recall)
- Specificity
- Matthews Correlation Coefficient (MCC)
- Area Under ROC Curve (AUROC)

## 🔬 Citation

If you use this code in your research, please cite:

```bibtex
@article{dey2024neurogpcrs,
  title={Evaluation of Deep Learning Architectures for Predicting GPCR-Mediated Neurotoxicity},
  author={Dey, Souvik and Lu, Pinyi and Wallqvist, Anders and AbdulHameed, Mohamed Diwan M.},
  journal={In preparation},
  year={2026},
}
```







