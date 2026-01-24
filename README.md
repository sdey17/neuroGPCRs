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

## 📊 Performance (Off-Target Profiling Task)

| Model | Accuracy | Sensitivity | Specificity | MCC | AUROC |
|-------|----------|-------------|-------------|-----|-------|
| **CosSim** | 0.785 ± 0.018 | 0.374 ± 0.025 | 0.912 ± 0.025 | 0.338 ± 0.041 | 0.640 ± 0.012 |
| **Transformers** | 0.744 ± 0.017 | 0.472 ± 0.053 | 0.827 ± 0.021 | 0.296 ± 0.048 | 0.632 ± 0.013 |
| **CA-Base** | **0.795 ± 0.010** | 0.494 ± 0.041 | **0.888 ± 0.018** | 0.403 ± 0.029 | **0.714 ± 0.017** |
| **CA-Lig** | 0.794 ± 0.021 | **0.506 ± 0.037** | 0.882 ± 0.024 | **0.407 ± 0.051** | 0.698 ± 0.014 |
| **CA-Prot** | 0.670 ± 0.028 | 0.331 ± 0.102 | 0.774 ± 0.064 | 0.103 ± 0.040 | 0.553 ± 0.025 |
| **CA-Full** | 0.709 ± 0.009 | 0.410 ± 0.044 | 0.802 ± 0.025 | 0.208 ± 0.014 | 0.606 ± 0.010 |
| **XGB** | 0.754 ± 0.014 | 0.353 ± 0.031 | 0.878 ± 0.013 | 0.257 ± 0.042 | 0.623 ± 0.011 |

**Key Finding**: CA-Base (frozen encoders) achieves the best performance, suggesting that task-specific training of pre-trained encoders may lead to overfitting for this dataset.

## 🔬 Dataset

- **Training Set**: 119 GPCRs with binding data
- **Validation Set**: Random split from training distribution
- **Test Set (Unseen Proteins)**: 9 GPCRs not seen during training
- **Test Set (Unseen Ligands)**: Ligands not seen during training

## 🚀 Quick Start

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

### Pre-computing Embeddings (Optional)

For CosSim, Transformers, and XGBoost models, pre-compute embeddings using ProtBert and MolFormer. See `examples/generate_embeddings.py` for details.

**Note:** Cross-attention models can compute embeddings on-the-fly, eliminating the need for pre-computed features.

### Training Models

#### 1. Cosine Similarity Model
```bash
python scripts/train_cosine.py \
    --config config.yaml \
    --protein_feat /path/to/ProtBert_features.h5 \
    --mol_feat /path/to/MolFormer_features.h5
```

#### 2. Transformer Encoder Model
```bash
python scripts/train_transformer.py \
    --config config.yaml \
    --protein_feat /path/to/ProtBert_features.h5 \
    --mol_feat /path/to/MolFormer_features.h5
```

#### 3. Cross-Attention Models (Unified Script)

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

## 📁 Repository Structure

```
neuroGPCRs/
├── data/                       # Data directory
│   ├── training_set.csv
│   ├── validation_set.csv
│   ├── test_set_unseen_protein.csv
│   └── test_set_unseen_ligands.csv
├── src/                        # Source code
│   ├── models/                 # Model architectures
│   │   ├── cosine_model.py
│   │   ├── transformer_model.py
│   │   ├── cross_attention_model.py  # For pre-computed embeddings
│   │   └── cross_attention_finetune.py  # For on-the-fly encoding
│   └── utils/                  # Utility functions
│       ├── data_loader.py      # Data loading (pre-computed embeddings)
│       ├── finetune_data_loader.py  # Data loading (raw sequences)
│       ├── metrics.py          # Evaluation metrics
│       ├── training.py         # Training utilities
│       └── finetune_training.py  # Fine-tuning utilities
├── scripts/                    # Training scripts
│   ├── train_cosine.py
│   ├── train_transformer.py
│   ├── train_cross_attention.py
│   └── train_cross_attention_unified.py  # All 4 CA variants
├── examples/                   # Example notebooks and scripts
├── manuscript/                 # Research manuscript
│   ├── Main_text.docx
│   └── Supplementary_info.docx
├── old_scripts/               # Original implementation (reference)
├── results/                   # Output directory for results
├── tests/                     # Unit tests
├── docs/                      # Documentation
├── config.yaml                # Configuration file
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── LICENSE                    # MIT License
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
  learning_rate: 0.00005  # Task-specific layers
  encoder_lr: 0.00001     # Pre-trained encoders (if trainable)
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
Traditional machine learning baseline using XGBoost on projected embeddings.

## 📊 Evaluation

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
  year={2024},
  institution={DoD Biotechnology HPC Software Applications Institute}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

For questions or issues, please open an issue on GitHub.

**Authors:**
- Souvik Dey
- Pinyi Lu
- Anders Wallqvist (sven.a.wallqvist.civ@health.mil)
- Mohamed Diwan M. AbdulHameed (mabdulhameed@bhsai.org)

**Affiliation:**
DoD Biotechnology High Performance Computing Software Applications Institute
Defense Health Agency Research & Development
Fort Detrick, MD 21702-5012

## 🙏 Acknowledgments

- ProtBert: Protein language model from Rostlab
- MolFormer: Molecular transformer from IBM Research
- PyTorch and Hugging Face Transformers for deep learning infrastructure
- Department of Defense Biotechnology HPC Software Applications Institute

## 📝 Key Findings

1. **CA-Base outperforms fine-tuning approaches**: Freezing both encoders achieves the best AUROC (0.714), suggesting pre-trained representations are already optimal for this task.

2. **Ligand encoder fine-tuning helps**: CA-Lig slightly outperforms CA-Base in sensitivity and MCC.

3. **Full fine-tuning can hurt performance**: CA-Full shows degraded performance compared to CA-Base, indicating potential overfitting when fine-tuning both encoders.

4. **Cross-attention superior to simpler architectures**: CA models outperform CosSim and Transformer baselines.

5. **Dataset characteristics matter**: The strong performance of frozen encoders may be specific to this dataset size and diversity.
