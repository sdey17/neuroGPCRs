# neuroGPCRs: Deep Learning for GPCR-Ligand Binding Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Abstract

G protein-coupled receptors (GPCRs) are therapeutic targets for over 30% of approved drugs, yet specific GPCR subtypes act as molecular initiating events in several neurotoxic adverse outcome pathways. This repository implements three deep learning architectures for predicting GPCR-ligand binding interactions:

1. **Dual-Projection Cosine Similarity Networks** - Projects protein and ligand embeddings to a shared space and computes cosine similarity
2. **Transformer Encoder Networks** - Uses self-attention mechanisms to process concatenated protein-ligand representations
3. **Cross-Attention Networks** - Employs bidirectional cross-attention between protein and ligand representations

All models use pre-trained language models (ProtBert for proteins, MolFormer for molecules) to generate initial embeddings, achieving strong performance on multiple evaluation scenarios including unseen proteins and ligands.

## 📊 Performance

| Model | Validation AUC | Unseen Protein AUC | Unseen Ligand AUC |
|-------|---------------|-------------------|-------------------|
| Cosine Similarity | 0.91 | 0.61 | 0.81 |
| Transformer | 0.91 | 0.61 | 0.81 |
| Cross-Attention | 0.91 | 0.61 | 0.81 |

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

For the Cosine Similarity and Transformer models, you'll need to pre-compute embeddings using ProtBert and MolFormer. See `examples/generate_embeddings.py` for details.

**Note:** The Cross-Attention model with fine-tuning (`train_cross_attention_finetune.py`) computes embeddings on-the-fly and doesn't require pre-computed features.

### Training Models

#### Cosine Similarity Model
```bash
python scripts/train_cosine.py \
    --config config.yaml \
    --protein_feat /path/to/ProtBert_features.h5 \
    --mol_feat /path/to/MolFormer_features.h5
```

#### Transformer Model
```bash
python scripts/train_transformer.py \
    --config config.yaml \
    --protein_feat /path/to/ProtBert_features.h5 \
    --mol_feat /path/to/MolFormer_features.h5
```

#### Cross-Attention Model (with pre-computed embeddings)
```bash
python scripts/train_cross_attention.py \
    --config config.yaml \
    --protein_feat /path/to/ProtBert_features.h5 \
    --mol_feat /path/to/MolFormer_features.h5
```

#### Cross-Attention Model (with end-to-end fine-tuning)
For the cross-attention model, you can also train end-to-end without pre-computing embeddings. This allows fine-tuning of the pre-trained encoders:

```bash
python scripts/train_cross_attention_finetune.py \
    --config config.yaml
```

**Note:** Fine-tuning requires more GPU memory and training time but may achieve better performance by adapting the encoders to your specific task.

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
│   │   └── cross_attention_model.py
│   └── utils/                  # Utility functions
│       ├── data_loader.py      # Data loading utilities
│       ├── metrics.py          # Evaluation metrics
│       └── training.py         # Training utilities
├── scripts/                    # Training scripts
│   ├── train_cosine.py
│   ├── train_transformer.py
│   └── train_cross_attention.py
├── examples/                   # Example notebooks and scripts
├── results/                    # Output directory for results
├── tests/                      # Unit tests
├── docs/                       # Documentation
├── config.yaml                 # Configuration file
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── LICENSE                     # License file
```

## 🔧 Configuration

Edit `config.yaml` to customize:
- Data paths and file names
- Model architecture parameters
- Training hyperparameters (learning rate, batch size, etc.)
- Device settings (CPU/GPU)
- Output directories

Example configuration:
```yaml
model:
  type: "transformer"
  protein_encoder: "ProtBert"
  molecule_encoder: "MolFormer"
  latent_dim: 1024
  n_heads: 4
  dropout: 0.1

training:
  num_epochs: 50
  batch_size: 32
  learning_rate: 0.0001
```

## 📚 Model Architectures

### 1. Cosine Similarity Model
Projects protein and ligand embeddings to a shared 1024-dimensional space using linear layers, then computes cosine similarity as the binding score.

**Key Features:**
- Simple and interpretable
- Low computational cost
- Good baseline performance

### 2. Transformer Encoder Model
Concatenates projected protein and ligand embeddings, processes through a multi-head transformer encoder, and classifies with an MLP.

**Key Features:**
- Self-attention mechanism
- Captures complex interactions
- Moderate computational cost

### 3. Cross-Attention Model
Uses bidirectional cross-attention layers to allow proteins and ligands to attend to each other, capturing fine-grained interaction patterns.

**Key Features:**
- Bidirectional attention
- Most expressive architecture
- Higher computational cost
- Best for complex binding patterns

**Two Modes:**
1. **Pre-computed embeddings** (`train_cross_attention.py`): Uses frozen embeddings from h5 files
2. **End-to-end fine-tuning** (`train_cross_attention_finetune.py`): Computes embeddings on-the-fly and fine-tunes the encoders
   - Supports freezing encoders or training them with lower learning rates
   - Requires more GPU memory but can achieve better performance
   - Recommended for tasks where you have sufficient training data

## 📊 Evaluation

Models are evaluated on three test scenarios:
1. **Random Split Validation**: Standard validation set from training distribution
2. **Unseen Proteins**: Test set containing proteins not seen during training
3. **Unseen Ligands**: Test set containing ligands not seen during training

Metrics computed:
- Accuracy
- Precision
- Sensitivity (Recall)
- Specificity
- Matthews Correlation Coefficient (MCC)
- Area Under ROC Curve (AUC)

## 🔬 Citation

If you use this code in your research, please cite:

```bibtex
@article{neurogpcrs2024,
  title={Deep Learning Models for GPCR-Ligand Binding Prediction},
  author={Your Name},
  journal={Journal Name},
  year={2024}
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

For questions or issues, please open an issue on GitHub or contact [your.email@example.com](mailto:your.email@example.com).

## 🙏 Acknowledgments

- ProtBert: Protein language model from Rostlab
- MolFormer: Molecular transformer from IBM Research
- PyTorch and Hugging Face Transformers for deep learning infrastructure

## 📝 TODO

- [ ] Add Jupyter notebook tutorials
- [ ] Implement ensemble methods
- [ ] Add attention visualization tools
- [ ] Support for additional protein encoders (ESM-2, etc.)
- [ ] Support for additional molecular encoders (ChemBERTa, etc.)
- [ ] Hyperparameter optimization examples
- [ ] Docker containerization
