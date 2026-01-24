# Examples

This directory contains example scripts and notebooks for using the neuroGPCRs package.

## Quick Start

The `quick_start.py` script provides a minimal example of training a DTI prediction model:

```bash
python examples/quick_start.py
```

**Note:** You'll need to update the paths to your feature files in the script.

## Pre-computing Embeddings

Before training CosSim, Transformer, or XGBoost models, you need to generate embeddings using pre-trained language models (ProtBert for proteins, MolFormer for molecules).

**Note:** Cross-attention models can encode sequences on-the-fly and don't require pre-computed embeddings.

### Using the Embedding Generation Script

The repository provides `scripts/generate_embeddings.py` for automated embedding generation with proper attention-masked mean pooling:

```bash
# Generate both protein and molecule embeddings from all datasets
python scripts/generate_embeddings.py \
    --data_files data/training_set.csv data/validation_set.csv \
                data/test_set_unseen_protein.csv data/test_set_unseen_ligands.csv \
    --output_dir . \
    --protein_batch_size 8 \
    --molecule_batch_size 32
```

This will create:
- `prot_bert_features.h5` - Protein embeddings (1024-dim)
- `MolFormer-XL-both-10pct_features.h5` - Molecule embeddings (768-dim)

### Advanced Options

**Skip protein or molecule generation:**
```bash
# Only generate protein embeddings
python scripts/generate_embeddings.py --skip_molecules

# Only generate molecule embeddings
python scripts/generate_embeddings.py --skip_proteins
```

**Use custom models:**
```bash
python scripts/generate_embeddings.py \
    --protein_model "Rostlab/prot_bert_bfd" \
    --molecule_model "seyonec/ChemBERTa-zinc-base-v1"
```

**Adjust batch sizes for your GPU:**
```bash
# Smaller batches for limited GPU memory
python scripts/generate_embeddings.py \
    --protein_batch_size 4 \
    --molecule_batch_size 16
```

### Output Format

Embeddings are saved as HDF5 files with sequences/SMILES as keys:
```python
import h5py

# Load protein embeddings
with h5py.File("prot_bert_features.h5", "r") as f:
    sequence = "MPIMGSSVYITVELAIAVLAILGNVLVCWAVWLNS..."
    embedding = f[sequence][:]  # Shape: (1024,)

# Load molecule embeddings
with h5py.File("MolFormer-XL-both-10pct_features.h5", "r") as f:
    smiles = "CCOCc1sc(NC(=O)c2ccco2)nc1-c1ccccc1"
    embedding = f[smiles][:]  # Shape: (768,)
```

### Technical Details

The script uses **attention-masked mean pooling** from the last hidden state:
```python
attention_mask = inputs['attention_mask'].unsqueeze(-1)
embeddings = (outputs.last_hidden_state * attention_mask).sum(1) / attention_mask.sum(1)
```

This properly handles padding tokens, unlike simple `.mean(dim=1)` which includes padding in the average.

## Advanced Examples

### Custom Training Loop

For more control over the training process, you can implement a custom training loop:

```python
from src.models import DTIProfilerCosine
from src.utils.data_loader import load_datasets, extract_features, DTIDataset
from torch.utils.data import DataLoader

# Load data and features
train_df, val_df, _, _ = load_datasets()
prot_feat, mol_feat, target_dim, drug_dim = extract_features(
    "ProtBert_features.h5", "MolFormer_features.h5"
)

# Create datasets
train_dataset = DTIDataset(train_df, prot_feat, mol_feat)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize model
model = DTIProfilerCosine(drug_dim, target_dim)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.BCELoss()

# Custom training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        # Your custom training logic here
        pass
```

### Model Evaluation

```python
from src.utils.metrics import evaluate_predictions, print_metrics
import pandas as pd

# Load predictions
df = pd.read_csv("results/val_predictions_transformer.csv")

# Calculate metrics
metrics = evaluate_predictions(df)
print_metrics(metrics, "Validation Set")
```

### Comparing Models

```python
from src.utils.metrics import compare_models
import pandas as pd

results = {
    'Cosine': evaluate_predictions(pd.read_csv("results/val_predictions_cosine.csv")),
    'Transformer': evaluate_predictions(pd.read_csv("results/val_predictions_transformer.csv")),
    'Cross-Attention': evaluate_predictions(pd.read_csv("results/val_predictions_cross_attention.csv"))
}

comparison_df = compare_models(results)
print(comparison_df)
```

## Jupyter Notebooks

(Coming soon)
- `01_data_exploration.ipynb` - Explore the GPCR-ligand dataset
- `02_model_training.ipynb` - Interactive model training
- `03_model_evaluation.ipynb` - Detailed model evaluation and visualization
- `04_attention_visualization.ipynb` - Visualize attention weights

## Tips

1. **Start small**: Use a subset of your data for initial testing
2. **Monitor GPU memory**: Adjust batch size if you run out of memory
3. **Save checkpoints**: Models are automatically saved during training
4. **Use validation set**: Always evaluate on unseen data
5. **Experiment**: Try different architectures and hyperparameters
