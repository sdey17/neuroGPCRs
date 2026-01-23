# Examples

This directory contains example scripts and notebooks for using the neuroGPCRs package.

## Quick Start

The `quick_start.py` script provides a minimal example of training a DTI prediction model:

```bash
python examples/quick_start.py
```

**Note:** You'll need to update the paths to your feature files in the script.

## Pre-computing Embeddings

Before training any models, you need to generate embeddings for your proteins and molecules using pre-trained language models.

### Protein Embeddings (ProtBert)

```python
from transformers import AutoTokenizer, AutoModel
import torch
import h5py
import pandas as pd

# Load ProtBert
tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
model = AutoModel.from_pretrained("Rostlab/prot_bert")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Load your data
df = pd.read_csv("data/training_set.csv")

# Create h5 file for storing embeddings
with h5py.File("ProtBert_features.h5", "w") as f:
    for sequence in df['Target Sequence'].unique():
        # Add spaces between amino acids
        spaced_seq = ' '.join(list(sequence))

        # Tokenize and encode
        inputs = tokenizer(spaced_seq, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # Use mean pooling of last hidden state
        embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

        # Store in h5 file
        f.create_dataset(sequence, data=embedding)

print("Protein embeddings saved to ProtBert_features.h5")
```

### Molecule Embeddings (MolFormer)

```python
from transformers import AutoTokenizer, AutoModel
import torch
import h5py
import pandas as pd

# Load MolFormer
tokenizer = AutoTokenizer.from_pretrained("ibm/MolFormer-XL-both-10pct", trust_remote_code=True)
model = AutoModel.from_pretrained("ibm/MolFormer-XL-both-10pct", trust_remote_code=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Load your data
df = pd.read_csv("data/training_set.csv")

# Create h5 file for storing embeddings
with h5py.File("MolFormer_features.h5", "w") as f:
    for smiles in df['SMILES'].unique():
        # Tokenize and encode
        inputs = tokenizer(smiles, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # Use mean pooling of last hidden state
        embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

        # Store in h5 file
        f.create_dataset(smiles, data=embedding)

print("Molecule embeddings saved to MolFormer_features.h5")
```

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
