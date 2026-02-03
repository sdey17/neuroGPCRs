"""Data loading utilities for fine-tuning with raw sequences and SMILES."""

import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, Callable


class DTIFineTuneDataset(Dataset):
    """Dataset class for DTI with raw sequences and SMILES (for fine-tuning)."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        protein_tokenizer: Callable,
        molecule_tokenizer: Callable,
        max_protein_len: int = 1024,
        max_molecule_len: int = 512
    ):
        """
        Initialize fine-tuning dataset.

        Args:
            dataframe: DataFrame containing protein sequences, SMILES, and labels
            protein_tokenizer: Tokenizer for protein sequences
            molecule_tokenizer: Tokenizer for molecules
            max_protein_len: Maximum protein sequence length
            max_molecule_len: Maximum molecule sequence length
        """
        self.proteins = dataframe['Target Sequence'].tolist()
        self.smiles = dataframe['SMILES'].tolist()
        self.labels = dataframe['Label'].tolist()
        self.protein_tokenizer = protein_tokenizer
        self.molecule_tokenizer = molecule_tokenizer
        self.max_protein_len = max_protein_len
        self.max_molecule_len = max_molecule_len

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        protein_seq = self.proteins[idx]
        smiles_str = self.smiles[idx]
        label = self.labels[idx]

        # Tokenize protein (add spaces between amino acids for ProtBert)
        spaced_protein = ' '.join(list(protein_seq))
        protein_tokens = self.protein_tokenizer(
            spaced_protein,
            max_length=self.max_protein_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Tokenize molecule
        molecule_tokens = self.molecule_tokenizer(
            smiles_str,
            max_length=self.max_molecule_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'protein_input_ids': protein_tokens['input_ids'].squeeze(0),
            'protein_attention_mask': protein_tokens['attention_mask'].squeeze(0),
            'molecule_input_ids': molecule_tokens['input_ids'].squeeze(0),
            'molecule_attention_mask': molecule_tokens['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float32)
        }


