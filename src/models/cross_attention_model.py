"""Cross-attention model for DTI prediction with end-to-end fine-tuning support."""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
from transformers import AutoTokenizer, AutoModel


class CrossAttentionLayer(nn.Module):
    """
    Cross-attention layer for protein-ligand interaction.

    Args:
        d_model: Model dimensionality
        n_heads: Number of attention heads
        dropout: Dropout rate
    """

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

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with optional attention weights return.

        Args:
            query: Query tensor (batch_size, seq_len_q, d_model)
            key: Key tensor (batch_size, seq_len_k, d_model)
            value: Value tensor (batch_size, seq_len_k, d_model)
            key_mask: Mask for key padding (batch_size, seq_len_k)
            return_attention: Whether to return attention weights

        Returns:
            Tuple of (output, attention_weights)
        """
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


class DTIProfilerCrossAttention(nn.Module):
    """
    Cross-attention network for DTI prediction using pre-computed embeddings.

    This is a simplified version that works with pre-computed embeddings from
    ProtBert and MolFormer, rather than doing end-to-end fine-tuning.

    Args:
        drug_dim: Dimensionality of drug embeddings
        target_dim: Dimensionality of protein embeddings
        d_model: Model dimensionality for cross-attention
        n_heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(
        self,
        drug_dim: int = 768,
        target_dim: int = 1024,
        d_model: int = 512,
        n_heads: int = 4,
        dropout: float = 0.1
    ):
        super(DTIProfilerCrossAttention, self).__init__()

        self.d_model = d_model

        # Input projections to common dimension
        self.protein_projector = nn.Linear(target_dim, d_model)
        self.mol_projector = nn.Linear(drug_dim, d_model)

        # Self-attention layers
        self.protein_self_attention = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True
        )
        self.mol_self_attention = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True
        )

        # Cross-attention layers
        self.protein_to_mol_attention = CrossAttentionLayer(d_model, n_heads, dropout)
        self.mol_to_protein_attention = CrossAttentionLayer(d_model, n_heads, dropout)

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        drug: torch.Tensor,
        target: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            drug: Drug embeddings (batch_size, drug_dim)
            target: Protein embeddings (batch_size, target_dim)
            return_attention: Whether to return attention weights

        Returns:
            Binding predictions (batch_size, 1)
        """
        # Add sequence dimension and project
        protein_proj = self.protein_projector(target.unsqueeze(1))
        mol_proj = self.mol_projector(drug.unsqueeze(1))

        # Self-attention
        protein_self_att = self.protein_self_attention(protein_proj)
        mol_self_att = self.mol_self_attention(mol_proj)

        # Cross-attention
        protein_cross_att, prot_to_mol_att = self.protein_to_mol_attention(
            protein_self_att, mol_self_att, mol_self_att, return_attention=return_attention
        )
        mol_cross_att, mol_to_prot_att = self.mol_to_protein_attention(
            mol_self_att, protein_self_att, protein_self_att, return_attention=return_attention
        )

        # Pool and combine
        protein_pooled = protein_cross_att.squeeze(1)
        mol_pooled = mol_cross_att.squeeze(1)
        combined_features = torch.cat([protein_pooled, mol_pooled], dim=1)

        # Classify
        logits = self.classifier(combined_features)

        if return_attention:
            return logits, {'protein_to_mol': prot_to_mol_att, 'mol_to_protein': mol_to_prot_att}

        return logits
