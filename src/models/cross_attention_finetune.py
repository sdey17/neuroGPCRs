"""Cross-attention model with end-to-end fine-tuning support."""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict
from transformers import AutoModel


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


class DTIFineTuneCrossAttention(nn.Module):
    """
    Cross-attention model with end-to-end fine-tuning of encoders.

    This model loads pre-trained protein and molecule encoders and fine-tunes
    them along with the cross-attention layers for DTI prediction.

    Args:
        protein_model_name: HuggingFace model name for protein encoder
        molecule_model_name: HuggingFace model name for molecule encoder
        d_model: Model dimensionality for cross-attention
        n_heads: Number of attention heads
        n_layers: Number of cross-attention layers
        dropout: Dropout rate
        freeze_encoders: Whether to freeze encoder weights
    """

    def __init__(
        self,
        protein_model_name: str = "Rostlab/prot_bert",
        molecule_model_name: str = "ibm/MolFormer-XL-both-10pct",
        d_model: int = 512,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        freeze_encoders: bool = False
    ):
        super(DTIFineTuneCrossAttention, self).__init__()

        self.d_model = d_model
        self.freeze_encoders = freeze_encoders

        # Load pre-trained encoders
        print(f"Loading protein encoder: {protein_model_name}")
        self.protein_encoder = AutoModel.from_pretrained(protein_model_name)
        protein_hidden_size = self.protein_encoder.config.hidden_size

        print(f"Loading molecule encoder: {molecule_model_name}")
        self.molecule_encoder = AutoModel.from_pretrained(
            molecule_model_name,
            trust_remote_code=True
        )
        molecule_hidden_size = self.molecule_encoder.config.hidden_size

        # Freeze encoders if specified
        if freeze_encoders:
            print("Freezing encoder weights")
            for param in self.protein_encoder.parameters():
                param.requires_grad = False
            for param in self.molecule_encoder.parameters():
                param.requires_grad = False

        # Projection layers to common dimension
        self.protein_projector = nn.Linear(protein_hidden_size, d_model)
        self.molecule_projector = nn.Linear(molecule_hidden_size, d_model)

        # Cross-attention layers (stacked)
        self.cross_attention_layers = nn.ModuleList([
            nn.ModuleDict({
                'protein_to_mol': CrossAttentionLayer(d_model, n_heads, dropout),
                'mol_to_protein': CrossAttentionLayer(d_model, n_heads, dropout)
            })
            for _ in range(n_layers)
        ])

        # Feed-forward layers after cross-attention
        self.protein_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )

        self.molecule_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )

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
        protein_input_ids: torch.Tensor,
        protein_attention_mask: torch.Tensor,
        molecule_input_ids: torch.Tensor,
        molecule_attention_mask: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass with on-the-fly encoding.

        Args:
            protein_input_ids: Tokenized protein sequences (batch_size, seq_len)
            protein_attention_mask: Attention mask for proteins
            molecule_input_ids: Tokenized molecule SMILES (batch_size, seq_len)
            molecule_attention_mask: Attention mask for molecules
            return_attention: Whether to return attention weights

        Returns:
            Binding predictions (batch_size, 1) or tuple with attention if requested
        """
        # Encode protein and molecule
        protein_outputs = self.protein_encoder(
            input_ids=protein_input_ids,
            attention_mask=protein_attention_mask
        )
        protein_embeddings = protein_outputs.last_hidden_state  # (batch, seq_len, hidden_size)

        molecule_outputs = self.molecule_encoder(
            input_ids=molecule_input_ids,
            attention_mask=molecule_attention_mask
        )
        molecule_embeddings = molecule_outputs.last_hidden_state  # (batch, seq_len, hidden_size)

        # Project to common dimension
        protein_proj = self.protein_projector(protein_embeddings)
        molecule_proj = self.molecule_projector(molecule_embeddings)

        # Apply stacked cross-attention layers
        attention_weights_list = []
        protein_cross = protein_proj
        molecule_cross = molecule_proj

        for layer in self.cross_attention_layers:
            # Protein attends to molecule
            protein_cross, p2m_att = layer['protein_to_mol'](
                protein_cross, molecule_cross, molecule_cross,
                key_mask=molecule_attention_mask,
                return_attention=return_attention
            )

            # Molecule attends to protein
            molecule_cross, m2p_att = layer['mol_to_protein'](
                molecule_cross, protein_cross, protein_cross,
                key_mask=protein_attention_mask,
                return_attention=return_attention
            )

            if return_attention:
                attention_weights_list.append({
                    'protein_to_mol': p2m_att,
                    'mol_to_protein': m2p_att
                })

        # Apply feed-forward layers
        protein_cross = self.protein_ffn(protein_cross)
        molecule_cross = self.molecule_ffn(molecule_cross)

        # Pool over sequence dimension (mean pooling with attention mask)
        protein_mask_expanded = protein_attention_mask.unsqueeze(-1).float()
        protein_pooled = (protein_cross * protein_mask_expanded).sum(1) / protein_mask_expanded.sum(1)

        molecule_mask_expanded = molecule_attention_mask.unsqueeze(-1).float()
        molecule_pooled = (molecule_cross * molecule_mask_expanded).sum(1) / molecule_mask_expanded.sum(1)

        # Combine and classify
        combined_features = torch.cat([protein_pooled, molecule_pooled], dim=1)
        logits = self.classifier(combined_features)

        if return_attention:
            return logits, attention_weights_list

        return logits

    def get_num_params(self) -> Dict[str, int]:
        """Get number of parameters in each component."""
        protein_params = sum(p.numel() for p in self.protein_encoder.parameters())
        molecule_params = sum(p.numel() for p in self.molecule_encoder.parameters())
        cross_attention_params = sum(p.numel() for layer in self.cross_attention_layers for p in layer.parameters())
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        total_params = sum(p.numel() for p in self.parameters())

        return {
            'protein_encoder': protein_params,
            'molecule_encoder': molecule_params,
            'cross_attention': cross_attention_params,
            'classifier': classifier_params,
            'total': total_params
        }
