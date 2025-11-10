"""
Attention Mechanisms for Multimodal Fusion

Implements:
1. CrossModalAttention: Attention between different modalities
2. TemporalAttention: Attention over time steps in sequences
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention: Modality A attends to Modality B.
    
    Example: Video features attend to IMU features to incorporate
    relevant motion information at each timestep.
    """
    
    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            query_dim: Dimension of query modality features
            key_dim: Dimension of key/value modality features  
            hidden_dim: Hidden dimension for attention computation
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        
        # Multi-head attention projections
        # Query from modality A, Key and Value from modality B
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(key_dim, hidden_dim)
        self.value_proj = nn.Linear(key_dim, hidden_dim)
        
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for cross-modal attention.
        
        Args:
            query: (batch_size, query_dim) - features from modality A
            key: (batch_size, key_dim) - features from modality B
            value: (batch_size, key_dim) - features from modality B
            mask: Optional (batch_size,) - binary mask for valid keys
            
        Returns:
            attended_features: (batch_size, hidden_dim) - query attended by key/value
            attention_weights: (batch_size, num_heads, 1, 1) - attention scores
        """
        batch_size = query.size(0)

        # Project to hidden_dim
        Q = self.query_proj(query)   # (B, hidden_dim)
        K = self.key_proj(key)       # (B, hidden_dim)
        V = self.value_proj(value)   # (B, hidden_dim)

        # Reshape to multi-head format
        # We treat this as a single "time step" per sample (seq_len = 1)
        # Shapes: (B, num_heads, 1, head_dim)
        Q = Q.view(batch_size, self.num_heads, 1, self.head_dim)
        K = K.view(batch_size, self.num_heads, 1, self.head_dim)
        V = V.view(batch_size, self.num_heads, 1, self.head_dim)

        # Dot-product attention scores (B, H, 1, 1)
        scores = (Q * K).sum(dim=-1, keepdim=True) / (self.head_dim ** 0.5)

        # Mask: if mask[b] == 0, we want attention weight ~ 0 and output ~ 0
        if mask is not None:
            # mask: (B,) -> (B, 1, 1, 1)
            m = mask.view(batch_size, 1, 1, 1).to(scores.dtype)
            # Instead of -inf + softmax (which can produce NaNs if all masked),
            # we just multiply scores by the mask.
            scores = scores * m

        # Use sigmoid for a stable scalar weight in [0,1]
        attn_weights = torch.sigmoid(scores)  # (B, H, 1, 1)

        # Apply attention to values
        attended_heads = attn_weights * V     # (B, H, 1, head_dim)
        # Remove the length dimension and merge heads
        attended = attended_heads.squeeze(2).reshape(batch_size, self.hidden_dim)  # (B, hidden_dim)

        # Output projection + dropout
        attended = self.out_proj(attended)
        attended = self.dropout(attended)

        # Ensure weights shape exactly matches spec: (B, H, 1, 1)
        return attended, attn_weights


class TemporalAttention(nn.Module):
    """
    Temporal attention: Attend over sequence of time steps.
    
    Useful for: Variable-length sequences, weighting important timesteps
    """
    
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            feature_dim: Dimension of input features at each timestep
            hidden_dim: Hidden dimension for attention
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"

        # Self-attention over temporal dimension
        self.q_proj = nn.Linear(feature_dim, hidden_dim)
        self.k_proj = nn.Linear(feature_dim, hidden_dim)
        self.v_proj = nn.Linear(feature_dim, hidden_dim)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        sequence: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for temporal attention.
        
        Args:
            sequence: (batch_size, seq_len, feature_dim) - temporal sequence
            mask: Optional (batch_size, seq_len) - binary mask for valid timesteps
            
        Returns:
            attended_sequence: (batch_size, seq_len, hidden_dim) - attended features
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        B, T, _ = sequence.shape

        # Project to Q,K,V
        Q = self.q_proj(sequence)  # (B, T, hidden_dim)
        K = self.k_proj(sequence)  # (B, T, hidden_dim)
        V = self.v_proj(sequence)  # (B, T, hidden_dim)

        # Reshape to multi-head: (B, num_heads, T, head_dim)
        def reshape_heads(x):
            return x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        Q = reshape_heads(Q)
        K = reshape_heads(K)
        V = reshape_heads(V)

        # Attention scores: (B, H, T, T)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Apply mask over keys (and effectively values)
        if mask is not None:
            # mask: (B, T) -> (B, 1, 1, T)
            key_mask = mask.view(B, 1, 1, T)
            scores = scores.masked_fill(key_mask == 0, -1e9)

        # Softmax over key positions
        attn_weights = torch.softmax(scores, dim=-1)  # (B, H, T, T)

        # If mask exists, zero out weights on invalid positions and renormalize
        if mask is not None:
            key_mask = mask.view(B, 1, 1, T).to(attn_weights.dtype)
            attn_weights = attn_weights * key_mask
            denom = attn_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            attn_weights = attn_weights / denom

        # Apply attention to values: (B, H, T, Dh)
        attended = torch.matmul(attn_weights, V)

        # Merge heads back: (B, T, hidden_dim)
        attended = attended.transpose(1, 2).contiguous().view(B, T, self.hidden_dim)
        attended = self.out_proj(attended)
        attended = self.dropout(attended)

        return attended, attn_weights
    
    def pool_sequence(
        self,
        sequence: torch.Tensor,
        attention_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Pool sequence to fixed-size representation using attention weights.
        
        Args:
            sequence: (batch_size, seq_len, hidden_dim)
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
            
        Returns:
            pooled: (batch_size, hidden_dim) - fixed-size representation
        """
        B, T, Hdim = sequence.shape

        # Compute overall importance of each timestep as:
        # mean over heads and queries of attention towards that timestep (as key)
        # attn: (B, H, Q, K)
        # importance over K dimension:
        importance = attention_weights.mean(dim=1).mean(dim=1)  # (B, T)

        # Normalize importance just in case
        importance = importance / importance.sum(dim=1, keepdim=True).clamp_min(1e-6)

        # Weighted sum of sequence: (B, Hdim)
        pooled = torch.bmm(importance.unsqueeze(1), sequence).squeeze(1)
        return pooled


class PairwiseModalityAttention(nn.Module):
    """
    Pairwise attention between all modality combinations.
    
    For M modalities, computes M*(M-1)/2 pairwise attention operations.
    Example: {video, audio, IMU} -> {video<->audio, video<->IMU, audio<->IMU}
    """
    
    def __init__(
        self,
        modality_dims: dict,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            modality_dims: Dict mapping modality names to feature dimensions
                          Example: {'video': 512, 'audio': 128, 'imu': 64}
            hidden_dim: Hidden dimension for attention
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.modality_names = list(modality_dims.keys())
        self.num_modalities = len(self.modality_names)
        self.hidden_dim = hidden_dim
        
        # Create CrossModalAttention for each modality pair (A->B and B->A)
        self.attn_modules = nn.ModuleDict()
        for i, mod_a in enumerate(self.modality_names):
            for j, mod_b in enumerate(self.modality_names):
                if i == j:
                    continue
                key = f"{mod_a}_to_{mod_b}"
                self.attn_modules[key] = CrossModalAttention(
                    query_dim=modality_dims[mod_a],
                    key_dim=modality_dims[mod_b],
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
    
    def forward(
        self,
        modality_features: dict,
        modality_mask: Optional[torch.Tensor] = None
    ) -> Tuple[dict, dict]:
        """
        Apply pairwise attention between all modalities.
        
        Args:
            modality_features: Dict of {modality_name: features}
                             Each tensor: (batch_size, feature_dim)
            modality_mask: (batch_size, num_modalities) - availability mask
            
        Returns:
            attended_features: Dict of {modality_name: attended_features}
            attention_maps: Dict of {f"{mod_a}_to_{mod_b}": attention_weights}
        """
        device = next(iter(modality_features.values())).device
        batch_size = next(iter(modality_features.values())).shape[0]

        if modality_mask is None:
            modality_mask = torch.ones(
                batch_size, self.num_modalities, device=device, dtype=torch.float32
            )

        idx = {m: i for i, m in enumerate(self.modality_names)}
        attended_features = {}
        attention_maps = {}

        for mod_a in self.modality_names:
            # Aggregate contributions from all other modalities
            agg = torch.zeros(batch_size, self.hidden_dim, device=device)
            count = torch.zeros(batch_size, 1, device=device)

            for mod_b in self.modality_names:
                if mod_a == mod_b:
                    continue

                key = f"{mod_a}_to_{mod_b}"
                attn_module = self.attn_modules[key]

                query = modality_features[mod_a]  # (B, D_a)
                key_feat = modality_features[mod_b]  # (B, D_b)
                val_feat = modality_features[mod_b]

                # per-sample availability of modality B
                b_mask = modality_mask[:, idx[mod_b]]

                attended, weights = attn_module(query, key_feat, val_feat, b_mask)

                # zero-out if modality B missing
                attended = attended * b_mask.view(batch_size, 1)

                agg = agg + attended
                count = count + b_mask.view(batch_size, 1)

                attention_maps[key] = weights

            # Avoid division by zero: if no other modality is available, just keep zeros
            count = count.clamp_min(1.0)
            attended_features[mod_a] = agg / count

        return attended_features, attention_maps


def visualize_attention(
    attention_weights: torch.Tensor,
    modality_names: list,
    save_path: str = None
) -> None:
    """
    Visualize attention weights between modalities.
    
    Args:
        attention_weights: (num_heads, num_queries, num_keys) or (batch, num_heads, num_queries, num_keys)
        modality_names: List of modality names for labeling
        save_path: Optional path to save visualization
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Move to CPU numpy
    with torch.no_grad():
        aw = attention_weights
        if aw.dim() == 4:
            # (B, H, Q, K) -> average over batch
            aw = aw.mean(dim=0)  # (H, Q, K)
        elif aw.dim() != 3:
            raise ValueError("Expected attention_weights with 3 or 4 dimensions")
        aw_np = aw.mean(axis=0).cpu().numpy()  # (Q, K)

    plt.figure(figsize=(6, 5))
    plt.imshow(aw_np, aspect="auto")
    plt.colorbar(label="Attention weight")

    num_queries, num_keys = aw_np.shape
    # Truncate / pad modality names to match dimensions if necessary
    x_labels = modality_names[:num_keys] + [""] * max(0, num_keys - len(modality_names))
    y_labels = modality_names[:num_queries] + [""] * max(0, num_queries - len(modality_names))

    plt.xticks(ticks=np.arange(num_keys), labels=x_labels, rotation=45, ha="right")
    plt.yticks(ticks=np.arange(num_queries), labels=y_labels)
    plt.xlabel("Keys")
    plt.ylabel("Queries")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


if __name__ == '__main__':
    # Simple test
    print("Testing attention mechanisms...")
    
    batch_size = 4
    query_dim = 512  # e.g., video features
    key_dim = 64     # e.g., IMU features
    hidden_dim = 256
    num_heads = 4
    
    # Test CrossModalAttention
    print("\nTesting CrossModalAttention...")
    try:
        attn = CrossModalAttention(query_dim, key_dim, hidden_dim, num_heads)
        
        query = torch.randn(batch_size, query_dim)
        key = torch.randn(batch_size, key_dim)
        value = torch.randn(batch_size, key_dim)
        
        attended, weights = attn(query, key, value)
        
        assert attended.shape == (batch_size, hidden_dim)
        print(f"✓ CrossModalAttention working! Output shape: {attended.shape}")
        
    except NotImplementedError:
        print("✗ CrossModalAttention not implemented yet")
    except Exception as e:
        print(f"✗ CrossModalAttention error: {e}")
    
    # Test TemporalAttention
    print("\nTesting TemporalAttention...")
    try:
        seq_len = 10
        feature_dim = 128
        
        temporal_attn = TemporalAttention(feature_dim, hidden_dim, num_heads)
        sequence = torch.randn(batch_size, seq_len, feature_dim)
        
        attended_seq, weights = temporal_attn(sequence)
        
        assert attended_seq.shape == (batch_size, seq_len, hidden_dim)
        print(f"✓ TemporalAttention working! Output shape: {attended_seq.shape}")
        
    except NotImplementedError:
        print("✗ TemporalAttention not implemented yet")
    except Exception as e:
        print(f"✗ TemporalAttention error: {e}")
