"""
Multimodal Fusion Architectures for Sensor Integration

This module implements three fusion strategies:
1. Early Fusion: Concatenate features before processing
2. Late Fusion: Independent processing, combine predictions
3. Hybrid Fusion: Cross-modal attention + learned weighting
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from attention import CrossModalAttention


class EarlyFusion(nn.Module):
    """
    Early fusion: Concatenate encoder outputs and process jointly.
    
    Pros: Joint representation learning across modalities
    Cons: Requires temporal alignment, sensitive to missing modalities
    """
    
    def __init__(
        self,
        modality_dims: Dict[str, int],
        hidden_dim: int = 256,
        num_classes: int = 11,
        dropout: float = 0.1,
        **kwargs,  # absorbs extra args like num_heads from config
    ):
        """
        Args:
            modality_dims: Dictionary mapping modality name to feature dimension
                          Example: {'video': 512, 'imu': 64}
            hidden_dim: Hidden dimension for fusion network
            num_classes: Number of output classes
            dropout: Dropout probability
        """
        super().__init__()
        self.modality_names = list(modality_dims.keys())
        
        # Concatenate all modality features, pass through MLP
        concat_dim = sum(modality_dims[m] for m in self.modality_names)
        self.fusion = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        modality_features: Dict[str, torch.Tensor],
        modality_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with early fusion.
        
        Args:
            modality_features: Dict of {modality_name: features}
                             Each tensor shape: (batch_size, feature_dim)
                             or (batch_size, T, feature_dim) – we will mean-pool over T.
            modality_mask: Binary mask (batch_size, num_modalities)
                          1 = available, 0 = missing
                          
        Returns:
            logits: (batch_size, num_classes)
        """
        first_feat = next(iter(modality_features.values()))
        device = first_feat.device
        batch_size = first_feat.shape[0]

        if modality_mask is None:
            modality_mask = torch.ones(
                batch_size, len(self.modality_names),
                device=device, dtype=torch.float32
            )

        feats = []
        for i, name in enumerate(self.modality_names):
            x = modality_features[name]  # (B, D) or (B, T, D)
            # If encoded features are sequences (B, T, D), pool over time
            if x.dim() == 3:
                x = x.mean(dim=1)  # (B, D)
            m = modality_mask[:, i].view(batch_size, 1)  # (B, 1)
            x = x * m  # zero-out missing modality
            feats.append(x)

        fused_input = torch.cat(feats, dim=-1)  # (B, sum D_i)
        logits = self.fusion(fused_input)       # (B, num_classes)
        return logits


class LateFusion(nn.Module):
    """
    Late fusion: Independent classifiers per modality, combine predictions.
    
    Pros: Handles asynchronous sensors, modular per-modality training
    Cons: Limited cross-modal interaction, fusion only at decision level
    """
    
    def __init__(
        self,
        modality_dims: Dict[str, int],
        hidden_dim: int = 256,
        num_classes: int = 11,
        dropout: float = 0.1,
        **kwargs,
    ):
        """
        Args:
            modality_dims: Dictionary mapping modality name to feature dimension
            hidden_dim: Hidden dimension for per-modality classifiers
            num_classes: Number of output classes
            dropout: Dropout probability
        """
        super().__init__()
        self.modality_names = list(modality_dims.keys())
        self.num_modalities = len(self.modality_names)

        # Per-modality classifiers
        # Each: Linear(D_m, hidden_dim) -> ReLU -> Dropout -> Linear(hidden_dim, num_classes)
        classifiers = {}
        for name, dim in modality_dims.items():
            classifiers[name] = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        self.classifiers = nn.ModuleDict(classifiers)

        # Learnable fusion weights over modalities (global, then masked per batch)
        # Shape: (num_modalities,)
        self.fusion_logits = nn.Parameter(
            torch.ones(self.num_modalities, dtype=torch.float32)
        )
    
    def forward(
        self,
        modality_features: Dict[str, torch.Tensor],
        modality_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with late fusion.
        
        Args:
            modality_features: Dict of {modality_name: features}
                               Each tensor: (batch_size, D_m)
            modality_mask: Binary mask for available modalities
                           Shape: (batch_size, num_modalities)
            
        Returns:
            logits: (batch_size, num_classes) - fused predictions
            per_modality_logits: Dict of individual modality predictions
                                 {modality_name: (batch_size, num_classes)}
        """
        # Get batch size / device from any modality
        first_feat = next(iter(modality_features.values()))
        device = first_feat.device
        batch_size = first_feat.shape[0]

        if modality_mask is None:
            modality_mask = torch.ones(
                batch_size, self.num_modalities,
                device=device, dtype=torch.float32
            )

        # 1. Per-modality predictions
        per_modality_logits: Dict[str, torch.Tensor] = {}
        for i, name in enumerate(self.modality_names):
            x = modality_features[name]  # (B, D_m) or (B, T, D_m)
            # If somehow 3D slipped in, mean-pool over time
            if x.dim() == 3:
                x = x.mean(dim=1)  # (B, D_m)

            logits_m = self.classifiers[name](x)  # (B, C)
            per_modality_logits[name] = logits_m

        # 2. Compute fusion weights per batch
        # base logits: (M,) -> (B, M)
        fusion_logits = self.fusion_logits.unsqueeze(0).expand(batch_size, -1)  # (B, M)

        # mask out missing modalities before softmax
        # where mask == 0 → set to -inf so weight ~ 0
        masked_logits = fusion_logits.masked_fill(
            modality_mask == 0, float('-inf')
        )

        fusion_weights = torch.softmax(masked_logits, dim=-1)  # (B, M)

        # 3. Weighted sum of per-modality logits
        fused_logits = torch.zeros(
            batch_size,
            next(iter(per_modality_logits.values())).shape[1],
            device=device,
            dtype=next(iter(per_modality_logits.values())).dtype,
        )

        for i, name in enumerate(self.modality_names):
            w_i = fusion_weights[:, i].view(batch_size, 1)  # (B, 1)
            fused_logits = fused_logits + w_i * per_modality_logits[name]

        return fused_logits, per_modality_logits

class HybridFusion(nn.Module):
    """
    Hybrid fusion: shared latent space + learned adaptive fusion weights.
    
    Pros: Richer than pure early/late fusion, robust to missing modalities.
    """

    def __init__(
        self,
        modality_dims: Dict[str, int],
        hidden_dim: int = 256,
        num_classes: int = 11,
        num_heads: int = 4,   # kept in signature to match config, not used here
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.modality_names = list(modality_dims.keys())
        self.num_modalities = len(self.modality_names)
        self.hidden_dim = hidden_dim

        # 1) Project each modality to a common hidden dimension
        proj_layers = {}
        for name, dim in modality_dims.items():
            proj_layers[name] = nn.Linear(dim, hidden_dim)
        self.projections = nn.ModuleDict(proj_layers)

        # 2) Adaptive fusion weight predictor
        #    For each modality: take its hidden feature + mask bit -> scalar logit
        self.weight_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # 3) Final classifier on fused representation
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        modality_features: Dict[str, torch.Tensor],
        modality_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass with hybrid fusion.

        Args:
            modality_features: Dict of {modality_name: features}
                               Each tensor: (B, D_m) or (B, T, D_m)
            modality_mask: Binary mask for available modalities (B, M)
            return_attention: If True, returns fusion weights in the info dict

        Returns:
            logits: (B, num_classes)
            attention_info: dict with 'fusion_weights', 'projected_feats'
        """
        # Get batch/device
        first_feat = next(iter(modality_features.values()))
        device = first_feat.device
        batch_size = first_feat.shape[0]

        if modality_mask is None:
            modality_mask = torch.ones(
                batch_size, self.num_modalities,
                device=device, dtype=torch.float32
            )

        # 1) Project all modalities to common hidden dimension
        proj_feats_dict: Dict[str, torch.Tensor] = {}
        proj_list = []
        for i, name in enumerate(self.modality_names):
            x = modality_features[name]  # (B, D_m) or (B, T, D_m)
            # If features are sequences, mean-pool over time
            if x.dim() == 3:
                x = x.mean(dim=1)  # (B, D_m)
            z = self.projections[name](x)  # (B, H)
            proj_feats_dict[name] = z
            proj_list.append(z)

        # Stack to (B, M, H)
        proj_feats = torch.stack(proj_list, dim=1)  # (B, M, H)

        # 2) Compute adaptive fusion weights based on projected features + mask
        fusion_weights = self.compute_adaptive_weights(
            proj_feats_dict,  # dict of {name: (B,H)}
            modality_mask,    # (B, M)
        )  # (B, M)

        # 3) Fuse modality representations with learned weights
        fused = torch.zeros(
            batch_size,
            self.hidden_dim,
            device=device,
            dtype=proj_feats.dtype,
        )
        for i, name in enumerate(self.modality_names):
            w_i = fusion_weights[:, i].view(batch_size, 1)      # (B, 1)
            fused = fused + w_i * proj_feats_dict[name]         # (B, H)

        # 4) Final classifier
        logits = self.classifier(fused)  # (B, num_classes)

        attention_info = None
        if return_attention:
            attention_info = {
                "fusion_weights": fusion_weights,  # (B, M)
                "projected_feats": proj_feats,     # (B, M, H)
            }

        return logits, attention_info

    def compute_adaptive_weights(
        self,
        modality_features: Dict[str, torch.Tensor],
        modality_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute adaptive fusion weights based on modality features + availability.

        Args:
            modality_features: Dict {mod_name: (B, H)} AFTER projection
            modality_mask: (batch_size, num_modalities) binary mask

        Returns:
            weights: (batch_size, num_modalities) normalized fusion weights
        """
        # Stack features in modality order → (B, M, H)
        names = self.modality_names
        feat_list = [modality_features[name] for name in names]  # each (B,H)
        feats = torch.stack(feat_list, dim=1)  # (B, M, H)

        batch_size, num_modalities, H = feats.shape
        device = feats.device

        modality_mask = modality_mask.to(device=device, dtype=feats.dtype)  # (B, M)

        # Concatenate features with mask bit per modality: (B,M,H+1)
        mask_bit = modality_mask.unsqueeze(-1)          # (B,M,1)
        weight_input = torch.cat([feats, mask_bit], dim=-1)  # (B,M,H+1)

        # Predict logits per modality: (B,M,1) -> (B,M)
        logits = self.weight_mlp(weight_input).squeeze(-1)  # (B,M)

        # Mask out unavailable modalities before softmax
        masked_logits = logits.masked_fill(
            modality_mask == 0, float("-inf")
        )

        weights = torch.softmax(masked_logits, dim=-1)  # (B,M)
        return weights
# Helper functions

def build_fusion_model(
    fusion_type: str,
    modality_dims: Dict[str, int],
    num_classes: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to build fusion models.
    
    Args:
        fusion_type: One of ['early', 'late', 'hybrid']
        modality_dims: Dictionary mapping modality names to dimensions
        num_classes: Number of output classes
        **kwargs: Additional arguments for fusion model
        
    Returns:
        Fusion model instance
    """
    fusion_classes = {
        'early': EarlyFusion,
        'late': LateFusion,
        'hybrid': HybridFusion,
    }
    
    if fusion_type not in fusion_classes:
        raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    return fusion_classes[fusion_type](
        modality_dims=modality_dims,
        num_classes=num_classes,
        **kwargs
    )


if __name__ == '__main__':
    # Simple test to verify implementation
    print("Testing fusion architectures...")
    
    # Test configuration
    modality_dims = {'video': 512, 'imu': 64}
    num_classes = 11
    batch_size = 4
    
    # Create dummy features
    features = {
        'video': torch.randn(batch_size, 512),
        'imu': torch.randn(batch_size, 64)
    }
    mask = torch.tensor([[1, 1], [1, 0], [0, 1], [1, 1]])  # Different availability patterns
    
    # Test each fusion type
    for fusion_type in ['early', 'late', 'hybrid']:
        print(f"\nTesting {fusion_type} fusion...")
        try:
            model = build_fusion_model(fusion_type, modality_dims, num_classes)
            
            if fusion_type == 'late':
                logits, per_mod_logits = model(features, mask)
            elif fusion_type == 'hybrid':
                logits, attn_info = model(features, mask, return_attention=True)
            else:
                logits = model(features, mask)
            
            assert logits.shape == (batch_size, num_classes), \
                f"Expected shape ({batch_size}, {num_classes}), got {logits.shape}"
            print(f"✓ {fusion_type} fusion working! Output shape: {logits.shape}")
            
        except NotImplementedError:
            print(f"✗ {fusion_type} fusion not implemented yet")
        except Exception as e:
            print(f"✗ {fusion_type} fusion error: {e}")
