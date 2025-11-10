"""
Modality-Specific Encoders for Sensor Fusion

Implements lightweight encoders suitable for CPU training:
1. SequenceEncoder: For time-series data (IMU, audio, motion capture)
2. FrameEncoder: For frame-based data (video features)
3. SimpleMLPEncoder: For pre-extracted features
"""

import torch
import torch.nn as nn
from typing import Optional


class SequenceEncoder(nn.Module):
    """
    Encoder for sequential/time-series sensor data.
    
    Options: 1D CNN, LSTM, GRU, or Transformer
    Output: Fixed-size embedding per sequence
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 64,
        num_layers: int = 2,
        encoder_type: str = 'lstm',
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim: Dimension of input features at each timestep
            hidden_dim: Hidden dimension for RNN/Transformer
            output_dim: Output embedding dimension
            num_layers: Number of encoder layers
            encoder_type: One of ['lstm', 'gru', 'cnn', 'transformer']
            dropout: Dropout probability
        """
        super().__init__()
        self.encoder_type = encoder_type
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # TODO: Implement sequence encoder
        # Choose ONE of the following architectures:
        
        if encoder_type == 'lstm':
            # TODO: Implement LSTM encoder
            # self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, 
            #                    batch_first=True, dropout=dropout)
            # self.projection = nn.Linear(hidden_dim, output_dim)
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=False,
            )
            self.projection = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )
            
        elif encoder_type == 'gru':
            # TODO: Implement GRU encoder
            # Similar to LSTM
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=False,
            )
            self.projection = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )
            
        elif encoder_type == 'cnn':
            # TODO: Implement 1D CNN encoder
            # Stack of Conv1d -> BatchNorm -> ReLU -> Pool
            # Example:
            # self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
            # self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
            # self.pool = nn.AdaptiveAvgPool1d(1)
            self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2)
            self.bn1   = nn.BatchNorm1d(hidden_dim)
            self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
            self.bn2   = nn.BatchNorm1d(hidden_dim)
            self.pool  = nn.AdaptiveAvgPool1d(1)
            self.act   = nn.ReLU()
            self.drop  = nn.Dropout(dropout)
            self.projection = nn.Linear(hidden_dim, output_dim)
            
        elif encoder_type == 'transformer':
            # TODO: Implement Transformer encoder
            # encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4)
            # self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.in_proj = nn.Linear(input_dim, hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=4, batch_first=True, dropout=dropout
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.projection = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        
        # remove the global NotImplemented now that we implemented each variant
        # (do nothing)

    def forward(
        self,
        sequence: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode variable-length sequences.
        
        Args:
            sequence: (batch_size, seq_len, input_dim) - input sequence
            lengths: Optional (batch_size,) - actual sequence lengths (for padding)
            
        Returns:
            encoding: (batch_size, output_dim) - fixed-size embedding
        """
        # TODO: Implement forward pass based on encoder_type
        # Handle variable-length sequences if lengths provided
        # Return fixed-size embedding via pooling or taking last hidden state
        if self.encoder_type in ('lstm', 'gru'):
            if lengths is not None:
                # pack padded sequence for efficient RNN
                lengths_cpu = lengths.to('cpu')
                packed = nn.utils.rnn.pack_padded_sequence(
                    sequence, lengths_cpu, batch_first=True, enforce_sorted=False
                )
                out, h = self.rnn(packed)
                out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
                # get last valid timestep for each sequence
                last_idxs = lengths_cpu.clamp_min(1) - 1  # (B,)
                last_hidden = out[torch.arange(out.size(0), device=out.device), last_idxs]
            else:
                out, h = self.rnn(sequence)
                last_hidden = out[:, -1, :]
            return self.projection(last_hidden)

        elif self.encoder_type == 'cnn':
            # (B,T,D) -> (B,D,T)
            x = sequence.transpose(1, 2)
            x = self.act(self.bn1(self.conv1(x)))
            x = self.drop(self.act(self.bn2(self.conv2(x))))
            x = self.pool(x).squeeze(-1)  # (B, hidden_dim)
            return self.projection(x)

        elif self.encoder_type == 'transformer':
            x = self.in_proj(sequence)  # (B,T,H)
            # optional key-padding mask from lengths
            key_padding_mask = None
            if lengths is not None:
                B, T, _ = x.shape
                # True where position is PAD (to be ignored)
                key_padding_mask = torch.arange(T, device=x.device).expand(B, T) >= lengths.unsqueeze(1)
            x = self.transformer(x, src_key_padding_mask=key_padding_mask)  # (B,T,H)
            if lengths is not None:
                # masked mean over valid steps
                B, T, H = x.shape
                mask = (~key_padding_mask).float().unsqueeze(-1)  # (B,T,1)
                x_sum = (x * mask).sum(dim=1)
                denom = mask.sum(dim=1).clamp_min(1.0)
                pooled = x_sum / denom
            else:
                pooled = x.mean(dim=1)
            return self.projection(pooled)

        else:
            raise ValueError(f"Unknown encoder type: {self.encoder_type}")


class FrameEncoder(nn.Module):
    """
    Encoder for frame-based data (e.g., video features).
    
    Aggregates frame-level features into video-level embedding.
    """
    
    def __init__(
        self,
        frame_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 64,
        temporal_pooling: str = 'attention',
        dropout: float = 0.1
    ):
        """
        Args:
            frame_dim: Dimension of per-frame features
            hidden_dim: Hidden dimension
            output_dim: Output embedding dimension
            temporal_pooling: How to pool frames ['average', 'max', 'attention']
            dropout: Dropout probability
        """
        super().__init__()
        self.temporal_pooling = temporal_pooling
        
        # TODO: Implement frame encoder
        # 1. Frame-level processing (optional MLP)
        # 2. Temporal aggregation (pooling or attention)
        self.frame_mlp = nn.Sequential(
            nn.Linear(frame_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        if temporal_pooling == 'attention':
            # TODO: Implement attention-based pooling
            # Learn which frames are important
            # self.attention = nn.Linear(frame_dim, 1)
            self.attention = nn.Linear(hidden_dim, 1)
        elif temporal_pooling in ['average', 'max']:
            # Simple pooling, no learnable parameters needed
            self.attention = None
        else:
            raise ValueError(f"Unknown pooling: {temporal_pooling}")
        
        # TODO: Add projection layer
        # self.projection = nn.Sequential(...)
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(
        self,
        frames: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode sequence of frames.
        
        Args:
            frames: (batch_size, num_frames, frame_dim) - frame features
            mask: Optional (batch_size, num_frames) - valid frame mask
            
        Returns:
            encoding: (batch_size, output_dim) - video-level embedding
        """
        # TODO: Implement forward pass
        # 1. Process frames (optional)
        # 2. Apply temporal pooling
        # 3. Project to output dimension
        x = self.frame_mlp(frames)  # (B,F,H)
        if self.temporal_pooling == 'attention':
            pooled = self.attention_pool(x, mask)  # (B,H)
        elif self.temporal_pooling == 'average':
            if mask is not None:
                m = mask.float().unsqueeze(-1)  # (B,F,1)
                x_sum = (x * m).sum(dim=1)
                denom = m.sum(dim=1).clamp_min(1.0)
                pooled = x_sum / denom
            else:
                pooled = x.mean(dim=1)
        else:  # 'max'
            if mask is not None:
                # set invalid frames to very negative before max
                very_neg = torch.finfo(x.dtype).min
                x = x.masked_fill(~mask.bool().unsqueeze(-1), very_neg)
            pooled = x.max(dim=1).values
        return self.projection(pooled)
    
    def attention_pool(
        self,
        frames: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Pool frames using learned attention weights.
        
        Args:
            frames: (batch_size, num_frames, frame_dim)
            mask: Optional (batch_size, num_frames) - valid frames
            
        Returns:
            pooled: (batch_size, frame_dim) - attended frame features
        """
        # TODO: Implement attention pooling
        # 1. Compute attention scores for each frame
        # 2. Apply mask if provided
        # 3. Softmax to get weights
        # 4. Weighted sum of frames
        scores = self.attention(frames).squeeze(-1)  # (B,F)
        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), float('-inf'))
        weights = torch.softmax(scores, dim=-1)  # (B,F)
        pooled = torch.bmm(weights.unsqueeze(1), frames).squeeze(1)  # (B,H)
        return pooled


class SimpleMLPEncoder(nn.Module):
    """
    Simple MLP encoder for pre-extracted features.
    
    Use this when working with pre-computed features
    (e.g., ResNet features for images, MFCC for audio).
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        batch_norm: bool = True
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
            num_layers: Number of hidden layers
            dropout: Dropout probability
            batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        # TODO: Implement MLP encoder
        # Architecture: Input -> [Linear -> BatchNorm -> ReLU -> Dropout] x num_layers -> Output
        
        layers = []
        current_dim = input_dim
        
        # TODO: Add hidden layers
        # for i in range(num_layers):
        #     layers.append(nn.Linear(current_dim, hidden_dim))
        #     if batch_norm:
        #         layers.append(nn.BatchNorm1d(hidden_dim))
        #     layers.append(nn.ReLU())
        #     layers.append(nn.Dropout(dropout))
        #     current_dim = hidden_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        
        # TODO: Add output layer
        # layers.append(nn.Linear(current_dim, output_dim))
        layers.append(nn.Linear(current_dim, output_dim))
        
        # self.encoder = nn.Sequential(*layers)
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Encode features through MLP.
        
        Args:
            features: (batch_size, input_dim) - input features
            
        Returns:
            encoding: (batch_size, output_dim) - encoded features
        """
        # TODO: Implement forward pass
        # return self.encoder(features)
        return self.encoder(features)


def build_encoder(
    modality: str,
    input_dim: int,
    output_dim: int,
    encoder_config: dict = None
) -> nn.Module:
    """
    Factory function to build appropriate encoder for each modality.
    
    Args:
        modality: Modality name ('video', 'audio', 'imu', etc.)
        input_dim: Input feature dimension
        output_dim: Output embedding dimension
        encoder_config: Optional config dict with encoder hyperparameters
        
    Returns:
        Encoder module appropriate for the modality
    """
    if encoder_config is None:
        encoder_config = {}
    
    # TODO: Implement encoder selection logic
    # Example heuristics:
    # - 'video' -> FrameEncoder
    # - 'imu', 'audio', 'mocap' -> SequenceEncoder
    # - Pre-extracted features -> SimpleMLPEncoder
    
    if modality in ['video', 'frames']:
        return FrameEncoder(
            frame_dim=input_dim,
            output_dim=output_dim,
            **encoder_config
        )
    #elif modality in ['imu_hand', 'imu_chest', 'imu_ankle', 'heart_rate']:
    elif modality in ['imu_hand','imu_chest','heart_rate']:
        return SequenceEncoder(
            input_dim=input_dim,
            output_dim=output_dim,
            **encoder_config
        )
    else:
        # Default to MLP for unknown modalities
        return SimpleMLPEncoder(
            input_dim=input_dim,
            output_dim=output_dim,
            **encoder_config
        )


if __name__ == '__main__':
    # Test encoders
    print("Testing encoders...")
    
    batch_size = 4
    seq_len = 100
    input_dim = 17
    output_dim = 128
    
    # Test SequenceEncoder
    print("\nTesting SequenceEncoder...")
    for enc_type in ['lstm', 'gru', 'cnn']:
        try:
            encoder = SequenceEncoder(
                input_dim=input_dim,
                output_dim=output_dim,
                encoder_type=enc_type
            )
            
            sequence = torch.randn(batch_size, seq_len, input_dim)
            output = encoder(sequence)
            
            assert output.shape == (batch_size, output_dim)
            print(f"✓ {enc_type} encoder working! Output shape: {output.shape}")
            
        except NotImplementedError:
            print(f"✗ {enc_type} encoder not implemented yet")
        except Exception as e:
            print(f"✗ {enc_type} encoder error: {e}")
    
    # Test FrameEncoder
    print("\nTesting FrameEncoder...")
    try:
        num_frames = 30
        frame_dim = 512
        
        encoder = FrameEncoder(
            frame_dim=frame_dim,
            output_dim=output_dim,
            temporal_pooling='attention'
        )
        
        frames = torch.randn(batch_size, num_frames, frame_dim)
        output = encoder(frames)
        
        assert output.shape == (batch_size, output_dim)
        print(f"✓ FrameEncoder working! Output shape: {output.shape}")
        
    except NotImplementedError:
        print("✗ FrameEncoder not implemented yet")
    except Exception as e:
        print(f"✗ FrameEncoder error: {e}")
    
    # Test SimpleMLPEncoder
    print("\nTesting SimpleMLPEncoder...")
    try:
        encoder = SimpleMLPEncoder(
            input_dim=input_dim,
            output_dim=output_dim
        )
        
        features = torch.randn(batch_size, input_dim)
        output = encoder(features)
        
        assert output.shape == (batch_size, output_dim)
        print(f"✓ SimpleMLPEncoder working! Output shape: {output.shape}")
        
    except NotImplementedError:
        print("✗ SimpleMLPEncoder not implemented yet")
    except Exception as e:
        print(f"✗ SimpleMLPEncoder error: {e}")
