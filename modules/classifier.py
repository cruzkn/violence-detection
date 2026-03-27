"""
modules/03_classifier.py
========================
Module 3: Violence classification models.

Provides two architectures:
  1. BiLSTMClassifier   — Bidirectional LSTM (fast, good baseline)
  2. STTransformer      — Spatial-Temporal Transformer (higher accuracy)

Both accept input of shape (batch, seq_len, feature_dim) and output
(batch, 2) logits for binary classification (Fight / NonFight).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ══════════════════════════════════════════════════════════════════════════════
# Model 1: Bidirectional LSTM
# ══════════════════════════════════════════════════════════════════════════════
class BiLSTMClassifier(nn.Module):
    """
    Two-layer Bidirectional LSTM with attention pooling.

    Architecture:
        Input → LayerNorm → BiLSTM(×2) → Self-Attention Pool → MLP → Logits

    Args:
        input_dim:   feature vector size per timestep
        hidden_dim:  LSTM hidden units (default 256)
        num_layers:  number of LSTM layers (default 2)
        num_classes: 2 (Fight / NonFight)
        dropout:     dropout probability
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_out_dim = hidden_dim * 2     # bidirectional

        # Attention pooling over time dimension
        self.attn = nn.Linear(lstm_out_dim, 1)

        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_out_dim),
            nn.Linear(lstm_out_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, T, input_dim)
        Returns:
            logits: (batch, num_classes)
            attn_weights: (batch, T) for explainability
        """
        x = self.input_norm(x)
        lstm_out, _ = self.lstm(x)                    # (B, T, 2*H)

        # Attention pooling
        attn_scores = self.attn(lstm_out).squeeze(-1) # (B, T)
        attn_weights = F.softmax(attn_scores, dim=1)  # (B, T)
        context = (lstm_out * attn_weights.unsqueeze(-1)).sum(dim=1)  # (B, 2H)

        logits = self.classifier(context)             # (B, num_classes)
        return logits, attn_weights


# ══════════════════════════════════════════════════════════════════════════════
# Model 2: Spatial-Temporal Transformer
# ══════════════════════════════════════════════════════════════════════════════
class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 200, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))     # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class STTransformer(nn.Module):
    """
    Spatial-Temporal Transformer for skeleton-based violence detection.

    Architecture:
        Input projection → Positional Encoding →
        TransformerEncoder (N layers) → CLS token → MLP → Logits

    Args:
        input_dim:   feature vector size per timestep
        d_model:     transformer embedding dimension (default 128)
        nhead:       number of attention heads (default 4)
        num_layers:  number of transformer encoder layers (default 3)
        dim_ff:      feedforward dimension (default 256)
        num_classes: 2
        dropout:     dropout probability
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_ff: int = 256,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        # Input projection
        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, d_model),
            nn.GELU(),
        )

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,           # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: (batch, T, input_dim)
        Returns:
            logits:      (batch, num_classes)
            attn_weights: None (use transformer attention for XAI externally)
        """
        B = x.size(0)
        x = self.input_proj(x)                        # (B, T, d_model)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)        # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)                # (B, T+1, d_model)
        x = self.pos_enc(x)

        x = self.transformer(x)                       # (B, T+1, d_model)
        cls_out = x[:, 0]                             # CLS token output

        logits = self.classifier(cls_out)             # (B, num_classes)
        return logits, None


# ══════════════════════════════════════════════════════════════════════════════
# Model factory
# ══════════════════════════════════════════════════════════════════════════════
def build_model(
    arch: str,
    input_dim: int,
    device: str = "cpu",
    **kwargs,
) -> nn.Module:
    """
    Factory function. arch = 'lstm' or 'transformer'.

    Example:
        model = build_model('lstm', input_dim=204, device='mps')
        model = build_model('transformer', input_dim=204, d_model=128)
    """
    arch = arch.lower()
    if arch in ("lstm", "bilstm"):
        model = BiLSTMClassifier(input_dim=input_dim, **kwargs)
    elif arch in ("transformer", "st_transformer"):
        model = STTransformer(input_dim=input_dim, **kwargs)
    else:
        raise ValueError(f"Unknown arch: {arch}. Choose 'lstm' or 'transformer'")

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] {arch.upper()} | params: {n_params:,} | device: {device}")
    return model


# ── Quick smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import torch
    B, T, F = 4, 30, 204      # batch=4, seq=30 frames, feature_dim=204

    for arch in ["lstm", "transformer"]:
        model = build_model(arch, input_dim=F)
        x = torch.randn(B, T, F)
        logits, attn = model(x)
        print(f"  {arch}: logits={logits.shape}  attn={attn.shape if attn is not None else None}")
        probs = torch.softmax(logits, dim=-1)
        print(f"  probs[:2]: {probs[:2].detach().numpy().round(3)}")
