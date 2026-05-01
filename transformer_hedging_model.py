"""
================================================================================
  TimeSeriesTransformer + AsymmetricHedgingLoss + Training Loop
  For: PyTorch-based Deep RL — Corporate Hedging & Risk Management
  Author: Expert Quantitative AI Researcher
  Version: 1.0.0
================================================================================

PIPELINE POSITION
-----------------
  macro_data_pipeline.py      →  tensor (T, 12)
  sliding_window_dataset.py   →  DataLoader  →  (B, W, N) / (B, H, N) batches
         ↓
  transformer_hedging_model.py  ←  YOU ARE HERE
         ↓
  DRL Environment / Portfolio Optimizer

ARCHITECTURE OVERVIEW
---------------------
                    ┌──────────────────────────────────────────────────┐
  x (B, W, N=12)   │                                                  │
        │           │  1. InputProjection  Linear(12 → d_model=64)    │
        ▼           │          ↓                                       │
  (B, W, 64)        │  2. PositionalEncoding  (sinusoidal, learnable)  │
        │           │          ↓                                       │
        ▼           │  3. TransformerEncoder  (n_layers=3)             │
  (B, W, 64)        │     ┌─ MultiHeadAttention (n_heads=4)           │
        │           │     ├─ FeedForward       (dim_ff=256)            │
        │           │     ├─ LayerNorm + Dropout                       │
        ▼           │     └─ × 3 layers                                │
  (B, W, 64)        │          ↓                                       │
        │           │  4. TemporalPooling  mean over W dimension       │
        ▼           │          ↓                                       │
  (B, 64)           │  5. OutputProjection  Linear(64 → H×N)          │
        │           │          ↓                                       │
        ▼           │  6. Reshape → (B, H=5, N=12)                    │
  ŷ (B, H, N=12)   │                                                  │
                    └──────────────────────────────────────────────────┘

LOSS FUNCTION — ASYMMETRIC MSE
-------------------------------
  Standard MSE treats over- and under-prediction symmetrically.
  In corporate hedging, UNDER-hedging (actual > predicted shock) is
  catastrophically more expensive than OVER-hedging.

  ε  = actual − predicted    (residual)

  ε > 0  →  actual > predicted  →  UNDER-HEDGING  →  weight = α   (high, e.g. 0.8)
  ε < 0  →  actual < predicted  →  OVER-HEDGING   →  weight = 1-α (low,  e.g. 0.2)

  L = α·mean(ε²  where ε>0)  +  (1−α)·mean(ε²  where ε≤0)
"""

# ── Standard Library ──────────────────────────────────────────────────────────
import math
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List

# ── Third-Party ───────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("HedgingTransformer")


# ══════════════════════════════════════════════════════════════════════════════
# 0.  HYPERPARAMETER CONFIGURATION  (single source of truth)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ModelConfig:
    """
    All hyperparameters for the TimeSeriesTransformer.

    Keeping config in a dataclass makes it trivially serialisable (JSON/YAML)
    and eliminates magic numbers scattered across the codebase.
    """
    # ── Data dimensions ───────────────────────────────────────────────────────
    n_features:       int   = 12     # number of macro/market features (N)
    window_size:      int   = 60     # look-back length (W)
    forecast_horizon: int   = 5      # prediction horizon (H)

    # ── Transformer architecture ──────────────────────────────────────────────
    d_model:          int   = 64     # embedding / hidden dimension
    n_heads:          int   = 4      # attention heads  (d_model must be divisible)
    n_layers:         int   = 3      # number of TransformerEncoderLayers
    dim_feedforward:  int   = 256    # inner dimension of each layer's FFN
    dropout:          float = 0.1    # dropout probability

    # ── Positional encoding ───────────────────────────────────────────────────
    max_seq_len:      int   = 500    # upper bound for PE table; ≥ window_size
    pe_learnable:     bool  = False  # False → fixed sinusoidal; True → learned

    # ── Loss ─────────────────────────────────────────────────────────────────
    alpha:            float = 0.8    # under-hedge penalty weight (> 0.5 = stricter)

    # ── Training ─────────────────────────────────────────────────────────────
    learning_rate:    float = 1e-3
    weight_decay:     float = 1e-4   # L2 regularisation via AdamW
    batch_size:       int   = 32
    n_epochs:         int   = 10
    grad_clip:        float = 1.0    # max gradient norm (prevents exploding grads)
    device:           str   = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})."
        )
        assert 0.0 < self.alpha < 1.0, "alpha must be in (0, 1)."
        assert self.max_seq_len >= self.window_size, (
            "max_seq_len must be ≥ window_size."
        )


@dataclass
class TrainingHistory:
    """Accumulates per-epoch metrics for later plotting / analysis."""
    train_loss: List[float] = field(default_factory=list)
    val_loss:   List[float] = field(default_factory=list)
    epoch_time: List[float] = field(default_factory=list)

    def log_epoch(
        self,
        train_loss: float,
        val_loss:   float,
        elapsed:    float,
    ) -> None:
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.epoch_time.append(elapsed)

    def best_val_epoch(self) -> int:
        """Return 1-indexed epoch with lowest validation loss."""
        return int(np.argmin(self.val_loss)) + 1


# ══════════════════════════════════════════════════════════════════════════════
# 1.  POSITIONAL ENCODING
# ══════════════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """
    Injects positional information into the token embeddings.

    Transformers process all positions simultaneously (no recurrence), so they
    have NO built-in sense of token order.  Without PE, feeding the sequence
    forward or backward would produce identical attention weights.

    Two modes
    ----------
    Fixed sinusoidal (pe_learnable=False) — Vaswani et al., 2017
        PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
        ✓ No extra parameters.
        ✓ Generalises to sequences longer than seen during training.
        ✓ Preferred for financial time-series where order is deterministic.

    Learned (pe_learnable=True)
        A plain nn.Embedding of shape (max_seq_len, d_model).
        ✓ The model can adapt PE to the data distribution.
        ✗ Does NOT generalise beyond max_seq_len.

    Parameters
    ----------
    d_model      : model hidden dimension
    max_seq_len  : pre-compute PE up to this length
    dropout      : applied after adding PE to embeddings
    pe_learnable : if True, use a learned embedding instead of sinusoidal
    """

    def __init__(
        self,
        d_model:      int,
        max_seq_len:  int   = 500,
        dropout:      float = 0.1,
        pe_learnable: bool  = False,
    ) -> None:
        super().__init__()
        self.dropout      = nn.Dropout(p=dropout)
        self.pe_learnable = pe_learnable

        if pe_learnable:
            # Learned positional embeddings — shape (max_seq_len, d_model)
            self.pe_embed = nn.Embedding(max_seq_len, d_model)
        else:
            # ── Build fixed sinusoidal table ──────────────────────────────
            # pe: (max_seq_len, d_model)
            pe       = torch.zeros(max_seq_len, d_model)
            position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
            # div_term: (d_model // 2,)
            div_term = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float)
                * (-math.log(10_000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)   # even indices
            pe[:, 1::2] = torch.cos(position * div_term)   # odd  indices

            # Register as buffer: moves with .to(device) but NOT a parameter
            # Shape stored: (1, max_seq_len, d_model) — ready to broadcast over B
            self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (B, seq_len, d_model)

        Returns
        -------
        Tensor, shape (B, seq_len, d_model)  — embeddings + positional signal
        """
        seq_len = x.size(1)

        if self.pe_learnable:
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # (1, L)
            x = x + self.pe_embed(positions)                                  # (B, L, D)
        else:
            # self.pe: (1, max_seq_len, d_model) — slice to actual seq_len
            x = x + self.pe[:, :seq_len, :]                                   # (B, L, D)

        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  TIME-SERIES TRANSFORMER MODEL
# ══════════════════════════════════════════════════════════════════════════════

class TimeSeriesTransformer(nn.Module):
    """
    A Transformer Encoder model for multivariate financial time-series
    forecasting, designed as the perception backbone of a DRL hedging agent.

    Forward pass (shape trace)
    --------------------------
    Input   x          : (B, W,    N=12)   raw feature window from DataLoader
    InputProjection    : (B, W,    d=64)   project features → d_model
    PositionalEncoding : (B, W,    d=64)   add sequence-order signal
    TransformerEncoder : (B, W,    d=64)   contextual representations
    TemporalPooling    : (B,       d=64)   mean over W (sequence compression)
    OutputProjection   : (B,       H×N)    flatten forecast
    Reshape            : (B, H=5,  N=12)   final output

    Parameters
    ----------
    cfg : ModelConfig — all hyperparameters in one place
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # ── 1. Input Projection ───────────────────────────────────────────────
        # Projects from raw feature space (N=12) into the model's hidden space
        # (d_model=64).  A simple Linear layer + LayerNorm for training stability.
        self.input_projection = nn.Sequential(
            nn.Linear(cfg.n_features, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
        )

        # ── 2. Positional Encoding ────────────────────────────────────────────
        self.positional_encoding = PositionalEncoding(
            d_model      = cfg.d_model,
            max_seq_len  = cfg.max_seq_len,
            dropout      = cfg.dropout,
            pe_learnable = cfg.pe_learnable,
        )

        # ── 3. Transformer Encoder Stack ──────────────────────────────────────
        # Each layer: MultiHeadSelfAttention → Add&Norm → FFN → Add&Norm
        # batch_first=True → expects (B, seq, d_model) — matches our convention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = cfg.d_model,
            nhead           = cfg.n_heads,
            dim_feedforward = cfg.dim_feedforward,
            dropout         = cfg.dropout,
            activation      = "gelu",    # GELU outperforms ReLU on financial data
            batch_first     = True,      # CRITICAL: keep (B, W, D) convention
            norm_first      = True,      # Pre-LN: more stable gradients (Xiong 2020)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer = encoder_layer,
            num_layers    = cfg.n_layers,
            # Final LayerNorm applied after all layers
            norm          = nn.LayerNorm(cfg.d_model),
        )

        # ── 4. Output Projection ──────────────────────────────────────────────
        # After mean-pooling over W, we have (B, d_model).
        # Project to (B, H * N) then reshape to (B, H, N).
        out_dim = cfg.forecast_horizon * cfg.n_features
        self.output_projection = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model * 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model * 2, out_dim),
        )

        # ── Weight Initialisation ─────────────────────────────────────────────
        self._init_weights()
        self._log_parameter_count()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        """
        Xavier-uniform for Linear layers; zero-bias.
        Consistent initialisation reduces sensitivity to random seed.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _log_parameter_count(self) -> None:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"TimeSeriesTransformer  |  "
            f"total_params={total:,}  trainable={trainable:,}"
        )

    # ── Forward Pass ──────────────────────────────────────────────────────────

    def forward(
        self,
        x:           Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Parameters
        ----------
        x : FloatTensor, shape (B, W, N)
            A batch of historical windows from SlidingWindowDataset.
        src_key_padding_mask : BoolTensor (B, W) | None
            True at positions to IGNORE (e.g., padded tokens). Usually None
            for fixed-window financial data (all positions are valid).

        Returns
        -------
        FloatTensor, shape (B, H, N)
            Predicted values for the next H timesteps across all N features.
        """
        B, W, N = x.shape

        # ── Step 1: Project features → d_model ───────────────────────────
        # (B, W, N) → (B, W, d_model)
        x = self.input_projection(x)

        # ── Step 2: Add positional signal ─────────────────────────────────
        # (B, W, d_model) → (B, W, d_model)  [values shifted by PE]
        x = self.positional_encoding(x)

        # ── Step 3: Transformer Encoder ───────────────────────────────────
        # Self-attention: each timestep attends to every other timestep.
        # (B, W, d_model) → (B, W, d_model)
        x = self.transformer_encoder(
            x,
            src_key_padding_mask=src_key_padding_mask,
        )

        # ── Step 4: Temporal Pooling ──────────────────────────────────────
        # Compress the W dimension via mean pooling.
        # Alternatives: x[:, -1, :] (last token — causal) or max-pool.
        # Mean-pool is more robust for smooth financial trends.
        # (B, W, d_model) → (B, d_model)
        x = x.mean(dim=1)

        # ── Step 5: Output Projection ─────────────────────────────────────
        # (B, d_model) → (B, H × N)
        x = self.output_projection(x)

        # ── Step 6: Reshape to forecast grid ──────────────────────────────
        # (B, H × N) → (B, H, N)
        x = x.view(B, self.cfg.forecast_horizon, self.cfg.n_features)

        return x

    # ── Convenience helpers ───────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, x: Tensor) -> Tensor:
        """Inference-mode forward pass (no gradient tracking)."""
        self.eval()
        return self.forward(x)

    def count_parameters(self) -> Dict[str, int]:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


# ══════════════════════════════════════════════════════════════════════════════
# 3.  ASYMMETRIC HEDGING LOSS
# ══════════════════════════════════════════════════════════════════════════════

class AsymmetricHedgingLoss(nn.Module):
    """
    Asymmetric Mean Squared Error loss tailored for corporate hedging.

    Motivation
    ----------
    Standard MSE: L = mean((y - ŷ)²)
    This penalises over-prediction and under-prediction equally.

    In corporate risk management this symmetry is WRONG:
      • Under-hedging (actual shock > predicted) → the company bears the
        full unhedged loss, which can be catastrophic (margin calls,
        covenant breaches, forced deleveraging).
      • Over-hedging  (actual shock < predicted) → the company pays
        unnecessary hedging premium (option cost or swap MTM loss),
        which is painful but bounded and survivable.

    Therefore we penalise under-hedge residuals by factor α and
    over-hedge residuals by factor (1 − α), where α > 0.5.

    Mathematical Definition
    -----------------------
    Let  ε = y − ŷ   (residual; positive ⟹ actual > predicted ⟹ under-hedge)

    w(ε) = α       if  ε > 0   (under-hedge — heavier penalty)
           1 − α   if  ε ≤ 0   (over-hedge  — lighter penalty)

    L = mean( w(ε) · ε² )

    Gradient behaviour
    ------------------
    ∂L/∂ŷ = −2 · w(ε) · ε

    Because α > 0.5, the gradient is steeper when the model
    under-predicts, pushing it toward conservative (higher) forecasts.

    Parameters
    ----------
    alpha     : float in (0.5, 1.0)
        Under-hedge penalty weight.  Default 0.8.
        α = 0.5  →  reduces to standard MSE.
        α → 1.0  →  almost ignores over-hedging; maximally conservative.
    reduction : "mean" | "sum" | "none"
        How to aggregate the element-wise weighted losses.
    """

    def __init__(
        self,
        alpha:     float = 0.8,
        reduction: str   = "mean",
    ) -> None:
        super().__init__()
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none'.")

        self.alpha     = alpha
        self.reduction = reduction

        logger.info(
            f"AsymmetricHedgingLoss  |  "
            f"alpha={alpha}  under_hedge_weight={alpha:.2f}  "
            f"over_hedge_weight={1 - alpha:.2f}  reduction={reduction}"
        )

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        Parameters
        ----------
        y_pred : FloatTensor (B, H, N)  — model forecasts
        y_true : FloatTensor (B, H, N)  — ground-truth future values

        Returns
        -------
        Scalar loss tensor (if reduction != 'none'),
        or element-wise loss tensor of shape (B, H, N).
        """
        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"Shape mismatch: y_pred={tuple(y_pred.shape)}, "
                f"y_true={tuple(y_true.shape)}."
            )

        # ε = actual − predicted   (positive ⟹ under-hedge)
        residuals = y_true - y_pred                              # (B, H, N)

        # Squared residuals
        sq_residuals = residuals.pow(2)                          # (B, H, N)

        # ── Asymmetric weight mask ────────────────────────────────────────
        # under_mask: True where actual > predicted (under-hedge)
        under_mask = (residuals > 0).float()                     # (B, H, N)
        over_mask  = 1.0 - under_mask                            # complement

        # Weighted squared errors
        weighted = (
            self.alpha       * under_mask * sq_residuals
            + (1 - self.alpha) * over_mask  * sq_residuals
        )                                                        # (B, H, N)

        # ── Reduction ─────────────────────────────────────────────────────
        if self.reduction == "mean":
            return weighted.mean()
        elif self.reduction == "sum":
            return weighted.sum()
        else:   # "none"
            return weighted

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}, reduction='{self.reduction}'"

    @staticmethod
    def symmetry_check(alpha: float = 0.5) -> None:
        """Verify that alpha=0.5 exactly reproduces standard MSE."""
        loss_fn  = AsymmetricHedgingLoss(alpha=0.5)
        mse_fn   = nn.MSELoss()
        y_pred   = torch.randn(16, 5, 12)
        y_true   = torch.randn(16, 5, 12)
        asym_val = loss_fn(y_pred, y_true)
        mse_val  = mse_fn(y_pred, y_true)
        assert torch.allclose(asym_val, mse_val, atol=1e-5), (
            f"Symmetry check failed: AsymLoss={asym_val:.6f}, MSE={mse_val:.6f}"
        )
        logger.info(
            f"Symmetry check passed: AsymLoss(α=0.5)={asym_val:.6f} "
            f"≈ MSE={mse_val:.6f}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 4.  TRAINING UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

class EarlyStopping:
    """
    Halts training when validation loss stops improving.

    Parameters
    ----------
    patience  : int   — epochs to wait after last improvement
    min_delta : float — minimum change to qualify as improvement
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-5) -> None:
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float("inf")
        self.counter    = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        """Returns True when training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def _run_one_epoch(
    model:      TimeSeriesTransformer,
    loader:     DataLoader,
    loss_fn:    AsymmetricHedgingLoss,
    optimizer:  Optional[torch.optim.Optimizer],
    cfg:        ModelConfig,
    is_train:   bool,
) -> float:
    """
    Run a single training or validation epoch.

    Parameters
    ----------
    model     : the Transformer model
    loader    : DataLoader (train or val)
    loss_fn   : AsymmetricHedgingLoss instance
    optimizer : AdamW (None during validation)
    cfg       : ModelConfig with grad_clip, device
    is_train  : True → compute gradients & update weights

    Returns
    -------
    float — mean loss over all batches in this epoch
    """
    model.train(is_train)
    context = torch.enable_grad() if is_train else torch.no_grad()
    total_loss  = 0.0
    n_batches   = 0

    with context:
        for x_batch, y_batch in loader:
            # Move to target device (GPU/CPU)
            x_batch = x_batch.to(cfg.device, non_blocking=True)
            y_batch = y_batch.to(cfg.device, non_blocking=True)

            # ── Forward pass ──────────────────────────────────────────────
            y_pred = model(x_batch)                  # (B, H, N)
            loss   = loss_fn(y_pred, y_batch)

            # ── Backward pass (train only) ────────────────────────────────
            if is_train and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)   # faster than zero_grad()
                loss.backward()
                # Gradient clipping — essential for Transformers on volatile
                # financial time-series (prevents exploding gradients)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

    return total_loss / max(n_batches, 1)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  MAIN TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def train_model(
    cfg:          ModelConfig,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    checkpoint_path: str = "best_hedging_model.pt",
) -> Tuple[TimeSeriesTransformer, TrainingHistory]:
    """
    Full supervised training loop for the TimeSeriesTransformer.

    Steps per epoch
    ---------------
    1. Train  : forward → asymmetric loss → backward → AdamW → grad-clip
    2. Val    : forward → asymmetric loss (no gradients)
    3. LR Scheduler step (ReduceLROnPlateau on val_loss)
    4. Early Stopping check
    5. Checkpoint if val_loss improved

    Parameters
    ----------
    cfg              : ModelConfig  — all hyperparameters
    train_loader     : DataLoader   — training windows
    val_loader       : DataLoader   — validation windows
    checkpoint_path  : str          — where to save best weights

    Returns
    -------
    model   : trained TimeSeriesTransformer (best weights loaded)
    history : TrainingHistory  — lists of train/val loss per epoch
    """
    logger.info(f"{'═'*64}")
    logger.info("  TRAINING START")
    logger.info(f"  Device       : {cfg.device}")
    logger.info(f"  Epochs       : {cfg.n_epochs}")
    logger.info(f"  Batch size   : {cfg.batch_size}")
    logger.info(f"  d_model      : {cfg.d_model}")
    logger.info(f"  n_layers     : {cfg.n_layers}")
    logger.info(f"  Loss alpha   : {cfg.alpha}  (under-hedge penalty)")
    logger.info(f"{'═'*64}")

    # ── Instantiate model, loss, optimiser, scheduler ─────────────────────────
    model = TimeSeriesTransformer(cfg).to(cfg.device)

    loss_fn = AsymmetricHedgingLoss(alpha=cfg.alpha, reduction="mean")

    # AdamW: Adam + decoupled weight decay (Loshchilov & Hutter, 2019)
    # Superior to Adam for Transformers trained on noisy financial data
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg.learning_rate,
        weight_decay = cfg.weight_decay,
        betas        = (0.9, 0.98),   # standard Transformer betas
        eps          = 1e-9,
    )

    # ReduceLROnPlateau: halves LR if val_loss doesn't drop for `patience` epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode     = "min",
        factor   = 0.5,
        patience = 3,
        verbose  = True,
    )

    early_stopper = EarlyStopping(patience=7, min_delta=1e-5)
    history       = TrainingHistory()
    best_val_loss = float("inf")

    # ── Epoch loop ────────────────────────────────────────────────────────────
    for epoch in range(1, cfg.n_epochs + 1):
        t0 = time.time()

        train_loss = _run_one_epoch(
            model, train_loader, loss_fn, optimizer, cfg, is_train=True
        )
        val_loss = _run_one_epoch(
            model, val_loader, loss_fn, optimizer=None, cfg=cfg, is_train=False
        )

        elapsed = time.time() - t0
        history.log_epoch(train_loss, val_loss, elapsed)

        # ── LR scheduling ─────────────────────────────────────────────────
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # ── Console report ────────────────────────────────────────────────
        logger.info(
            f"Epoch {epoch:>3}/{cfg.n_epochs}  |  "
            f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  |  "
            f"lr={current_lr:.2e}  |  t={elapsed:.1f}s"
        )

        # ── Checkpoint best model ──────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch":      epoch,
                    "state_dict": model.state_dict(),
                    "val_loss":   val_loss,
                    "cfg":        cfg,
                },
                checkpoint_path,
            )
            logger.info(f"  ✓ New best val_loss={val_loss:.6f} — checkpoint saved.")

        # ── Early stopping ─────────────────────────────────────────────────
        if early_stopper.step(val_loss):
            logger.info(
                f"  Early stopping triggered at epoch {epoch} "
                f"(patience={early_stopper.patience})."
            )
            break

    # ── Load best weights before returning ────────────────────────────────────
    checkpoint = torch.load(checkpoint_path, map_location=cfg.device)
    model.load_state_dict(checkpoint["state_dict"])
    logger.info(
        f"{'═'*64}\n"
        f"  Training complete.  Best val_loss={best_val_loss:.6f}  "
        f"at epoch {history.best_val_epoch()}\n"
        f"{'═'*64}"
    )

    return model, history


# ══════════════════════════════════════════════════════════════════════════════
# 6.  ENTRY POINT — SMOKE TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── 6a. Instantiate config ────────────────────────────────────────────────
    cfg = ModelConfig(
        n_features       = 12,
        window_size      = 60,
        forecast_horizon = 5,
        d_model          = 64,
        n_heads          = 4,
        n_layers         = 3,
        dim_feedforward  = 256,
        dropout          = 0.1,
        alpha            = 0.8,
        learning_rate    = 1e-3,
        weight_decay     = 1e-4,
        batch_size       = 32,
        n_epochs         = 5,          # small for smoke test; use 50–200 in prod
        grad_clip        = 1.0,
    )

    # ── 6b. Verify loss symmetry (α=0.5 must reproduce MSE) ──────────────────
    AsymmetricHedgingLoss.symmetry_check()

    # ── 6c. Build dummy DataLoaders (replace with build_dataloaders() in prod) ─
    torch.manual_seed(42)
    T = 1_300   # ~5 years of daily data

    # Simulate the scaled macro tensor
    macro_tensor = torch.rand(T, cfg.n_features)

    # Build (x, y) pairs manually to avoid importing the full pipeline here
    W, H = cfg.window_size, cfg.forecast_horizon
    xs = torch.stack([macro_tensor[i     : i + W]     for i in range(T - W - H + 1)])
    ys = torch.stack([macro_tensor[i + W : i + W + H] for i in range(T - W - H + 1)])

    n          = len(xs)
    n_train    = int(n * 0.70)
    n_val      = int(n * 0.15)

    # ⚠️ NO SHUFFLE — chronological integrity is mandatory for time-series
    loader_kwargs = dict(batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    train_loader  = DataLoader(
        TensorDataset(xs[:n_train],            ys[:n_train]),           **loader_kwargs
    )
    val_loader    = DataLoader(
        TensorDataset(xs[n_train:n_train+n_val], ys[n_train:n_train+n_val]), **loader_kwargs
    )
    test_loader   = DataLoader(
        TensorDataset(xs[n_train+n_val:],       ys[n_train+n_val:]),     **loader_kwargs
    )

    # ── 6d. Quick forward-pass shape check before training ────────────────────
    print(f"\n{'═'*64}")
    print("  FORWARD PASS SHAPE CHECK")
    print(f"{'═'*64}")

    model    = TimeSeriesTransformer(cfg)
    x_sample = torch.rand(cfg.batch_size, cfg.window_size, cfg.n_features)
    y_sample = torch.rand(cfg.batch_size, cfg.forecast_horizon, cfg.n_features)

    with torch.no_grad():
        out = model(x_sample)

    print(f"  Input  x : {tuple(x_sample.shape)}")
    print(f"  Output ŷ : {tuple(out.shape)}")
    print(f"  Target y : {tuple(y_sample.shape)}")
    assert out.shape == y_sample.shape, "Output shape mismatch!"
    print("  ✓ Shapes match.\n")

    # ── 6e. Loss sanity check ─────────────────────────────────────────────────
    print(f"{'═'*64}")
    print("  LOSS FUNCTION SANITY CHECK")
    print(f"{'═'*64}")

    loss_fn     = AsymmetricHedgingLoss(alpha=cfg.alpha)
    loss_val    = loss_fn(out, y_sample)
    mse_val     = F.mse_loss(out, y_sample)

    print(f"  AsymHedgeLoss (α={cfg.alpha}) = {loss_val.item():.6f}")
    print(f"  Standard MSE                  = {mse_val.item():.6f}")
    print(f"  Ratio (Asym / MSE)            = {loss_val.item() / mse_val.item():.4f}")
    print(f"  (Ratio should be between {1 - cfg.alpha:.1f} and {cfg.alpha:.1f})\n")

    # ── 6f. Full training loop ────────────────────────────────────────────────
    print(f"{'═'*64}")
    print("  TRAINING LOOP")
    print(f"{'═'*64}\n")

    trained_model, history = train_model(
        cfg              = cfg,
        train_loader     = train_loader,
        val_loader       = val_loader,
        checkpoint_path  = "best_hedging_model.pt",
    )

    # ── 6g. Test-set evaluation ───────────────────────────────────────────────
    print(f"\n{'═'*64}")
    print("  TEST SET EVALUATION")
    print(f"{'═'*64}")

    test_loss = _run_one_epoch(
        trained_model, test_loader, loss_fn,
        optimizer=None, cfg=cfg, is_train=False
    )
    print(f"  Final test AsymHedgeLoss (α={cfg.alpha}) = {test_loss:.6f}")

    # ── 6h. Print training history ────────────────────────────────────────────
    print(f"\n{'═'*64}")
    print("  TRAINING HISTORY")
    print(f"{'─'*64}")
    print(f"  {'Epoch':>5}  {'Train Loss':>12}  {'Val Loss':>12}  {'Time (s)':>9}")
    print(f"{'─'*64}")
    for i, (tr, vl, t) in enumerate(
        zip(history.train_loss, history.val_loss, history.epoch_time), 1
    ):
        print(f"  {i:>5}  {tr:>12.6f}  {vl:>12.6f}  {t:>8.2f}s")
    print(f"{'═'*64}")
    print(f"  Best val epoch : {history.best_val_epoch()}")
    print(f"  Best val loss  : {min(history.val_loss):.6f}")
    print(f"{'═'*64}\n")

    # ── 6i. Production integration note ──────────────────────────────────────
    PRODUCTION_NOTE = """
# ── In production, replace the dummy loaders above with: ──────────────────

from macro_data_pipeline    import MacroDataPipeline
from sliding_window_dataset import build_dataloaders

pipeline = MacroDataPipeline(fred_api_key=..., evds_api_key=...)
_, _, tensor, scaler = pipeline.run()          # tensor: (T, 12)

train_loader, val_loader, test_loader, _ = build_dataloaders(
    tensor=tensor, window_size=60, forecast_horizon=5, batch_size=32
)

cfg           = ModelConfig()
trained_model, history = train_model(cfg, train_loader, val_loader)

# Inference on a new window
x_new = tensor[-60:].unsqueeze(0)             # (1, 60, 12)
y_hat = trained_model.predict(x_new)          # (1,  5, 12) — scaled
y_hat_original = scaler.inverse_transform(
    y_hat.squeeze(0).numpy().reshape(-1, 12)
).reshape(5, 12)                              # back to original price space
    """
    print(PRODUCTION_NOTE)
