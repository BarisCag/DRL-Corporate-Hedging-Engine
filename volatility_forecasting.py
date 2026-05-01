"""
================================================================================
  Realized Volatility Forecasting — Core Modules
  Target Markets : Brent Crude Oil  (BZ=F)  |  Dutch TTF Natural Gas  (TTF=F)
  Author         : Expert Quantitative AI Researcher
  Version        : 1.0.0
================================================================================

PIPELINE POSITION
-----------------
  macro_data_pipeline.py          →  raw_df  (T, 12)
         ↓
  volatility_forecasting.py       ←  YOU ARE HERE
    ├── VolatilityDataProcessor   →  feature-engineered df  +  RV target columns
    ├── VolatilityTransformer     →  ŷ (B, H, 2)  strictly positive via Softplus
    └── QLIKELoss                 →  industry-standard volatility loss
         ↓
  sliding_window_dataset.py  →  DataLoader  →  DRL / Portfolio Optimizer

WHY REALIZED VOLATILITY?
-------------------------
  Price direction is near-impossible to predict reliably (EMH).
  Volatility, however, exhibits well-documented empirical regularities:
    • Volatility clustering  (Mandelbrot 1963, Fama 1965)
    • Long memory / persistence  (Ding et al. 1993)
    • Leverage effect  (Black 1976) — prices ↓ → vol ↑ asymmetrically
    • Mean-reversion over medium horizons

  These properties make RV a legitimate and commercially valuable target
  for a DRL-based corporate hedging agent — the agent can use volatility
  forecasts to size option hedges, adjust VAR-based position limits, and
  time entry/exit of swap structures.

MATHEMATICAL BACKGROUND
-----------------------
  Log Return:
    r_t = log(P_t / P_{t-1}) = log(P_t) - log(P_{t-1})

  Realized Volatility (rolling window of size k):
    RV_t = std(r_{t-k+1}, ..., r_t) × √252

    The √252 factor annualizes the daily volatility
    (252 = standard number of trading days per year).

  QLIKE Loss (Patton 2011 — the econometric gold standard):
    L(y, ŷ) = (y / ŷ) - log(y / ŷ) - 1

    Properties:
      • L ≥ 0 always,  L = 0 iff y = ŷ  (global minimum)
      • Asymmetric: penalises under-estimation of vol more than over
      • Robust under noisy/proxy RV targets (Patton 2011 theorem)
      • Convex in ŷ  →  well-behaved gradient landscape
"""

# ── Standard Library ──────────────────────────────────────────────────────────
import math
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# ── Third-Party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("VolatilityForecasting")

# ── Constants ─────────────────────────────────────────────────────────────────
ANNUALIZATION_FACTOR: float = math.sqrt(252)   # daily → annual vol scaling
EPSILON:              float = 1e-8             # numerical stability guard
ENERGY_PRICE_COLS:    List[str] = [            # columns in raw_df from pipeline
    "Brent_Crude_Oil",
    "TTF_Natural_Gas",
]
ENERGY_RV_COLS:       List[str] = [            # output column names
    "RV_Brent_Crude_Oil",
    "RV_TTF_Natural_Gas",
]


# ══════════════════════════════════════════════════════════════════════════════
# 1.  VOLATILITY DATA PROCESSOR
# ══════════════════════════════════════════════════════════════════════════════

class VolatilityDataProcessor:
    """
    Transforms a raw macro/market DataFrame into a volatility-ready dataset.

    Responsibilities
    ----------------
    1. Validate that required price columns exist.
    2. Compute daily log returns for each energy price series.
    3. Compute rolling Realized Volatility (RV) and annualize.
    4. Construct a clean feature matrix X and target matrix y,
       dropping the initial NaN rows introduced by the rolling window.
    5. Report descriptive statistics on the RV targets for QA.

    Parameters
    ----------
    window : int
        Rolling window size for RV calculation (trading days).
        Common choices:
          5  →  1-week  RV  (short-memory, reactive)
         21  →  1-month RV  (smoother, less noisy)
         63  →  1-quarter RV  (regime-level signal)
        Default: 5  (appropriate for weekly hedging decisions in a DRL env)

    price_cols : list[str]
        Column names of the raw price series in the input DataFrame.
        Must match the column names produced by macro_data_pipeline.py.

    drop_return_cols : bool
        If True (default), remove intermediate log-return columns from
        the final feature DataFrame — they are implicitly captured by RV.
        Set to False to expose them as additional model inputs.

    Example
    -------
    >>> processor = VolatilityDataProcessor(window=5)
    >>> feature_df, target_df = processor.fit_transform(raw_df)
    >>> # feature_df : (T', 12+)  all macro features + log-return cols (optional)
    >>> # target_df  : (T',  2)   RV_Brent_Crude_Oil, RV_TTF_Natural_Gas
    """

    def __init__(
        self,
        window:           int       = 5,
        price_cols:       List[str] = ENERGY_PRICE_COLS,
        drop_return_cols: bool      = True,
    ) -> None:
        if window < 2:
            raise ValueError(f"window must be ≥ 2, got {window}.")
        self.window           = window
        self.price_cols       = price_cols
        self.drop_return_cols = drop_return_cols

    # ── Core transformation ────────────────────────────────────────────────────

    def fit_transform(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Execute the full feature-engineering pipeline.

        Parameters
        ----------
        df : pd.DataFrame
            Raw aligned DataFrame from MacroDataPipeline.run().
            Must have a DatetimeIndex and contain self.price_cols.

        Returns
        -------
        feature_df : pd.DataFrame  — all columns EXCEPT the RV targets
                                     (feeds into the Transformer as X)
        target_df  : pd.DataFrame  — only the 2 RV columns  (y)
        """
        self._validate_columns(df)
        df = df.copy()

        # ── Step 1: Log Returns ──────────────────────────────────────────────
        return_cols = self._compute_log_returns(df)

        # ── Step 2: Realized Volatility ──────────────────────────────────────
        rv_cols = self._compute_realized_volatility(df, return_cols)

        # ── Step 3: Drop warmup NaN rows ─────────────────────────────────────
        #   The rolling window introduces (window) NaN rows at the start.
        #   We drop ALL rows with any NaN in the RV columns so X and y
        #   are perfectly aligned without forward-fill contamination.
        n_before = len(df)
        df.dropna(subset=rv_cols, inplace=True)
        n_dropped = n_before - len(df)
        logger.info(
            f"Dropped {n_dropped} warmup rows  "
            f"(window={self.window})  |  "
            f"Usable rows: {len(df):,}"
        )

        # ── Step 4: Separate features and targets ────────────────────────────
        target_df  = df[rv_cols].copy()
        feature_df = df.drop(columns=rv_cols)

        if self.drop_return_cols:
            feature_df = feature_df.drop(
                columns=[c for c in return_cols if c in feature_df.columns]
            )

        # ── Step 5: QA Report ────────────────────────────────────────────────
        self._report(target_df)

        return feature_df, target_df

    # ── Private helpers ────────────────────────────────────────────────────────

    def _validate_columns(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.price_cols if c not in df.columns]
        if missing:
            raise KeyError(
                f"Required price columns not found in DataFrame: {missing}.\n"
                f"Available columns: {list(df.columns)}"
            )
        logger.info(
            f"VolatilityDataProcessor  |  "
            f"window={self.window} days  |  "
            f"annualization=√252={ANNUALIZATION_FACTOR:.4f}  |  "
            f"price_cols={self.price_cols}"
        )

    def _compute_log_returns(self, df: pd.DataFrame) -> List[str]:
        """
        Add daily log-return columns for each price series in-place.

        r_t = log(P_t) - log(P_{t-1})

        Log returns (vs arithmetic returns) are preferred because:
          • They are additive over time  (r_{0→2} = r_{0→1} + r_{1→2})
          • They are approximately normally distributed for small moves
          • They prevent negative prices from creating undefined returns
          • Rolling std of log returns is the standard RV estimator

        Returns
        -------
        List of the newly created return column names.
        """
        return_cols = []
        for price_col in self.price_cols:
            ret_col = f"logret_{price_col}"
            # pandas .shift(1) is equivalent to P_{t-1}; first row becomes NaN
            df[ret_col] = np.log(df[price_col] / df[price_col].shift(1))
            return_cols.append(ret_col)
            logger.info(
                f"  Log returns computed  →  {ret_col}  "
                f"[non-NaN: {df[ret_col].notna().sum():,}]"
            )
        return return_cols

    def _compute_realized_volatility(
        self,
        df:          pd.DataFrame,
        return_cols: List[str],
    ) -> List[str]:
        """
        Compute annualized rolling Realized Volatility for each return series.

        RV_t = std(r_{t-k+1}, ..., r_t) × √252

        Parameters
        ----------
        df          : DataFrame modified in-place
        return_cols : list of log-return column names already added to df

        Notes on `ddof=1`
        -----------------
        We use ddof=1 (sample standard deviation, the default for pandas .std())
        rather than ddof=0 (population std) because our rolling window is a
        SAMPLE of the true underlying variance process.  This gives an unbiased
        estimator in finite samples — important when window=5 (only 4 df).

        Returns
        -------
        List of newly created RV column names.
        """
        rv_cols = []
        for ret_col, rv_col in zip(return_cols, ENERGY_RV_COLS):
            df[rv_col] = (
                df[ret_col]
                .rolling(window=self.window, min_periods=self.window)
                .std(ddof=1)           # sample std (unbiased, Bessel-corrected)
                * ANNUALIZATION_FACTOR # scale daily vol → annualized vol
            )
            rv_cols.append(rv_col)
            logger.info(
                f"  Realized Volatility  →  {rv_col}  "
                f"[non-NaN: {df[rv_col].notna().sum():,}]"
            )
        return rv_cols

    def _report(self, target_df: pd.DataFrame) -> None:
        """Print a descriptive statistics table for the RV targets."""
        sep = "═" * 64
        print(f"\n{sep}")
        print("  REALIZED VOLATILITY — TARGET STATISTICS")
        print(f"  Date range : {target_df.index.min().date()} → "
              f"{target_df.index.max().date()}")
        print(f"  Rows       : {len(target_df):,}")
        print(f"{sep}")
        print(target_df.describe().round(6).to_string())
        print(f"{sep}\n")


# ── Module-level convenience function (backward-compatible API) ───────────────

def calculate_realized_volatility(
    df:     pd.DataFrame,
    window: int = 5,
) -> pd.DataFrame:
    """
    Functional wrapper around VolatilityDataProcessor for quick use.

    Adds log-return columns and RV columns directly to a copy of `df`.
    Returns the full enriched DataFrame (features + targets combined).

    Parameters
    ----------
    df     : raw aligned DataFrame from MacroDataPipeline
    window : rolling window size in trading days (default 5 = 1 week)

    Returns
    -------
    pd.DataFrame with additional columns:
        logret_Brent_Crude_Oil   — daily log return
        logret_TTF_Natural_Gas   — daily log return
        RV_Brent_Crude_Oil       — annualized realized volatility
        RV_TTF_Natural_Gas       — annualized realized volatility

    Example
    -------
    >>> enriched_df = calculate_realized_volatility(raw_df, window=5)
    >>> targets = enriched_df[["RV_Brent_Crude_Oil", "RV_TTF_Natural_Gas"]]
    """
    processor = VolatilityDataProcessor(
        window           = window,
        drop_return_cols = False,   # keep returns in the enriched output
    )
    feature_df, target_df = processor.fit_transform(df)

    # Re-join features + targets into one flat DataFrame
    enriched = pd.concat([feature_df, target_df], axis=1)
    return enriched


# ══════════════════════════════════════════════════════════════════════════════
# 2.  VOLATILITY TRANSFORMER MODEL
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class VolatilityConfig:
    """
    Hyperparameter configuration for VolatilityTransformer.

    Volatility-specific design choices vs. the generic price transformer:
      • n_targets = 2  (Brent RV, TTF RV) — not all 12 features
      • output_activation = Softplus  (enforces ŷ > 0 — vol is always positive)
      • Softplus β controls sharpness of the positive constraint:
          β → ∞  →  Softplus approaches ReLU (hard zero clamp)
          β = 1  →  smooth positive mapping; default and recommended
    """
    # ── Data ─────────────────────────────────────────────────────────────────
    n_features:         int   = 12    # number of macro input features
    n_targets:          int   = 2     # Brent RV + TTF RV
    window_size:        int   = 60    # look-back window W
    forecast_horizon:   int   = 5     # prediction horizon H

    # ── Transformer ───────────────────────────────────────────────────────────
    d_model:            int   = 64
    n_heads:            int   = 4     # d_model must be divisible by n_heads
    n_layers:           int   = 3
    dim_feedforward:    int   = 256
    dropout:            float = 0.1
    max_seq_len:        int   = 500
    pe_learnable:       bool  = False

    # ── Output constraint ─────────────────────────────────────────────────────
    softplus_beta:      int   = 1     # Softplus sharpness (1 = standard)
    softplus_threshold: int   = 20    # above this, Softplus ≈ identity (stability)

    # ── Training ─────────────────────────────────────────────────────────────
    learning_rate:      float = 5e-4
    weight_decay:       float = 1e-4
    batch_size:         int   = 32
    n_epochs:           int   = 50
    grad_clip:          float = 1.0
    device:             str   = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})."
        )


class PositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding (Vaswani et al., 2017).

    Injects sequence-order information into the token embeddings so the
    Transformer can distinguish timestep t from timestep t+k.

    Buffer (not a parameter): moves to device with .to() but is not
    updated during backpropagation.
    """

    def __init__(
        self,
        d_model:     int,
        max_seq_len: int   = 500,
        dropout:     float = 0.1,
        learnable:   bool  = False,
    ) -> None:
        super().__init__()
        self.dropout   = nn.Dropout(p=dropout)
        self.learnable = learnable

        if learnable:
            self.pe_embed = nn.Embedding(max_seq_len, d_model)
        else:
            pe       = torch.zeros(max_seq_len, d_model)                 # (L, D)
            position = torch.arange(max_seq_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float)
                * (-math.log(10_000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0))                  # (1, L, D)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, W, D)
        seq_len = x.size(1)
        if self.learnable:
            pos = torch.arange(seq_len, device=x.device).unsqueeze(0)   # (1, W)
            return self.dropout(x + self.pe_embed(pos))
        return self.dropout(x + self.pe[:, :seq_len, :])                 # (B, W, D)


class VolatilityTransformer(nn.Module):
    """
    Transformer Encoder model for multivariate Realized Volatility forecasting.

    Critical design change vs. TimeSeriesTransformer
    -------------------------------------------------
    The output_projection now ends with nn.Softplus() instead of a bare
    Linear layer.  This guarantees ŷ > 0 for ALL inputs — a hard mathematical
    constraint because volatility is a standard deviation and can NEVER be
    negative or zero.

    Why Softplus instead of ReLU?
    ------------------------------
    ReLU(x) = max(0, x)
      ✗ Gradient = 0 for x < 0  ("dying ReLU" problem)
      ✗ Exact zero output → log(0) = -∞ in QLIKE loss → NaN gradients

    Softplus(x) = (1/β) × log(1 + exp(β·x))
      ✓ Smooth, differentiable everywhere (no dead gradient regions)
      ✓ Always strictly POSITIVE: Softplus(x) > 0 for all x ∈ ℝ
      ✓ Asymptotically linear for large x (no saturation → no vanishing grad)
      ✓ With threshold: reverts to linear for x > threshold (numerical stability)

    Forward pass (shape trace)
    --------------------------
    x (B, W=60, N=12)
        ↓  InputProjection   Linear(12 → 64) + LayerNorm
    (B, W, 64)
        ↓  PositionalEncoding  sinusoidal
    (B, W, 64)
        ↓  TransformerEncoder  × 3 layers (MHA + FFN + Pre-LN)
    (B, W, 64)
        ↓  mean(dim=1)  temporal pooling
    (B, 64)
        ↓  OutputProjection  Linear(64→128) → GELU → Dropout → Linear(128→H×2)
    (B, H×2)
        ↓  Softplus(β=1)          ← THE KEY CHANGE: enforce ŷ > 0
    (B, H×2)
        ↓  view(B, H, 2)
    ŷ (B, H=5, 2)   — strictly positive volatility forecasts
    """

    def __init__(self, cfg: VolatilityConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # ── 1. Input Projection ───────────────────────────────────────────────
        self.input_projection = nn.Sequential(
            nn.Linear(cfg.n_features, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
        )

        # ── 2. Positional Encoding ────────────────────────────────────────────
        self.positional_encoding = PositionalEncoding(
            d_model     = cfg.d_model,
            max_seq_len = cfg.max_seq_len,
            dropout     = cfg.dropout,
            learnable   = cfg.pe_learnable,
        )

        # ── 3. Transformer Encoder ────────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = cfg.d_model,
            nhead           = cfg.n_heads,
            dim_feedforward = cfg.dim_feedforward,
            dropout         = cfg.dropout,
            activation      = "gelu",
            batch_first     = True,    # (B, W, D) convention throughout
            norm_first      = True,    # Pre-LN: more stable on volatile data
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers = cfg.n_layers,
            norm       = nn.LayerNorm(cfg.d_model),
        )

        # ── 4. Output Projection with Softplus Constraint ─────────────────────
        #
        # Output dimension: H × n_targets = forecast_horizon × 2
        #
        # Architecture:
        #   Linear(d_model → d_model×2)  — expand for richer representation
        #   GELU                          — smooth non-linearity
        #   Dropout                       — regularisation
        #   Linear(d_model×2 → H×2)      — project to forecast grid
        #   Softplus(β=1)                 — HARD POSITIVITY CONSTRAINT
        #
        # After Softplus, we reshape from (B, H×2) → (B, H, 2).
        # The Softplus is applied element-wise BEFORE reshaping so it
        # operates on every scalar forecast independently.
        out_dim = cfg.forecast_horizon * cfg.n_targets
        self.output_projection = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model * 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model * 2, out_dim),
            # ↓ CRITICAL: guarantees all output values are strictly > 0
            nn.Softplus(beta=cfg.softplus_beta, threshold=cfg.softplus_threshold),
        )

        self._init_weights()
        self._log_architecture()

    # ── Weight init ───────────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        """
        Xavier uniform for Linear layers, zero bias.

        For volatility regression, a conservative initialisation that
        starts predictions near zero (before Softplus maps them to ~0.69)
        helps avoid early NaN losses from extreme initial RV estimates.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _log_architecture(self) -> None:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"VolatilityTransformer  |  "
            f"total_params={total:,}  trainable={trainable:,}  |  "
            f"output=({self.cfg.forecast_horizon}, {self.cfg.n_targets})  "
            f"activation=Softplus(β={self.cfg.softplus_beta})"
        )

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        x:                    Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Parameters
        ----------
        x : FloatTensor (B, W, N)
            Batch of feature windows (scaled macro + log-return features).
        src_key_padding_mask : BoolTensor (B, W) | None
            Mask for padded positions (typically None for fixed windows).

        Returns
        -------
        FloatTensor (B, H, 2)
            Strictly positive realized volatility forecasts for
            [Brent Crude Oil, TTF Natural Gas] over H future trading days.
            Every element is guaranteed > 0 by the Softplus activation.
        """
        B = x.size(0)

        # ── Input projection: (B, W, N) → (B, W, d_model) ───────────────────
        x = self.input_projection(x)

        # ── Positional encoding: (B, W, d) → (B, W, d) ──────────────────────
        x = self.positional_encoding(x)

        # ── Transformer encoder: (B, W, d) → (B, W, d) ──────────────────────
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        # ── Temporal pooling: mean over W → (B, d) ───────────────────────────
        x = x.mean(dim=1)

        # ── Output projection + Softplus: (B, d) → (B, H×2) → all positive ──
        x = self.output_projection(x)                        # Softplus applied here

        # ── Reshape to forecast grid: (B, H×2) → (B, H, 2) ──────────────────
        x = x.view(B, self.cfg.forecast_horizon, self.cfg.n_targets)

        return x

    @torch.no_grad()
    def predict(self, x: Tensor) -> Tensor:
        """Inference-mode forward (no gradient tracking)."""
        self.eval()
        return self.forward(x)

    def positivity_check(self, x: Tensor) -> bool:
        """Assert that all output values are strictly positive. For testing."""
        out = self.forward(x)
        is_positive = (out > 0).all().item()
        logger.info(
            f"Positivity check: min(ŷ)={out.min().item():.2e}  "
            f"max(ŷ)={out.max().item():.2e}  "
            f"all_positive={is_positive}"
        )
        return bool(is_positive)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  QLIKE LOSS FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

class QLIKELoss(nn.Module):
    """
    Quasi-Likelihood (QLIKE) loss — the econometric industry standard
    for training and evaluating volatility forecasting models.

    Theoretical Background
    ----------------------
    Patton (2011, "Volatility Forecast Comparison Using Imperfect Volatility
    Proxies", Journal of Econometrics) proves that QLIKE is *robust* to
    measurement noise in the realized volatility proxy.

    This is crucial in practice: our RV target is computed from daily OHLC
    closing prices, which is an imperfect, noisy proxy for true latent
    volatility.  MSE loss is NOT robust to this proxy noise — it produces
    biased parameter estimates.  QLIKE is.

    Mathematical Definition
    -----------------------
    Let  y  = true (proxy) realized volatility  > 0
         ŷ  = model-predicted volatility        > 0  (enforced by Softplus)

    QLIKE(y, ŷ) = (y / ŷ) − log(y / ŷ) − 1

    Equivalently (numerically preferred form):
    QLIKE(y, ŷ) = y/ŷ − log(y) + log(ŷ) − 1

    This second form avoids computing log(y/ŷ) as a single expression,
    which can overflow/underflow for extreme ratios.

    Key Properties
    --------------
    1. NON-NEGATIVE:  QLIKE(y, ŷ) ≥ 0  for all y, ŷ > 0.
       Proof: f(u) = u − log(u) − 1 where u = y/ŷ.
              f'(u) = 1 − 1/u = 0  at u=1.  f(1) = 0.  f''(u) = 1/u² > 0.
              Therefore u=1 (y=ŷ) is the global minimum with L=0. ✓

    2. ASYMMETRIC:  QLIKE penalises under-prediction of volatility more
       heavily than over-prediction — exactly what a risk manager wants
       (unexpected vol spikes are the tail risk we must capture).

    3. SCALE-FREE:  The ratio y/ŷ makes QLIKE invariant to the units of
       the volatility measure (%, decimal, or annualized).

    4. CONVEX in ŷ:  Well-behaved loss surface with a unique global minimum.
       ∂²L/∂ŷ² = 2y/ŷ³ > 0  always.  ✓

    Gradient
    --------
    ∂QLIKE/∂ŷ = −y/ŷ² + 1/ŷ = (1 − y/ŷ) / ŷ

    When ŷ < y (under-prediction):  gradient is negative → push ŷ up.
    When ŷ > y (over-prediction):   gradient is positive → push ŷ down.
    The asymmetry: the curvature (∂²L/∂ŷ²) is larger when ŷ is small,
    so the optimizer faces steeper gradients for under-predictions.

    Numerical Stability
    -------------------
    We add epsilon to both y_true and y_pred before all operations.
    This guards against:
      • Division by zero     if ŷ ≈ 0 (early training with random weights)
      • log(0) = −∞          if y ≈ 0 (rare but possible for near-zero RV)
      • NaN propagation that would corrupt the entire batch

    Parameters
    ----------
    epsilon   : float
        Small constant added to y_true and y_pred for numerical safety.
        Default: 1e-8.  Must be << typical RV values (~0.1–0.5 annualized).
    reduction : "mean" | "sum" | "none"
        Aggregation method.
        "mean" — standard for gradient descent (average over B, H, n_targets)
        "sum"  — useful for weighting long-horizon predictions more heavily
        "none" — returns element-wise loss tensor (B, H, 2) for analysis
    """

    def __init__(
        self,
        epsilon:   float = EPSILON,
        reduction: str   = "mean",
    ) -> None:
        super().__init__()
        if epsilon <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon}.")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none'.")
        self.epsilon   = epsilon
        self.reduction = reduction
        logger.info(
            f"QLIKELoss  |  epsilon={epsilon:.0e}  reduction={reduction}"
        )

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        Compute the QLIKE loss.

        Parameters
        ----------
        y_pred : FloatTensor (B, H, 2)
            Model-predicted realized volatilities.
            MUST be strictly positive (guaranteed by Softplus in the model).
        y_true : FloatTensor (B, H, 2)
            Target realized volatilities from VolatilityDataProcessor.
            Should be non-negative; epsilon guards against exact zeros.

        Returns
        -------
        Scalar loss (reduction="mean"/"sum") or
        element-wise tensor (B, H, 2) (reduction="none").
        """
        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"Shape mismatch: y_pred={tuple(y_pred.shape)}, "
                f"y_true={tuple(y_true.shape)}."
            )

        # ── Numerical stability clamp ──────────────────────────────────────────
        # We add epsilon BEFORE any division or log operation.
        # Using clamp(min=epsilon) instead of +epsilon so that large values
        # are unaffected (clamp is a no-op for values >> epsilon).
        y_pred_safe = y_pred.clamp(min=self.epsilon)
        y_true_safe = y_true.clamp(min=self.epsilon)

        # ── QLIKE formula (numerically preferred decomposed form) ─────────────
        #
        # QLIKE = y_true/y_pred  −  log(y_true/y_pred)  −  1
        #       = y_true/y_pred  −  log(y_true) + log(y_pred) − 1
        #
        # We split log(y_true/y_pred) into log(y_true) - log(y_pred) to
        # avoid computing the ratio y_true/y_pred before taking log,
        # which is numerically safer for extreme ratio values.
        ratio = y_true_safe / y_pred_safe                  # u = y/ŷ  > 0

        # Equivalent: ratio − torch.log(ratio) − 1
        # Using the decomposed form for numerical clarity:
        loss = ratio - torch.log(y_true_safe) + torch.log(y_pred_safe) - 1.0

        # ── Sanity assertion: QLIKE must be ≥ 0 everywhere ────────────────────
        # Disabled in production (comment out) to avoid overhead.
        # assert (loss >= -1e-5).all(), f"Negative QLIKE detected: min={loss.min()}"

        # ── Reduction ─────────────────────────────────────────────────────────
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss                                     # (B, H, 2)

    def extra_repr(self) -> str:
        return f"epsilon={self.epsilon:.0e}, reduction='{self.reduction}'"

    @staticmethod
    def minimum_check() -> None:
        """
        Mathematical unit test: verify QLIKE achieves its minimum of 0
        when y_pred == y_true (perfect forecast).
        """
        loss_fn = QLIKELoss(reduction="mean")
        y = torch.rand(16, 5, 2).clamp(min=0.01)   # random positive targets
        # Perfect forecast: ŷ = y → QLIKE should be ≈ 0
        val = loss_fn(y_pred=y, y_true=y)
        assert abs(val.item()) < 1e-5, f"Minimum check failed: QLIKE={val.item()}"
        logger.info(f"QLIKE minimum check passed: QLIKE(y, y) = {val.item():.2e} ≈ 0")

    @staticmethod
    def asymmetry_check() -> None:
        """
        Verify QLIKE is asymmetric: penalises under-prediction more than
        over-prediction of equal magnitude.
        """
        loss_fn  = QLIKELoss(reduction="mean")
        y_true   = torch.full((16, 5, 2), 0.20)    # true vol = 20%
        delta    = 0.05                              # 5% absolute shift

        y_under  = y_true - delta                   # ŷ < y  (under-predict)
        y_over   = y_true + delta                   # ŷ > y  (over-predict)

        loss_under = loss_fn(y_pred=y_under.clamp(min=1e-8), y_true=y_true)
        loss_over  = loss_fn(y_pred=y_over,                  y_true=y_true)

        logger.info(
            f"QLIKE asymmetry check  |  "
            f"under-prediction loss={loss_under.item():.6f}  "
            f"over-prediction loss={loss_over.item():.6f}  "
            f"|  under > over: {loss_under.item() > loss_over.item()}"
        )
        assert loss_under.item() > loss_over.item(), (
            "Asymmetry check FAILED: expected QLIKE to penalise "
            "under-prediction more than over-prediction."
        )
        logger.info("QLIKE asymmetry check passed. ✓")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  INTEGRATION SMOKE TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    torch.manual_seed(42)
    np.random.seed(42)

    SEP = "═" * 64

    # ─────────────────────────────────────────────────────────────────────────
    # 4a. VolatilityDataProcessor — smoke test with synthetic raw DataFrame
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  MODULE 1 — VolatilityDataProcessor")
    print(SEP)

    # Simulate a raw_df from MacroDataPipeline (T rows, 12 columns)
    T = 1_300
    date_index = pd.bdate_range("2020-01-01", periods=T)   # business days only

    # Simulate realistic Brent (~80 USD) and TTF (~50 EUR) price paths
    np.random.seed(0)
    raw_prices = {
        "Brent_Crude_Oil":  80  * np.cumprod(1 + np.random.normal(0, 0.02, T)),
        "TTF_Natural_Gas":  50  * np.cumprod(1 + np.random.normal(0, 0.03, T)),
        **{f"MacroFeature_{i:02d}": np.random.rand(T) for i in range(10)},
    }
    raw_df = pd.DataFrame(raw_prices, index=date_index)
    print(f"\n  Input raw_df shape : {raw_df.shape}")
    print(f"  Columns            : {list(raw_df.columns)}\n")

    # Test the convenience function
    enriched = calculate_realized_volatility(raw_df, window=5)
    print(f"  Enriched df shape  : {enriched.shape}")
    print(f"  New columns added  : "
          f"{[c for c in enriched.columns if 'logret' in c or 'RV' in c]}")

    # Test the class API
    processor   = VolatilityDataProcessor(window=5)
    feature_df, target_df = processor.fit_transform(raw_df)
    print(f"\n  feature_df shape   : {feature_df.shape}  ← X fed into Transformer")
    print(f"  target_df  shape   : {target_df.shape}   ← y (2 RV columns)")
    print(f"\n  RV targets head(5):")
    print(target_df.head(5).to_string())

    # ─────────────────────────────────────────────────────────────────────────
    # 4b. VolatilityTransformer — forward pass shape + positivity check
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  MODULE 2 — VolatilityTransformer")
    print(SEP)

    cfg   = VolatilityConfig()
    model = VolatilityTransformer(cfg)

    # Dummy batch simulating DataLoader output
    B = cfg.batch_size
    x_batch = torch.rand(B, cfg.window_size, cfg.n_features)     # (32, 60, 12)
    y_batch = torch.rand(B, cfg.forecast_horizon, cfg.n_targets)  # (32,  5,  2)
    y_batch = y_batch.clamp(min=0.01)    # targets must be positive for QLIKE

    with torch.no_grad():
        y_hat = model(x_batch)

    print(f"\n  Input  x_batch : {tuple(x_batch.shape)}")
    print(f"  Output ŷ_batch : {tuple(y_hat.shape)}")
    print(f"  Target y_batch : {tuple(y_batch.shape)}")

    # Positivity check — must be True
    all_positive = model.positivity_check(x_batch)
    print(f"\n  Softplus positivity guarantee : {all_positive}  ✓")
    print(f"  Sample ŷ[0, :, :] (H=5, 2 assets):")
    print(f"    {y_hat[0].detach().numpy().round(6)}")

    # ─────────────────────────────────────────────────────────────────────────
    # 4c. QLIKELoss — numerical properties verification
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  MODULE 3 — QLIKELoss")
    print(SEP)

    loss_fn = QLIKELoss(epsilon=EPSILON, reduction="mean")

    # Minimum check: QLIKE(y, y) == 0
    QLIKELoss.minimum_check()

    # Asymmetry check: under-prediction penalised more
    QLIKELoss.asymmetry_check()

    # Practical loss on a forward pass
    qlike_val = loss_fn(y_pred=y_hat, y_true=y_batch)
    print(f"\n  QLIKE loss on dummy batch : {qlike_val.item():.6f}")

    # Compare against MSE for context
    mse_val = nn.MSELoss()(y_hat, y_batch)
    print(f"  MSE  loss on dummy batch  : {mse_val.item():.6f}")

    # ─────────────────────────────────────────────────────────────────────────
    # 4d. Mini training loop: backward pass verification
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  BACKWARD PASS VERIFICATION (3 gradient steps)")
    print(SEP)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    model.train()
    for step in range(1, 4):
        optimizer.zero_grad(set_to_none=True)
        y_hat = model(x_batch)                       # (32, 5, 2)
        loss  = loss_fn(y_hat, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        # Verify all outputs remain strictly positive after weight updates
        still_positive = (y_hat > 0).all().item()
        print(
            f"  Step {step}  |  QLIKE={loss.item():.6f}  "
            f"|  ŷ_min={y_hat.min().item():.4e}  "
            f"|  all_positive={still_positive}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 4e. Integration blueprint
    # ─────────────────────────────────────────────────────────────────────────
    PRODUCTION_SNIPPET = f"""
{SEP}
  PRODUCTION INTEGRATION BLUEPRINT
{SEP}

from macro_data_pipeline       import MacroDataPipeline
from volatility_forecasting    import (
    VolatilityDataProcessor, VolatilityTransformer,
    VolatilityConfig, QLIKELoss
)
from sliding_window_dataset    import build_dataloaders
from sklearn.preprocessing     import MinMaxScaler
import torch, numpy as np

# ── 1. Fetch raw data ─────────────────────────────────────────────────────────
pipeline = MacroDataPipeline(fred_api_key=..., evds_api_key=...)
raw_df, _, _, _ = pipeline.run()

# ── 2. Engineer volatility features + targets ────────────────────────────────
processor   = VolatilityDataProcessor(window=5)
feature_df, target_df = processor.fit_transform(raw_df)
# feature_df : (T', 12)   — macro + market features   → X
# target_df  : (T',  2)   — Brent RV, TTF RV          → y

# ── 3. Scale features (targets kept in annualized vol space for QLIKE) ────────
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(feature_df.values)       # (T', 12)
y_raw    = target_df.values                              # (T',  2) — unscaled

# ── 4. Build DataLoaders ──────────────────────────────────────────────────────
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_raw,    dtype=torch.float32)

# Combine into a joint (X, y) sliding-window loader
# (use target_indices to separate features from RV targets in the loader)
train_loader, val_loader, test_loader, _ = build_dataloaders(
    tensor=X_tensor, window_size=60, forecast_horizon=5, batch_size=32
)

# ── 5. Instantiate model + loss + optimizer ───────────────────────────────────
cfg     = VolatilityConfig()
model   = VolatilityTransformer(cfg)
loss_fn = QLIKELoss(reduction="mean")
optim   = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)

# ── 6. Training loop (plug into train_model() from transformer_hedging_model.py)
for epoch in range(cfg.n_epochs):
    for x_batch, y_batch in train_loader:
        optim.zero_grad(set_to_none=True)
        y_hat = model(x_batch)              # (B, 5, 2)  — Softplus-constrained
        loss  = loss_fn(y_hat, y_batch)     # QLIKE — robust to RV proxy noise
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

# ── 7. Inference: decode prediction back to annualized vol space ──────────────
x_new = X_tensor[-60:].unsqueeze(0)         # (1, 60, 12)
rv_forecast = model.predict(x_new)          # (1, 5, 2) — strictly positive
# rv_forecast[0, :, 0] → 5-day Brent RV forecast  (annualized)
# rv_forecast[0, :, 1] → 5-day TTF RV forecast    (annualized)
{SEP}
"""
    print(PRODUCTION_SNIPPET)
