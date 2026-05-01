"""
================================================================================
  SlidingWindowDataset & DataLoader Infrastructure
  For: PyTorch-based Deep Reinforcement Learning — Corporate Hedging & Risk Mgmt
  Feeds: Scaled tensor of shape (T, N_features) from macro_data_pipeline.py
  Author: Expert PyTorch Developer
  Version: 1.0.0
================================================================================

PIPELINE POSITION
-----------------
  macro_data_pipeline.py  →  [tensor: (T, 12)]
         ↓
  sliding_window_dataset.py  →  DataLoader  →  DRL Environment / Transformer

HOW IT WORKS — SLIDING WINDOW CONCEPT
--------------------------------------
  Given a continuous time-series of T timesteps and N features:

  Full tensor  : [d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, ...]  shape (T, N)

  Window 0 (i=0):
    x (state)   → [d0, d1, d2, d3, d4]     shape (window_size=5, N)
    y (target)  → [d5, d6]                  shape (forecast_horizon=2, N)

  Window 1 (i=1):
    x  →  [d1, d2, d3, d4, d5]
    y  →  [d6, d7]

  Window k (i=k):
    x  →  tensor[k : k + window_size]
    y  →  tensor[k + window_size : k + window_size + forecast_horizon]

  Total number of valid windows = T - window_size - forecast_horizon + 1
"""

# ── Standard Library ──────────────────────────────────────────────────────────
import logging
from typing import Optional, Tuple

# ── Third-Party ───────────────────────────────────────────────────────────────
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("SlidingWindowDataset")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  CORE DATASET CLASS
# ══════════════════════════════════════════════════════════════════════════════

class SlidingWindowDataset(Dataset):
    """
    A PyTorch Dataset that produces (x, y) sliding-window pairs from a
    continuous multivariate time-series tensor.

    Parameters
    ----------
    tensor : torch.Tensor, shape (T, N_features)
        The full scaled time-series produced by MacroDataPipeline.
        T = number of trading days, N = number of macro/market features.

    window_size : int
        Number of past timesteps the model observes as its "state".
        Example: 60  →  the model sees the last 60 trading days (~3 months).

    forecast_horizon : int
        Number of future timesteps the model must predict / act on.
        Example: 5  →  the model forecasts / hedges over the next 5 trading days.

    target_indices : list[int] | None
        If provided, `y` will contain only these feature columns.
        Example: [0, 1, 2]  →  predict only the first 3 features.
        Default: None  →  return all N features in `y`.

    Returns (per __getitem__)
    -------------------------
    x : torch.FloatTensor, shape (window_size, N_features)
        Historical observation window — the "state" fed into the model/env.

    y : torch.FloatTensor, shape (forecast_horizon, N_targets)
        Future target window — what the model needs to predict or plan for.
        N_targets = len(target_indices) if specified, else N_features.

    Raises
    ------
    ValueError
        If the tensor is too short for the requested window + horizon.
    """

    def __init__(
        self,
        tensor:           Tensor,
        window_size:      int,
        forecast_horizon: int,
        target_indices:   Optional[list] = None,
    ) -> None:
        # ── Input validation ──────────────────────────────────────────────
        if tensor.ndim != 2:
            raise ValueError(
                f"Expected a 2-D tensor (T, N), got shape {tuple(tensor.shape)}."
            )

        T, N = tensor.shape
        min_length = window_size + forecast_horizon

        if T < min_length:
            raise ValueError(
                f"Tensor has {T} timesteps, but window_size ({window_size}) + "
                f"forecast_horizon ({forecast_horizon}) = {min_length}. "
                "Reduce window/horizon or provide more data."
            )

        if target_indices is not None:
            bad = [i for i in target_indices if not (0 <= i < N)]
            if bad:
                raise ValueError(
                    f"target_indices {bad} are out of range for a tensor with "
                    f"{N} features (valid: 0–{N - 1})."
                )

        # ── Store attributes ──────────────────────────────────────────────
        self.tensor           = tensor.float()         # ensure float32
        self.T                = T
        self.N                = N
        self.window_size      = window_size
        self.forecast_horizon = forecast_horizon
        self.target_indices   = target_indices         # None → use all columns

        # Pre-compute number of valid windows (computed once, reused in __len__)
        self.n_samples: int = T - window_size - forecast_horizon + 1

        logger.info(
            f"SlidingWindowDataset ready  |  "
            f"T={T}, N_features={N}, "
            f"window_size={window_size}, forecast_horizon={forecast_horizon}  |  "
            f"n_samples={self.n_samples:,}"
        )

    # ── Required Dataset methods ──────────────────────────────────────────────

    def __len__(self) -> int:
        """Total number of (x, y) windows in the dataset."""
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """
        Fetch the i-th sliding window pair.

        Parameters
        ----------
        idx : int
            Window index in [0, n_samples).

        Returns
        -------
        x : FloatTensor (window_size, N_features)
        y : FloatTensor (forecast_horizon, N_features | N_targets)
        """
        if idx < 0 or idx >= self.n_samples:
            raise IndexError(
                f"Index {idx} out of range for dataset with {self.n_samples} samples."
            )

        # ── Slice the window ──────────────────────────────────────────────
        x_start = idx
        x_end   = idx + self.window_size           # exclusive
        y_start = x_end
        y_end   = y_start + self.forecast_horizon  # exclusive

        x: Tensor = self.tensor[x_start : x_end]                     # (W, N)
        y: Tensor = self.tensor[y_start : y_end]                     # (H, N)

        # ── Optionally slice target columns ──────────────────────────────
        if self.target_indices is not None:
            y = y[:, self.target_indices]                             # (H, N_t)

        return x, y

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def x_shape(self) -> Tuple[int, int]:
        """Shape of a single x sample: (window_size, N_features)."""
        return (self.window_size, self.N)

    @property
    def y_shape(self) -> Tuple[int, int]:
        """Shape of a single y sample: (forecast_horizon, N_targets)."""
        n_targets = (
            len(self.target_indices) if self.target_indices is not None
            else self.N
        )
        return (self.forecast_horizon, n_targets)

    def __repr__(self) -> str:
        return (
            f"SlidingWindowDataset("
            f"T={self.T}, N={self.N}, "
            f"window_size={self.window_size}, "
            f"forecast_horizon={self.forecast_horizon}, "
            f"n_samples={self.n_samples})"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 2.  DATALOADER FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def build_dataloaders(
    tensor:             Tensor,
    window_size:        int   = 60,
    forecast_horizon:   int   = 5,
    target_indices:     Optional[list] = None,
    batch_size:         int   = 32,
    train_ratio:        float = 0.70,
    val_ratio:          float = 0.15,
    # test_ratio is implicitly 1 - train_ratio - val_ratio
    num_workers:        int   = 0,    # set >0 for GPU training (e.g., 4)
    pin_memory:         bool  = False,  # set True when using GPU
    seed:               int   = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, SlidingWindowDataset]:
    """
    Build train / validation / test DataLoaders from a continuous time-series.

    Split strategy
    --------------
    ┌─────────────────────────────────────────────────────────┐
    │  IMPORTANT: Time-series data must NOT be shuffled       │
    │  randomly — that would cause data leakage (future data  │
    │  leaking into the training set).                        │
    │                                                         │
    │  We use a strict chronological split:                   │
    │    Train  → first 70% of windows                        │
    │    Val    → next  15%                                   │
    │    Test   → last  15%                                   │
    └─────────────────────────────────────────────────────────┘

    Parameters
    ----------
    tensor           : FloatTensor (T, N) — scaled macro tensor
    window_size      : int — observation window (default 60 trading days)
    forecast_horizon : int — prediction horizon (default 5 trading days)
    target_indices   : list[int] | None — columns to return in y
    batch_size       : int — samples per gradient-update step
    train_ratio      : float — fraction of windows for training
    val_ratio        : float — fraction of windows for validation
    num_workers      : int — parallel data-loading workers (0 = main thread)
    pin_memory       : bool — faster CPU→GPU transfer; enable for CUDA training
    seed             : int — reproducibility seed

    Returns
    -------
    train_loader : DataLoader
    val_loader   : DataLoader
    test_loader  : DataLoader
    full_dataset : SlidingWindowDataset (useful for DRL environment wrappers)
    """
    if not (0 < train_ratio < 1 and 0 < val_ratio < 1):
        raise ValueError("train_ratio and val_ratio must be strictly between 0 and 1.")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError(
            f"train_ratio ({train_ratio}) + val_ratio ({val_ratio}) must be < 1.0 "
            "to leave room for a test set."
        )

    # ── Build the full dataset ────────────────────────────────────────────
    full_dataset = SlidingWindowDataset(
        tensor=tensor,
        window_size=window_size,
        forecast_horizon=forecast_horizon,
        target_indices=target_indices,
    )
    n = len(full_dataset)

    # ── Chronological split (no shuffling) ───────────────────────────────
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    n_test  = n - n_train - n_val          # absorbs rounding remainder

    # torch.utils.data.Subset preserves original indices (chronological order)
    from torch.utils.data import Subset   # noqa: PLC0415
    train_ds = Subset(full_dataset, range(0,               n_train))
    val_ds   = Subset(full_dataset, range(n_train,         n_train + n_val))
    test_ds  = Subset(full_dataset, range(n_train + n_val, n))

    logger.info(
        f"Dataset split  |  "
        f"total={n:,}  train={n_train:,}  val={n_val:,}  test={n_test:,}"
    )

    # ── Shared DataLoader kwargs ──────────────────────────────────────────
    _loader_kwargs = dict(
        batch_size  = batch_size,
        num_workers = num_workers,
        pin_memory  = pin_memory,
    )

    # ── Instantiate loaders ───────────────────────────────────────────────
    # shuffle=False is CRITICAL for all splits in time-series tasks.
    train_loader = DataLoader(train_ds, shuffle=False, **_loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **_loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **_loader_kwargs)

    # ── Log batch shape for quick verification ────────────────────────────
    _x_batch, _y_batch = next(iter(train_loader))
    logger.info(
        f"Batch shapes  |  "
        f"x_batch={tuple(_x_batch.shape)}  "   # (B, window_size, N)
        f"y_batch={tuple(_y_batch.shape)}"      # (B, horizon, N_targets)
    )

    return train_loader, val_loader, test_loader, full_dataset


# ══════════════════════════════════════════════════════════════════════════════
# 3.  DATASET INSPECTOR UTILITY
# ══════════════════════════════════════════════════════════════════════════════

def inspect_dataset(dataset: SlidingWindowDataset, n_samples: int = 3) -> None:
    """
    Print shape, dtype, value-range, and sample windows for quick QA.

    Parameters
    ----------
    dataset  : SlidingWindowDataset
    n_samples: int — number of random windows to display
    """
    sep = "═" * 62
    print(f"\n{sep}")
    print("  DATASET INSPECTION REPORT")
    print(sep)
    print(f"  {repr(dataset)}")
    print(f"  x shape per sample : {dataset.x_shape}")
    print(f"  y shape per sample : {dataset.y_shape}")
    print(f"  Tensor dtype       : {dataset.tensor.dtype}")
    print(f"  Tensor device      : {dataset.tensor.device}")
    print(f"  Value range (full) : [{dataset.tensor.min():.4f}, "
          f"{dataset.tensor.max():.4f}]")
    print(f"\n  Sample windows (first / middle / last):")
    print(f"{'─'*62}")

    indices = [0, len(dataset) // 2, len(dataset) - 1]
    for i in indices:
        x, y = dataset[i]
        print(f"  Window idx={i:>6,}  |  x={tuple(x.shape)}  y={tuple(y.shape)}  "
              f"|  x[0]=[{x[0, :3].tolist()!s:.50s}...]"
              f"  y[0]=[{y[0, :3].tolist()!s:.50s}...]")

    print(sep + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  COMPLETE USAGE EXAMPLE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── 4a. Simulate the tensor output from MacroDataPipeline ─────────────
    #    In production, replace this block with:
    #        raw_df, scaled_df, tensor, scaler = pipeline.run()
    # ──────────────────────────────────────────────────────────────────────
    torch.manual_seed(42)
    T           = 1_300    # ~5 years of daily trading data (252 days/yr × 5)
    N_FEATURES  = 12       # matches your macro pipeline output
    WINDOW      = 60       # 60-day look-back  (~3 months)
    HORIZON     = 5        # 5-day forecast    (~1 trading week)
    BATCH_SIZE  = 32

    # Simulated MinMax-scaled tensor: values ∈ [0, 1]
    tensor = torch.rand(T, N_FEATURES)

    print(f"\n{'═'*62}")
    print(f"  INPUT TENSOR  shape={tuple(tensor.shape)}  dtype={tensor.dtype}")
    print(f"{'═'*62}")

    # ─────────────────────────────────────────────────────────────────────
    # 4b. Instantiate the Dataset directly
    # ─────────────────────────────────────────────────────────────────────
    dataset = SlidingWindowDataset(
        tensor           = tensor,
        window_size      = WINDOW,
        forecast_horizon = HORIZON,
        # target_indices = [0, 3, 7]  ← uncomment to predict only 3 columns
    )

    # ── Inspect a single sample ───────────────────────────────────────────
    x0, y0 = dataset[0]
    print(f"\n  Single sample[0]:")
    print(f"    x (state / history)  : shape={tuple(x0.shape)}  "
          f"dtype={x0.dtype}")
    print(f"    y (target / forecast): shape={tuple(y0.shape)}  "
          f"dtype={y0.dtype}")

    # ─────────────────────────────────────────────────────────────────────
    # 4c. Build Train / Validation / Test DataLoaders
    # ─────────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, full_dataset = build_dataloaders(
        tensor           = tensor,
        window_size      = WINDOW,
        forecast_horizon = HORIZON,
        batch_size       = BATCH_SIZE,
        train_ratio      = 0.70,         # 70% train
        val_ratio        = 0.15,         # 15% val  → 15% test implicit
        num_workers      = 0,            # 0 for CPU; set 4+ for GPU training
        pin_memory       = False,        # True if using .cuda()
    )

    # ─────────────────────────────────────────────────────────────────────
    # 4d. Inspect the dataset
    # ─────────────────────────────────────────────────────────────────────
    inspect_dataset(full_dataset)

    # ─────────────────────────────────────────────────────────────────────
    # 4e. Iterate one batch from each split — verify shapes
    # ─────────────────────────────────────────────────────────────────────
    print(f"{'═'*62}")
    print("  BATCH SHAPE VERIFICATION")
    print(f"{'═'*62}")
    for split_name, loader in [("Train", train_loader),
                                ("Val  ", val_loader),
                                ("Test ", test_loader)]:
        x_b, y_b = next(iter(loader))
        print(
            f"  {split_name}  |  "
            f"batches={len(loader):>4,}  |  "
            f"x_batch={tuple(x_b.shape)}  "   # (32, 60, 12)
            f"y_batch={tuple(y_b.shape)}"      # (32,  5, 12)
        )
    print(f"{'═'*62}\n")

    # ─────────────────────────────────────────────────────────────────────
    # 4f. Minimal training loop skeleton (plug in your model here)
    # ─────────────────────────────────────────────────────────────────────
    print("  TRAINING LOOP SKELETON (plug in your Transformer / DRL agent)")
    print(f"{'─'*62}")

    # Example stub — replace with your actual model
    class StubModel(torch.nn.Module):
        """
        Placeholder model.  Replace with:
          - Temporal Fusion Transformer
          - Seq2Seq LSTM Encoder-Decoder
          - PPO / SAC policy network
        """
        def __init__(self, n_features: int, window: int, horizon: int):
            super().__init__()
            self.fc = torch.nn.Linear(window * n_features, horizon * n_features)
            self.n_features = n_features
            self.horizon    = horizon

        def forward(self, x: Tensor) -> Tensor:
            # x: (B, W, N)  →  flatten  →  linear  →  (B, H, N)
            B = x.size(0)
            return self.fc(x.view(B, -1)).view(B, self.horizon, self.n_features)

    model     = StubModel(N_FEATURES, WINDOW, HORIZON)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    N_EPOCHS = 2   # keep small for demonstration
    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            # x_batch : (B, window_size=60, N=12)
            # y_batch : (B, horizon=5,      N=12)
            optimizer.zero_grad()
            y_pred = model(x_batch)           # forward pass
            loss   = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch}/{N_EPOCHS}  —  avg_train_loss = {avg_loss:.6f}")

    print(f"\n{'═'*62}")
    print("  Dataset & DataLoader infrastructure verified successfully.")
    print(f"{'═'*62}\n")

    # ─────────────────────────────────────────────────────────────────────
    # 4g. Integration with MacroDataPipeline (production snippet)
    # ─────────────────────────────────────────────────────────────────────
    PRODUCTION_SNIPPET = """
# ── In production, replace the simulated tensor above with: ──────────────────

from macro_data_pipeline import MacroDataPipeline
import os

pipeline = MacroDataPipeline(
    fred_api_key = os.getenv("FRED_API_KEY"),
    evds_api_key = os.getenv("EVDS_API_KEY"),
    start_date   = "2020-01-01",
)
raw_df, scaled_df, tensor, scaler = pipeline.run()
# tensor.shape → (T, 12)  ← pass directly into build_dataloaders()

train_loader, val_loader, test_loader, full_dataset = build_dataloaders(
    tensor           = tensor,
    window_size      = 60,
    forecast_horizon = 5,
    batch_size       = 32,
)
# ── Then train your model exactly as shown in the loop above. ────────────────
    """
    print(PRODUCTION_SNIPPET)
