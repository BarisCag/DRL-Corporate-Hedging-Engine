"""
================================================================================
  MAIN EXECUTION ENGINE (V5 - Final Institutional Edition)
  Project: Transformer-based Realized Volatility Forecasting
  Author: Baris Cagri Malci
================================================================================
"""

import sys
import os
sys.path.append('/content/')
sys.path.append('/content/sample_data/')

import torch
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from macro_data_pipeline import MacroDataPipeline
from volatility_forecasting import (
    VolatilityDataProcessor, VolatilityTransformer, 
    VolatilityConfig, QLIKELoss
)
from sliding_window_dataset import build_dataloaders

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MainEngine")

def main():
    logger.info("Starting End-to-End Volatility Forecasting Pipeline...")

    # NOTE: Insert your API keys here.
    FRED_API_KEY = "YOUR_FRED_API_KEY"
    EVDS_API_KEY = "YOUR_EVDS_API_KEY"

    logger.info("--- STEP 1: Fetching Global Macro Data ---")
    pipeline = MacroDataPipeline(
        fred_api_key=FRED_API_KEY, 
        evds_api_key=EVDS_API_KEY, 
        start_date="2020-01-01"
    )
    raw_df, _, _, _ = pipeline.run()

    logger.info("--- STEP 2: Computing Realized Volatility Targets ---")
    processor = VolatilityDataProcessor(window=5, drop_return_cols=True)
    feature_df, target_df = processor.fit_transform(raw_df)

    # --- CRITICAL FIX: Do not drop columns, preserve dimensions ---
    logger.info("--- STEP 2.5: Sanitization (Maintaining Strict Feature Dimensions) ---")
    # If API failures create gaps, we DO NOT drop the column.
    # Forward fill existing values, then fill the rest with 0.0 to maintain tensor shape.
    feature_df = feature_df.ffill().bfill().fillna(0.0)
    
    feature_df = feature_df.pct_change().fillna(0.0)
    feature_df = feature_df.replace([np.inf, -np.inf], 0.0)

    # --- AUTO-REGRESSION ---
    logger.info("--- STEP 2.7: Adding Autoregressive Features (Lagged RV) ---")
    feature_df = pd.concat([feature_df, target_df], axis=1)
    # --------------------------------------------------------

    logger.info("--- STEP 3: Scaling Features ---")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(feature_df.values)
    y_raw = target_df.values

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_raw, dtype=torch.float32)

    logger.info("--- STEP 4: Building PyTorch DataLoaders ---")
    combined_tensor = torch.cat([X_tensor, y_tensor], dim=1)
    
    n_features = X_tensor.shape[1] 
    target_indices = [n_features, n_features + 1]

    train_loader, val_loader, test_loader, _ = build_dataloaders(
        tensor=combined_tensor,
        window_size=60,
        forecast_horizon=5,
        target_indices=target_indices,
        batch_size=32
    )

    logger.info(f"--- STEP 5: Initializing Transformer (Features={n_features}) ---")
    cfg = VolatilityConfig(
        n_features=n_features,
        n_targets=2,
        window_size=60,
        forecast_horizon=5,
        n_epochs=150,          
        learning_rate=1e-4       
    )
    
    model = VolatilityTransformer(cfg).to(cfg.device)
    loss_fn = QLIKELoss(reduction="mean")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)

    logger.info("--- STEP 6: Starting High-Precision Training Loop ---")
    
    best_val_loss = float('inf')
    for epoch in range(1, cfg.n_epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x_batch = batch[0][:, :, :n_features].to(cfg.device) 
            y_batch = batch[1].to(cfg.device)

            optimizer.zero_grad(set_to_none=True)
            y_hat = model(x_batch)
            loss = loss_fn(y_hat, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x_batch = batch[0][:, :, :n_features].to(cfg.device)
                y_batch = batch[1].to(cfg.device)
                y_hat = model(x_batch)
                loss = loss_fn(y_hat, y_batch)
                val_loss += loss.item()
                
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        
        if not np.isnan(avg_val) and avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), "best_volatility_model.pt")

        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"Epoch {epoch:03d}/{cfg.n_epochs} | Train QLIKE: {avg_train:.4f} | Val QLIKE: {avg_val:.4f}")

    logger.info(f"--- TRAINING COMPLETE! Best Val QLIKE: {best_val_loss:.4f} ---")

    logger.info("--- STEP 7: Generating Out-of-Sample Forecast Plot ---")
    
    model.load_state_dict(torch.load("best_volatility_model.pt", map_location=cfg.device, weights_only=True))
    model.eval()

    true_rv = []
    pred_rv = []

    with torch.no_grad():
        for batch in test_loader:
            x_batch = batch[0][:, :, :n_features].to(cfg.device)
            y_batch = batch[1]
            
            y_hat = model(x_batch).cpu().numpy()
            y_true = y_batch.numpy()
            
            pred_rv.extend(y_hat[:, 0, 0]) 
            true_rv.extend(y_true[:, 0, 0])

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(14, 6))

    plot_len = min(150, len(true_rv))

    plt.plot(true_rv[-plot_len:], label='Realized Volatility (True RV)', color='#1f77b4', linewidth=2)
    plt.plot(pred_rv[-plot_len:], label='Transformer Forecast (Predicted RV)', color='#ff7f0e', linewidth=2.5, linestyle='--')

    plt.title('Deep Transformer: Energy Market Volatility Forecast (Test Set - Brent Crude)', fontsize=14, fontweight='bold')
    plt.ylabel('Annualized Volatility', fontsize=12)
    plt.xlabel(f'Trading Days (Last {plot_len} Days)', fontsize=12)
    plt.legend(fontsize=12, loc='upper right')
    plt.fill_between(range(plot_len), true_rv[-plot_len:], pred_rv[-plot_len:], color='gray', alpha=0.1) 
    plt.tight_layout()

    plt.savefig('volatility_forecast_brent.png', dpi=300, bbox_inches='tight')
    logger.info("Final plot saved as 'volatility_forecast_brent.png'!")

if __name__ == "__main__":
    main()