"""
Prediction and Inference Engine for Standalone LSTM.
Loads trained model artifacts, generates next-day closing price forecasts,
calculates expected directional movements, and supports What-If scenario simulations.
"""

import sys
import pickle
import json
from pathlib import Path
from datetime import timedelta
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import (
    MODEL_PATH,
    SCALER_X_PATH,
    SCALER_Y_PATH,
    METADATA_PATH,
    MODEL_VERSION,
    LOOKBACK_WINDOW,
    MODEL_FEATURES,
    TARGET_COLUMN
)
from src.feature_engineering import prepare_inference_sequence, calculate_technical_indicators

# Global in-memory cache for artifacts
_MODEL_CACHE = None
_SCALER_X_CACHE = None
_SCALER_Y_CACHE = None
_METADATA_CACHE = None


def load_model_artifacts():
    """Loads and caches model, scalers, and metadata."""
    global _MODEL_CACHE, _SCALER_X_CACHE, _SCALER_Y_CACHE, _METADATA_CACHE

    if _MODEL_CACHE is not None:
        return _MODEL_CACHE, _SCALER_X_CACHE, _SCALER_Y_CACHE, _METADATA_CACHE

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}. Run src/train_export.py first.")

    _MODEL_CACHE = load_model(str(MODEL_PATH))

    with open(SCALER_X_PATH, 'rb') as f:
        _SCALER_X_CACHE = pickle.load(f)

    with open(SCALER_Y_PATH, 'rb') as f:
        _SCALER_Y_CACHE = pickle.load(f)

    if METADATA_PATH.exists():
        with open(METADATA_PATH, 'r') as f:
            _METADATA_CACHE = json.load(f)
    else:
        _METADATA_CACHE = {}

    return _MODEL_CACHE, _SCALER_X_CACHE, _SCALER_Y_CACHE, _METADATA_CACHE


def predict_next_day(df_features: pd.DataFrame) -> dict:
    """
    Executes next-day price forecasting on the latest 30-day sequence.
    Returns prediction payload with price, expected change, and directional momentum.
    """
    model, scaler_X, scaler_y, metadata = load_model_artifacts()

    if len(df_features) < LOOKBACK_WINDOW:
        raise ValueError(f"Feature dataset has {len(df_features)} rows; minimum {LOOKBACK_WINDOW} required.")

    # Prepare latest 30-day scaled sequence
    sequence = prepare_inference_sequence(df_features, scaler_X, lookback=LOOKBACK_WINDOW)

    # Predict scaled price
    pred_scaled = model.predict(sequence, verbose=0)

    # Inverse transform
    pred_price = float(scaler_y.inverse_transform(pred_scaled)[0][0])

    latest_row = df_features.iloc[-1]
    latest_date = pd.to_datetime(latest_row['Date'])
    target_date = (latest_date + timedelta(days=1)).strftime('%Y-%m-%d')
    current_price = float(latest_row['Close'])

    expected_change_usd = pred_price - current_price
    expected_change_pct = (expected_change_usd / current_price) * 100.0 if current_price > 0 else 0.0
    direction = "UP" if pred_price >= current_price else "DOWN"

    return {
        "status": "SUCCESS",
        "model_version": MODEL_VERSION,
        "latest_date": latest_date.strftime('%Y-%m-%d'),
        "target_date": target_date,
        "current_price": current_price,
        "predicted_price": pred_price,
        "expected_change_usd": expected_change_usd,
        "expected_change_pct": expected_change_pct,
        "direction": direction,
        "metadata": metadata
    }


def predict_what_if(df_features: pd.DataFrame, adjustments: dict) -> dict:
    """
    Generates a hypothetical prediction based on user-modified market conditions.
    adjustments: {
        'btc_price_change_pct': float (e.g. +3.0 or -5.0),
        'sp500_change_pct': float (e.g. +1.5 or -2.0),
        'volume_multiplier': float (e.g. 1.5),
        'rsi_override': float or None (e.g. 75.0)
    }
    """
    model, scaler_X, scaler_y, _ = load_model_artifacts()

    df_mod = df_features.copy()
    latest_idx = df_mod.index[-1]

    # Apply adjustments to latest row
    curr_close = df_mod.loc[latest_idx, 'Close']
    curr_sp = df_mod.loc[latest_idx, 'SP500']
    curr_vol = df_mod.loc[latest_idx, 'Volume']

    btc_pct = adjustments.get('btc_price_change_pct', 0.0)
    sp_pct = adjustments.get('sp500_change_pct', 0.0)
    vol_mult = adjustments.get('volume_multiplier', 1.0)
    rsi_val = adjustments.get('rsi_override', None)

    new_close = curr_close * (1 + btc_pct / 100.0)
    new_sp = curr_sp * (1 + sp_pct / 100.0)
    new_vol = curr_vol * vol_mult

    df_mod.loc[latest_idx, 'Close'] = new_close
    df_mod.loc[latest_idx, 'High'] = max(df_mod.loc[latest_idx, 'High'], new_close)
    df_mod.loc[latest_idx, 'Low'] = min(df_mod.loc[latest_idx, 'Low'], new_close)
    df_mod.loc[latest_idx, 'SP500'] = new_sp
    df_mod.loc[latest_idx, 'Volume'] = new_vol

    # Recalculate dependent technical features on latest 30 rows
    sma7 = df_mod['Close'].tail(7).mean()
    df_mod.loc[latest_idx, 'SMA_7'] = sma7
    df_mod.loc[latest_idx, 'Price_SMA7_Ratio'] = new_close / sma7 if sma7 > 0 else 1.0
    df_mod.loc[latest_idx, 'Daily_Return'] = btc_pct / 100.0

    if rsi_val is not None:
        df_mod.loc[latest_idx, 'RSI'] = rsi_val

    # Create sequence and predict
    sequence = prepare_inference_sequence(df_mod, scaler_X, lookback=LOOKBACK_WINDOW)
    pred_scaled = model.predict(sequence, verbose=0)
    pred_price = float(scaler_y.inverse_transform(pred_scaled)[0][0])

    expected_change_usd = pred_price - new_close
    expected_change_pct = (expected_change_usd / new_close) * 100.0 if new_close > 0 else 0.0
    direction = "UP" if pred_price >= new_close else "DOWN"

    return {
        "status": "SIMULATION",
        "model_version": MODEL_VERSION,
        "base_price": curr_close,
        "hypothetical_current_price": new_close,
        "predicted_price": pred_price,
        "expected_change_usd": expected_change_usd,
        "expected_change_pct": expected_change_pct,
        "direction": direction,
        "adjustments": adjustments
    }
