"""
Feature Engineering Module for Bitcoin Price Prediction.
Implements the exact technical indicators and feature pipeline used during model training.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import ta

# Ensure repository root is in pythonpath
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import MODEL_FEATURES, LOOKBACK_WINDOW, TARGET_COLUMN


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all technical indicators matching main_model_notebook.ipynb.
    Requires 'Close' column in the DataFrame.
    """
    df = df.copy()

    # Ensure numeric types
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # RSI (Relative Strength Index)
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()

    # MACD (Moving Average Convergence Divergence)
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_Middle'] = bb.bollinger_mavg()
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Lower'] = bb.bollinger_lband()
    df['BB_Width'] = bb.bollinger_wband()

    # Daily Return
    df['Daily_Return'] = df['Close'].pct_change()

    # Rate of Change (ROC - 12 days)
    df['ROC'] = ta.momentum.ROCIndicator(df['Close'], window=12).roc()

    # Lag & Moving Average Features
    df['Close_Lag_7'] = df['Close'].shift(7)
    df['SMA_7'] = df['Close'].rolling(window=7).mean()
    df['Price_SMA7_Ratio'] = df['Close'] / df['SMA_7']

    return df


def build_feature_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds the clean feature DataFrame containing exactly the 17 required model features.
    Drops any initial rows with NaN values resulting from rolling/lag calculations.
    """
    df_feat = calculate_technical_indicators(df)

    # Validate that SP500 exists
    if 'SP500' not in df_feat.columns:
        raise ValueError("Macro feature 'SP500' is missing from DataFrame.")

    # Select features + Target + Date
    cols_to_keep = MODEL_FEATURES + [TARGET_COLUMN]
    if 'Date' in df_feat.columns:
        cols_to_keep = ['Date'] + cols_to_keep

    # Remove rows with NaN values
    df_clean = df_feat[cols_to_keep].dropna().reset_index(drop=True)
    return df_clean


def prepare_inference_sequence(df_clean: pd.DataFrame, scaler_X, lookback: int = LOOKBACK_WINDOW) -> np.ndarray:
    """
    Extracts the latest `lookback` days of features, scales them using scaler_X,
    and shapes the tensor into (1, lookback, 17) for LSTM prediction.
    """
    if len(df_clean) < lookback:
        raise ValueError(f"Insufficient historical data: required {lookback} rows, got {len(df_clean)} rows.")

    latest_data = df_clean[MODEL_FEATURES].tail(lookback).copy()
    latest_scaled = scaler_X.transform(latest_data.values)
    sequence = latest_scaled.reshape(1, lookback, len(MODEL_FEATURES))
    return sequence


def create_sequences(data: np.ndarray, target: np.ndarray, n_steps: int = LOOKBACK_WINDOW):
    """Generates sequence batches for training or batch evaluation."""
    X_seq, y_seq = [], []
    for i in range(n_steps, len(data)):
        X_seq.append(data[i-n_steps:i])
        y_seq.append(target[i])
    return np.array(X_seq), np.array(y_seq)
