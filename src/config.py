"""
Configuration module for the Bitcoin Price Prediction System.
Defines feature sets, model hyper-parameters, file paths, visual themes, and external data sources.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DATASET_DIR = BASE_DIR / "dataset"
APP_DIR = BASE_DIR / "app"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset File Paths
RAW_HISTORICAL_BTC_CSV = DATASET_DIR / "dataset_upto_20august.csv"
DXY_CSV = DATASET_DIR / "DXY_clean.csv"
VIX_CSV = DATASET_DIR / "VIX.csv"
SP500_CSV = DATASET_DIR / "SP500.csv"
GOLD_CSV = DATASET_DIR / "Gold.csv"

# Model & Scaler Artifact Paths
MODEL_PATH = MODELS_DIR / "lstm_model.keras"
SCALER_X_PATH = MODELS_DIR / "scaler_X.pkl"
SCALER_Y_PATH = MODELS_DIR / "scaler_y.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

# Database Path
DB_PATH = DATA_DIR / "prediction_history.db"

# Model Specifications
MODEL_VERSION = "Standalone-LSTM-v1.0"
LOOKBACK_WINDOW = 30  # 30 previous days lookback
TARGET_COLUMN = "Close"

# 17 Engineered Features in EXACT order used during model training
MODEL_FEATURES = [
    'Open', 'High', 'Low', 'Volume',
    'RSI', 'MACD', 'MACD_Signal',
    'BB_Middle', 'BB_Upper', 'BB_Lower', 'BB_Width',
    'Daily_Return', 'ROC',
    'Close_Lag_7', 'SMA_7', 'Price_SMA7_Ratio',
    'SP500'
]

# External Macro Tickers (Yahoo Finance)
YFINANCE_TICKERS = {
    "SP500": "^GSPC",
    "DXY": "DX-Y.NYB",
    "GOLD": "GC=F",
    "VIX": "^VIX",
    "BTC_FALLBACK": "BTC-USD"
}

# Binance API
BINANCE_KLINES_URL = "https://data-api.binance.vision/api/v3/klines"
BINANCE_SYMBOL = "BTCUSDT"

# UI Theme Color Palette (Dark Financial Terminal)
THEME_COLORS = {
    "bitcoin_orange": "#F7931A",
    "bitcoin_orange_glow": "rgba(247, 147, 26, 0.25)",
    "profit_green": "#00C853",
    "profit_green_bg": "rgba(0, 200, 83, 0.12)",
    "loss_red": "#FF3B30",
    "loss_red_bg": "rgba(255, 59, 48, 0.12)",
    "ui_blue": "#0E9CFC",
    "ui_teal": "#34F2D2",
    "dark_bg": "#0B0E14",
    "card_bg": "#151A24",
    "card_border": "#212836",
    "text_primary": "#F0F6FC",
    "text_secondary": "#8B949E",
    "grid_color": "#1F2937"
}

# Benchmark Evaluation Metrics Reported for Final Deployed Model
BENCHMARK_METRICS = {
    "MAE": 5537.18,
    "RMSE": 5919.28,
    "R2": 0.8889,
    "ALT_MAPE": 4.35,
    "ALT_RMSE": 5211.90,
    "ALT_R2": 0.9138
}
