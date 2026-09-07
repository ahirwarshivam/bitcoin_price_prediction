"""
Model Training and Export Pipeline.
Reproduces the exact Standalone LSTM training pipeline from main_model_notebook.ipynb,
evaluates performance against reported metrics, and serializes the model, scalers, and metadata.
"""

import os
import sys
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

# Add repository root to pythonpath
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
import ta

from src.config import (
    RAW_HISTORICAL_BTC_CSV,
    DXY_CSV,
    VIX_CSV,
    SP500_CSV,
    GOLD_CSV,
    MODEL_PATH,
    SCALER_X_PATH,
    SCALER_Y_PATH,
    METADATA_PATH,
    MODEL_VERSION,
    LOOKBACK_WINDOW,
    MODEL_FEATURES,
    TARGET_COLUMN
)


def create_sequences(X, y, n_steps):
    """Create lookback sequences for LSTM input."""
    X_seq, y_seq = [], []
    for i in range(n_steps, len(X)):
        X_seq.append(X[i-n_steps:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)


def load_and_preprocess_training_data():
    """Load and merge historical Bitcoin data with macro indicators and calculate technical features."""
    print("Loading raw datasets...")
    maindf = pd.read_csv(RAW_HISTORICAL_BTC_CSV)
    df = maindf.drop(columns=['Ignore', 'Close time'], errors='ignore')

    dxy = pd.read_csv(DXY_CSV)
    vix = pd.read_csv(VIX_CSV)
    sp500 = pd.read_csv(SP500_CSV)
    gold = pd.read_csv(GOLD_CSV)

    # Clean + normalize dates
    df['Date'] = pd.to_datetime(df['Open time'], utc=True).dt.tz_localize(None).dt.normalize()
    for x in [dxy, vix, sp500, gold]:
        x['Date'] = pd.to_datetime(x['Date'], utc=True).dt.tz_localize(None).dt.normalize()

    dxy = dxy.drop(columns=['Unnamed: 0'], errors='ignore')

    # Merge external features
    df = df.merge(dxy, on='Date', how='left')
    df = df.merge(vix, on='Date', how='left')
    df = df.merge(sp500, on='Date', how='left')
    df = df.merge(gold, on='Date', how='left')

    # Fill missing market-day values
    ext = ['DXY', 'VIX', 'SP500', 'Gold']
    df[ext] = df[ext].ffill()
    df = df.dropna(subset=ext).reset_index(drop=True)

    # Calculate Technical Indicators
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    df['MACD'] = ta.trend.MACD(df['Close']).macd()
    df['MACD_Signal'] = ta.trend.MACD(df['Close']).macd_signal()

    bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_Middle'] = bb.bollinger_mavg()
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Lower'] = bb.bollinger_lband()
    df['BB_Width'] = bb.bollinger_wband()

    # Additional features
    df['Daily_Return'] = df['Close'].pct_change()
    df['ROC'] = ta.momentum.ROCIndicator(df['Close'], window=12).roc()
    df['Close_Lag_7'] = df['Close'].shift(7)
    df['SMA_7'] = df['Close'].rolling(window=7).mean()
    df['Price_SMA7_Ratio'] = df['Close'] / df['SMA_7']

    # Remove NA rows from rolling calculations
    df = df.dropna().reset_index(drop=True)
    print(f"Preprocessed dataset shape: {df.shape}, Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    return df


def train_and_export_model():
    """Train the Standalone LSTM model and export artifacts."""
    tf.random.set_seed(42)
    np.random.seed(42)

    df = load_and_preprocess_training_data()
    df_model = df[MODEL_FEATURES + [TARGET_COLUMN]].dropna().copy()

    data = df_model[MODEL_FEATURES].values
    target = df_model[[TARGET_COLUMN]].values

    n_steps_in = LOOKBACK_WINDOW
    split = int(len(data) * 0.8)

    # Scaling fit on training set only
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    scaler_X.fit(data[:split])
    scaler_y.fit(target[:split])

    data_scaled = scaler_X.transform(data)
    target_scaled = scaler_y.transform(target)

    # Create sequences
    X_train, y_train = create_sequences(
        data_scaled[:split],
        target_scaled[:split],
        n_steps_in
    )

    X_test, y_test = create_sequences(
        data_scaled[split - n_steps_in:],
        target_scaled[split - n_steps_in:],
        n_steps_in
    )

    print(f"Training sequences: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Testing sequences:  X_test={X_test.shape}, y_test={y_test.shape}")

    # Build Standalone LSTM Model Architecture
    inp = Input(shape=(n_steps_in, X_train.shape[2]))
    lstm = LSTM(128, activation='relu')(inp)
    dense = Dense(64, activation='relu')(lstm)
    drop = Dropout(0.2)(dense)
    out = Dense(1)(drop)

    lstm_model = Model(inp, out)
    lstm_model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='mse'
    )
    lstm_model.summary()

    # Train Model
    print("Training Standalone LSTM for 200 epochs...")
    history = lstm_model.fit(
        X_train,
        y_train,
        epochs=200,
        batch_size=64,
        validation_split=0.2,
        shuffle=False,
        verbose=1
    )

    # Evaluate on Test Set
    y_pred_scaled = lstm_model.predict(X_test, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_actual = scaler_y.inverse_transform(y_test).flatten()

    mae = float(mean_absolute_error(y_actual, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_actual, y_pred)))
    r2 = float(r2_score(y_actual, y_pred))
    mape = float(mean_absolute_percentage_error(y_actual, y_pred) * 100)

    print("\n==========================================")
    print("STANDALONE LSTM FINAL EVALUATION METRICS")
    print("==========================================")
    print(f"MAE  : ${mae:,.2f}")
    print(f"RMSE : ${rmse:,.2f}")
    print(f"R²   : {r2:.4f}")
    print(f"MAPE : {mape:.2f}%")
    print("==========================================")

    # Test latest prediction (August 21, 2026)
    latest_data = df[MODEL_FEATURES].dropna().tail(n_steps_in)
    latest_scaled = scaler_X.transform(latest_data)
    X_future = latest_scaled.reshape(1, n_steps_in, len(MODEL_FEATURES))
    pred_future_scaled = lstm_model.predict(X_future, verbose=0)
    pred_future_price = float(scaler_y.inverse_transform(pred_future_scaled)[0][0])
    print(f"Next-Day Predicted Price for Aug 21, 2026: ${pred_future_price:,.2f}")

    # Serialize Artifacts
    print(f"\nSaving model artifact to: {MODEL_PATH}")
    lstm_model.save(str(MODEL_PATH))

    print(f"Saving scaler_X to: {SCALER_X_PATH}")
    with open(SCALER_X_PATH, 'wb') as f:
        pickle.dump(scaler_X, f)

    print(f"Saving scaler_y to: {SCALER_Y_PATH}")
    with open(SCALER_Y_PATH, 'wb') as f:
        pickle.dump(scaler_y, f)

    # Save test set predictions & actuals mapping for history initialization
    test_dates = df['Date'].iloc[split:].dt.strftime('%Y-%m-%d').tolist()

    metadata = {
        "model_version": MODEL_VERSION,
        "model_type": "Standalone LSTM",
        "lookback_window": LOOKBACK_WINDOW,
        "features": MODEL_FEATURES,
        "n_features": len(MODEL_FEATURES),
        "target": TARGET_COLUMN,
        "train_split_ratio": 0.8,
        "evaluation_metrics": {
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "R2": round(r2, 4),
            "MAPE": round(mape, 2)
        },
        "reported_metrics": {
            "MAE": 5537.18,
            "RMSE": 5919.28,
            "R2": 0.8889,
            "ALT_MAPE": 4.35,
            "ALT_RMSE": 5211.90,
            "ALT_R2": 0.9138
        },
        "training_data_date_range": {
            "start": df['Date'].min().strftime('%Y-%m-%d'),
            "end": df['Date'].max().strftime('%Y-%m-%d')
        },
        "total_samples": len(df),
        "test_samples": len(y_test)
    }

    print(f"Saving metadata to: {METADATA_PATH}")
    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\nModel training and artifact serialization completed successfully!")
    return df, y_actual, y_pred, test_dates


if __name__ == '__main__':
    train_and_export_model()
