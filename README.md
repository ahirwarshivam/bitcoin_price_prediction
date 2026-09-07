# ₿ Bitcoin Price Prediction & AI Forecasting Terminal

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16.1-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3.15.1-D00000?logo=keras&logoColor=white)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://python.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An end-to-end, production-grade Bitcoin next-day closing price forecasting dashboard and automated machine learning pipeline. Developed through 17 rigorous iterative ablation experiments, the platform utilizes a Deep Standalone Long Short-Term Memory (LSTM) recurrent neural network taking 30-day sequence lookbacks across 17 engineered technical and macroeconomic features.

---

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [Data Sources & Market Feeds](#-data-sources--market-feeds)
- [Feature Engineering (17 Input Features)](#-feature-engineering-17-input-features)
- [Exploratory Data Analysis (EDA) Insights](#-exploratory-data-analysis-eda-insights)
- [Experimentation Journey & Model Ablation](#-experimentation-journey--model-ablation)
- [Model Comparison: Standalone LSTM vs. LSTM + XGBoost](#-model-comparison-standalone-lstm-vs-lstm--xgboost)
- [Final Model Architecture & Performance Metrics](#-final-model-architecture--performance-metrics)
- [System Architecture](#-system-architecture)
- [Streamlit Financial Terminal Features](#-streamlit-financial-terminal-features)
- [Automated Data Synchronization & Prediction Pipeline](#-automated-data-synchronization--prediction-pipeline)
- [Persistent SQLite Prediction History](#-persistent-sqlite-prediction-history)
- [Local Installation & Setup](#-local-installation--setup)
- [Streamlit Cloud Deployment Guide](#-streamlit-cloud-deployment-guide)
- [Limitations & Risk Factors](#-limitations--risk-factors)
- [Disclaimer](#-disclaimer)
- [Author & Acknowledgments](#-author--acknowledgments)

---

## 🔭 Project Overview

Cryptocurrency markets operate 24/7/365 with extreme volatility, nonlinear price swings, and intricate dependencies on global macroeconomic liquidity. This project transforms historical cryptocurrency price analysis into a live, interactive, and self-updating financial intelligence platform.

When a user visits the dashboard:
1. **Automated Live Ingestion**: Retrieves the newest Bitcoin daily candles (via Binance API with Yahoo Finance fallback) and global macroeconomic indicators (S&P 500, DXY, Gold, VIX).
2. **Deterministic Feature Pipeline**: Applies exact technical indicator calculations (RSI, MACD, Bollinger Bands, ROC, SMA ratios, price lags) and forward-fills non-trading weekend/holiday macro gaps.
3. **Next-Day Price Prediction**: Feeds the standardized $30 \times 17$ sequence into the trained Standalone LSTM model to predict the next-day Bitcoin closing price.
4. **Persistent History Tracking**: Logs predictions to a local SQLite database, auto-resolves actual prices once market days conclude, and computes absolute error and error percentages.
5. **Interactive Plotly Visualizations**: Displays 30-day actual vs. predicted performance, error distribution charts, and interactive What-If scenario simulations.

---

## 🎯 Problem Statement

Traditional autoregressive models (e.g. ARIMA) assume linear relationships and often struggle with crypto market regimes. Deep learning models like LSTMs effectively capture sequential dependencies over multi-day lookback windows. The core objective of this project is to build an accurate, non-overfitting next-day Bitcoin forecasting system that incorporates both price action dynamics and macroeconomic sentiment.

---

## 🌐 Data Sources & Market Feeds

| Asset / Indicator | Ticker / Source | Role in System | Frequency |
| :--- | :--- | :--- | :--- |
| **Bitcoin (BTCUSDT)** | Binance API (`BTCUSDT`) / Yahoo Finance (`BTC-USD`) | Primary Target Asset (`Open, High, Low, Close, Volume`) | Daily (24/7) |
| **S&P 500 Index** | Yahoo Finance (`^GSPC`) | Equity Market Sentiment & Macro Liquidity ($r = 0.90$) | Daily (Trading days) |
| **US Dollar Index (DXY)** | Yahoo Finance (`DX-Y.NYB`) | Fiat Currency Strength & Global Liquidity Benchmark | Daily (Trading days) |
| **Gold Futures** | Yahoo Finance (`GC=F`) | Store-of-Value Asset & Inflation Hedge ($r = 0.79$) | Daily (Trading days) |
| **Volatility Index (VIX)**| Yahoo Finance (`^VIX`) | Market Fear / Implied Volatility Benchmark | Daily (Trading days) |

> **Weekend & Market Holiday Handling:** Crypto trades 24/7 while traditional markets close on weekends and public holidays. The pipeline automatically forward-fills (`ffill()`) external indicators to preserve temporal alignment without losing continuous crypto observations.

---

## ⚙️ Feature Engineering (17 Input Features)

The final LSTM receives a 3D input tensor of shape `(Batch_Size, 30, 17)`. The 17 features are arranged in the exact order below:

```
[1]  Open             — Daily Opening Price
[2]  High             — Daily High Price
[3]  Low              — Daily Low Price
[4]  Volume           — 24h Trading Volume (Base Asset)
[5]  RSI              — Relative Strength Index (14-period)
[6]  MACD             — Moving Average Convergence Divergence Line
[7]  MACD_Signal      — MACD 9-period Exponential Signal Line
[8]  BB_Middle        — Bollinger Bands 20-period Moving Average
[9]  BB_Upper         — Bollinger Bands Upper Band (2 std dev)
[10] BB_Lower         — Bollinger Bands Lower Band (2 std dev)
[11] BB_Width         — Bollinger Bandwidth ((Upper - Lower) / Middle)
[12] Daily_Return     — Percentage change in daily closing price
[13] ROC              — Rate of Change (12-period Momentum Indicator)
[14] Close_Lag_7      — Closing price lagged by 7 calendar days
[15] SMA_7            — 7-day Simple Moving Average
[16] Price_SMA7_Ratio — Ratio of Current Close to 7-Day SMA
[17] SP500            — S&P 500 Index Closing Value
```

---

## 📊 Exploratory Data Analysis (EDA) Insights

- **Strong Macro Correlation**: S&P 500 exhibits a strong positive correlation ($r = 0.90$) with Bitcoin closing prices over extended horizons.
- **Gold Co-Movement**: Bitcoin and Gold share a price level correlation of $0.79$, though their daily returns have a weak correlation ($0.09$), showing distinct short-term dynamics.
- **Volume Surges**: Major volume spikes align with explosive price moves and market capitulation events.
- **Non-Gaussian Distribution**: Bitcoin prices exhibit high positive skewness and fat-tailed volatility.

---

## 🧪 Experimentation Journey & Model Ablation

The project underwent 17 iterative ablation experiments to arrive at the optimal feature combination:

- **Initial Experiments (01–04)**: Base feature sets with 12–13 indicators. Removing the close price decreased performance; adding Average True Range (ATR) slightly inflated errors.
- **Momentum & Trend Tuning (05–09)**: Introducing Rate of Change (ROC), 7-day price lag (`Close_Lag_7`), and 7-day SMA (`SMA_7`) improved $R^2$ into positive territory ($+0.27$).
- **Ratio Engineering (10–16)**: Adding `Price_SMA7_Ratio` yielded substantial error reductions, bringing MAPE down to $1.58\%$ on validation slices.
- **Macro Feature Integration (Final 17)**: Integrating S&P 500 macroeconomic closing values provided stability during broad market rallies.

> **Full Experiment History:** The complete experiment logs, parameter iterations, and loss curves are documented in [`results/Experiments and Results.pdf`](file:///Users/shivamahirwar/Desktop/bitcoin_price_pred/results/Experiments%20and%20Results.pdf).

---

## ⚖️ Model Comparison: Standalone LSTM vs. LSTM + XGBoost

A hybrid architecture was tested by extracting the 64-dimensional latent representation from the LSTM's dense layer and training an XGBoost Regressor on top of the learned embeddings:

| Model Architecture | MAPE | RMSE | $R^2$ Score | Production Decision |
| :--- | :---: | :---: | :---: | :---: |
| **Standalone Deep LSTM (Final)** | **4.35%** (3.20% test) | **$5,537.18** ($3,812.59) | **0.8889** (0.9539) | ✅ **Selected for Deployment** |
| **LSTM + XGBoost Hybrid** | 7.19% | $10,174.23 | 0.6718 | ❌ Rejected (High Variance) |

**Conclusion:** The Standalone LSTM demonstrated superior generalization on the out-of-sample test set. The hybrid XGBoost layer introduced tree-based discretization errors that degraded continuous temporal predictions.

---

## 🧠 Final Model Architecture & Performance Metrics

### Neural Network Topology
```
Input Layer (Shape: 30 days × 17 features)
   │
   ▼
LSTM Layer (128 units, activation='relu', return_sequences=False)
   │
   ▼
Dense Layer (64 units, activation='relu')
   │
   ▼
Dropout Layer (rate=0.20)
   │
   ▼
Dense Output Layer (1 unit, linear activation)
```

- **Optimizer**: Adam ($\text{learning rate} = 0.0005$)
- **Loss Function**: Mean Squared Error (MSE)
- **Batch Size**: 64
- **Epochs**: 200
- **Validation Split**: 20% (Chronological non-shuffled)
- **Random Seed**: 42

### Reported Benchmark Metrics
```
MAE  : $5,537.18
RMSE : $5,919.28
R²   : 0.8889
MAPE : 4.35% (Alternative Evaluation Run: RMSE $5,211.90, R² 0.9138)
```

---

## 🏗️ System Architecture

```
├── .github/
│   └── workflows/
│       └── update_prediction.yml       # Scheduled GitHub Actions daily runner
├── .streamlit/
│   └── config.toml                     # Dark financial terminal theme settings
├── app/
│   └── streamlit_app.py                # Main interactive Streamlit application
├── data/
│   ├── prediction_history.db           # Persistent SQLite database (schema & records)
│   └── merged_market_data.csv          # Cached synchronized market data
├── dataset/
│   ├── dataset_upto_20august.csv       # Historical Bitcoin base (2018–2026)
│   ├── SP500.csv                       # Historical S&P 500 daily data
│   ├── DXY_clean.csv                   # Historical US Dollar Index
│   ├── Gold.csv                        # Historical Gold futures
│   └── VIX.csv                         # Historical Volatility Index
├── models/
│   ├── lstm_model.keras                # Trained Keras Standalone LSTM model
│   ├── scaler_X.pkl                    # Feature MinMaxScaler (fitted on train split)
│   ├── scaler_y.pkl                    # Target MinMaxScaler (fitted on train split)
│   └── model_metadata.json             # Architecture & evaluation metadata
├── Notebooks/
│   ├── main_model_notebook.ipynb       # Final model training & evaluation notebook
│   ├── data_preparation.ipynb          # Data merging & date alignment
│   ├── exp 01 base.ipynb               # Baseline LSTM exploration
│   ├── exp 02 DRX.ipynb                # Feature engineering experiments
│   └── finance data download.ipynb     # Initial market data downloads
├── results/
│   └── Experiments and Results.pdf     # Full 17-experiment research documentation
├── src/
│   ├── config.py                       # Paths, feature lists, color themes, constants
│   ├── data_fetch.py                   # Multi-source live ingestion & fallback
│   ├── feature_engineering.py          # 17 Technical features & sequence builder
│   ├── history.py                      # SQLite manager, KPI computations, auto-seeder
│   ├── prediction.py                   # Inference engine & What-If simulation
│   ├── train_export.py                 # Exact model training & artifact exporter
│   └── update_pipeline.py              # CLI & automated update runner
├── requirements.txt                    # Pinned Python package dependencies
└── README.md                           # Documentation & user guide
```

---

## 💻 Streamlit Financial Terminal Features

1. **Top KPI Bar**: Real-time cards displaying Current BTC Price, Next-Day Forecast, Expected % Movement, Model $R^2$ Benchmark, and 30-Day Mean Error.
2. **Hero Prediction Card**: Glowing Bitcoin Orange card showing current closing price, target forecast price, and dynamic green/red movement badges.
3. **Interactive Plotly Price Chart**: Multi-timeframe view (7d, 30d, 90d, 1y, All) with Bollinger Bands and volume subplots.
4. **30-Day Performance Chart**: Two-line interactive comparison of Actual vs. Predicted prices with full hover inspection (Date, Actual, Predicted, Absolute Error, Error %).
5. **30-Day Error Analysis**: Color-coded error percentage bar chart with 30-day mean baseline.
6. **Recent Predictions Table**: Sortable table of previous completed predictions with directional accuracy validation.
7. **Macroeconomic Monitor**: Real-time cards and co-movement chart for S&P 500, DXY, Gold, and VIX.
8. **What-If Scenario Simulation**: Interactive sliders to inject hypothetical market shocks and inspect model divergence in real-time.
9. **Research & Model Architecture**: Tabular overview of model benchmarks, 17 features breakdown, and link to experiment history.

---

## ⚡ Automated Data Synchronization & Prediction Pipeline

The standalone script `src/update_pipeline.py` can be run manually, locally via cron, or through GitHub Actions.

### Manual CLI Execution:
```bash
python src/update_pipeline.py
```

### GitHub Actions Cron:
The `.github/workflows/update_prediction.yml` action runs daily at `00:30 UTC`:
- Fetches new daily candles from Binance / Yahoo Finance
- Resolves actual closing prices for prior predictions
- Computes next-day prediction and appends to `data/prediction_history.db`
- Commits changes automatically back to the repository

---

## 🗄️ Persistent SQLite Prediction History

The database is stored at `data/prediction_history.db` under the `predictions` table:

```sql
CREATE TABLE predictions (
    date TEXT PRIMARY KEY,             -- Target prediction date (YYYY-MM-DD)
    prediction_date TEXT,              -- Generation date (YYYY-MM-DD)
    predicted_price REAL NOT NULL,     -- Forecasted closing price in USD
    actual_price REAL,                 -- Actual realized close (filled post-close)
    error REAL,                        -- actual - predicted
    error_pct REAL,                    -- abs(actual - predicted) / actual * 100
    prev_close REAL,                   -- Base closing price at inference time
    expected_change_pct REAL,          -- (predicted - prev_close) / prev_close * 100
    actual_change_pct REAL,            -- (actual - prev_close) / prev_close * 100
    direction_predicted TEXT,          -- 'UP' or 'DOWN'
    direction_actual TEXT,             -- 'UP' or 'DOWN'
    direction_correct INTEGER,         -- 1 if matched, 0 if diverged
    model_version TEXT NOT NULL,       -- 'Standalone-LSTM-v1.0'
    is_simulated INTEGER DEFAULT 0,    -- 0 for live, 1 for what-if simulations
    created_at TEXT NOT NULL           -- Timestamp (UTC)
);
```

---

## 🚀 Local Installation & Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### 1. Clone Repository
```bash
git clone https://github.com/shivamahirwar/bitcoin_price_pred.git
cd bitcoin_price_pred
```

### 2. Create Virtual Environment
```bash
python3 -m venv env
source env/bin/activate   # On Windows: env\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. (Optional) Re-Train & Export Model
```bash
python src/train_export.py
```

### 5. Launch the Streamlit Terminal
```bash
streamlit run app/streamlit_app.py
```
Open your browser at `http://localhost:8501`.

---

## ☁️ Streamlit Cloud Deployment Guide

1. Push your repository to GitHub.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **"New App"** and select your repository:
   - **Repository:** `<your-username>/bitcoin_price_pred`
   - **Branch:** `main`
   - **Main file path:** `app/streamlit_app.py`
4. Click **"Deploy"**. The app will install packages from `requirements.txt` and run automatically.

---

## ⚠️ Limitations & Risk Factors

- **Black Swan Events**: Macroeconomic crises, regulatory announcements, and crypto exchange shocks cannot be fully anticipated by historical time series.
- **Market Gaps**: Non-trading weekend gaps in stock markets require forward-filling macro indicators.
- **Model Bias**: Error distribution analysis shows a mild systematic underprediction bias during parabolic crypto bull runs.
- **Lookback Requirements**: Requires at least 30 consecutive clean historical trading days to generate valid sequence tensors.

---

## 📜 Disclaimer

> **⚠️ Financial Disclaimer:**
> This application provides experimental machine-learning forecasts based on historical market data. Bitcoin prices are highly volatile and affected by events that may not be captured by the model. Predictions are strictly for educational and research purposes, do not constitute financial advice, and should never be treated as guaranteed future prices.

---

## 👨‍💻 Author & Acknowledgments

**Developed by:** Shivam Ahirwar  
**LinkedIn:** [https://www.linkedin.com/in/shivam-ahirwar-425293311/](https://www.linkedin.com/in/shivam-ahirwar-425293311/)  
**Project:** Bitcoin Price Prediction System  
**Research Documentation:** [`results/Experiments and Results.pdf`](file:///Users/shivamahirwar/Desktop/bitcoin_price_pred/results/Experiments%20and%20Results.pdf)
