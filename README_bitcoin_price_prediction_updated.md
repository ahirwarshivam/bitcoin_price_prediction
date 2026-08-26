# Bitcoin Price Prediction Using LSTM

## Project Overview

This project focuses on predicting Bitcoin's future closing price using historical market data, technical indicators, and selected macro-financial variables.

The project was developed as an experimental machine-learning workflow rather than as a single-model implementation. Multiple feature combinations, look-back windows, technical indicators, and model configurations were tested before selecting the final model.

The complete experiment log and result analysis are available in:

`results/Experiments and Results.pdf`

---

## Project Effort and Experimental Approach

The main objective was to understand which features and modelling choices improve Bitcoin price prediction.

The work included:

- Historical Bitcoin price and volume analysis
- Exploratory Data Analysis (EDA)
- Daily return and price-distribution analysis
- Price-volume relationship analysis
- Correlation analysis with Gold and other market variables
- Feature engineering using technical indicators
- Feature correlation analysis
- LSTM-based time-series modelling
- XGBoost modelling and comparison
- Prediction-error analysis
- Multiple feature-selection experiments
- Evaluation using MAE, RMSE, MAPE and R²

A total of **17 major LSTM experiments** were documented. Different features such as ROC, `Close_Lag_7`, volume ROC, EMA, volatility, high-low range, SMA features and price/SMA ratios were tested. The experiment log shows that some feature additions improved performance while others reduced it, demonstrating why feature selection was treated as an experimental process rather than simply adding more variables. fileciteturn23file0L11-L39 fileciteturn23file1L50-L121

The complete experiment history is preserved in `results/Experiments and Results.pdf` so that the progression from the initial model to the final model can be reviewed.

---

## Dataset

The project uses daily financial-market data.

### Bitcoin Data

The repository contains multiple stages of the Bitcoin dataset:

- `btc_1d_data_2018_to_2025.csv` — historical daily Bitcoin data covering 2018–2025.
- `btc_new_data_june_aug_2026.csv` — newer Bitcoin observations added during the project.
- `dataset_upto_20august.csv` — working dataset prepared up to **20 August 2026**.
- `bitcoin_dxy_merged.csv` — merged Bitcoin and DXY dataset used during the analysis.

The project therefore combines a long historical period with newly collected observations from 2026. The final experiments were performed around the August 2026 evaluation period; the experiment log reports predictions for 21–22 August 2026. fileciteturn23file0L11-L16 fileciteturn23file1L50-L61

### Additional Market Data

The repository also contains:

- `DXY.csv` — U.S. Dollar Index data
- `Gold.csv` — Gold market data
- `SP500.csv` — S&P 500 market data
- `VIX.csv` — Volatility Index data

These variables were investigated to understand whether broader financial-market movements have a relationship with Bitcoin.

### Main Bitcoin Variables

The modelling workflow uses market variables such as:

- Open
- High
- Low
- Close
- Volume

Additional engineered variables include technical indicators and derived features such as:

- RSI
- MACD
- MACD Signal
- ROC
- SMA
- EMA
- Bollinger Bands
- Bollinger Band Width
- Volatility
- High-Low Range
- Close Lag
- Price/SMA ratio
- Other derived price and momentum features

---

## Exploratory Data Analysis

EDA was performed before modelling to understand the structure and behaviour of Bitcoin prices.

Major analyses included:

1. Bitcoin closing-price distribution
2. Daily-return behaviour
3. Bitcoin price versus trading volume
4. Feature correlation heatmap
5. Correlation between Bitcoin and Gold
6. Relationships with broader market indicators
7. Feature importance analysis for XGBoost
8. LSTM prediction-error analysis
9. Prediction-error distribution

The generated analysis figures are stored in the `analysis/` directory.

---

## Feature Engineering

Feature engineering was an important part of the project.

Technical indicators were created from historical price and volume data to provide the model with information about:

- Momentum
- Trend
- Volatility
- Recent price behaviour
- Trading activity

Examples include RSI, MACD, ROC, SMA, EMA, Bollinger Bands, volatility measures, lagged closing prices and price-to-SMA relationships.

Experiments showed that adding a feature does not necessarily improve prediction performance. For example, the experiment log records improvements after adding `SMA_7` and `Price_SMA7_Ratio`, while several other feature additions produced worse R² and RMSE values. fileciteturn23file1L95-L121

---

## Models Tested

### 1. LSTM

The primary model is a **Long Short-Term Memory (LSTM)** neural network.

LSTM was selected because Bitcoin price data is a time series and the model can learn patterns from sequences of previous observations.

The final model uses sequential historical observations to predict the next Bitcoin closing price.

### 2. XGBoost

XGBoost was also investigated as an alternative model.

Feature importance analysis was performed to understand which engineered inputs contributed most to its predictions.

However, XGBoost did not provide sufficiently competitive results for the final objective, so it was not selected as the final model. LSTM was retained because it provided substantially better final predictive performance on the project's evaluation setup.

---

## Final Model Performance

The final LSTM model achieved:

| Metric | Result |
|---|---:|
| MAE | **$5,537.18** |
| RMSE | **$5,919.28** |
| R² | **0.8889** |

### Interpretation

- **MAE** of approximately $5.54K means the model's predictions differ from actual prices by about $5.54K on average.
- **RMSE** of approximately $5.92K indicates the typical prediction error while giving greater weight to larger errors.
- **R² = 0.8889** indicates that the model explains approximately 88.89% of the variance in the target prices on the evaluated test data.

These values represent the final model evaluation, while the experiment PDF records the performance of individual feature-engineering experiments separately.

---

## Error Analysis

Prediction errors were analysed using both:

- Error over test samples
- Error distribution histogram

The error analysis helps identify periods where the model struggles, particularly during unusually large market movements.

This is important because Bitcoin is highly volatile, and a model can perform well during normal market conditions while producing larger errors during sudden price movements.

---

## Repository Structure

```text
bitcoin_price_prediction/
│
├── Notebooks/
│   ├── data_preparation.ipynb
│   ├── exp 01 base.ipynb
│   ├── exp 02 DRX.ipynb
│   ├── finance data download.ipynb
│   └── main_model_notebook.ipynb
│
├── analysis/
│   └── EDA and analysis figures
│
├── dataset/
│   ├── Bitcoin datasets
│   ├── DXY.csv
│   ├── Gold.csv
│   ├── SP500.csv
│   └── VIX.csv
│
├── iframe_figures/
│   └── Interactive Plotly figures
│
├── results/
│   ├── Experiments and Results.pdf
│   └── Final result screenshots
│
├── .gitignore
└── README.md
```

---

## Reproducibility

The notebooks are included in the repository so that the complete workflow can be inspected and reproduced.

Recommended workflow:

1. Download/load the datasets.
2. Perform data preparation.
3. Generate technical indicators.
4. Perform EDA and correlation analysis.
5. Prepare sequential data for LSTM.
6. Train the LSTM model.
7. Generate predictions.
8. Inverse-transform predictions to the original Bitcoin price scale.
9. Evaluate using MAE, RMSE, MAPE and R².
10. Compare results with alternative experiments and XGBoost.

---

## Limitations

Bitcoin prices are influenced by many unpredictable factors, including market sentiment, news, regulation, liquidity and macroeconomic events.

Therefore, the model should be considered a **research and learning project**, not a financial advisory or guaranteed price-prediction system.

The model's historical performance does not guarantee future performance.

---

## Conclusion

This project demonstrates an end-to-end machine-learning workflow for Bitcoin price prediction, starting from raw financial data and EDA, followed by feature engineering, multiple experiments, model comparison and final evaluation.

The key outcome was not simply building an LSTM model, but systematically testing different feature configurations and understanding how they affected prediction performance. The documented experiments helped identify a stronger feature set and ultimately led to the final LSTM model with an **R² of 0.8889** on the evaluated test data.

The complete experimental progression is available in `results/Experiments and Results.pdf`.

---

## Author

**Shivam Ahirwar**

LinkedIn: https://www.linkedin.com/in/shivam-ahirwar-425293311/
