import os
import sys
import json
import pickle
import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np

# Root path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import MODEL_FEATURES, LOOKBACK_WINDOW, MODEL_PATH, SCALER_X_PATH, SCALER_Y_PATH, DB_PATH
from src.prediction import load_model_artifacts, predict_next_day, predict_what_if
from src.data_fetch import fetch_all_market_data
from src.feature_engineering import build_feature_dataframe
from src.history import get_latest_completed_predictions, get_history_kpis, get_all_predictions_df
from src.update_pipeline import run_pipeline

def run_all_tests():
    print("=== STARTING AUTOMATED VERIFICATION SUITE ===")

    # 1. Config Check
    assert len(MODEL_FEATURES) == 17, "Features count must be 17"
    assert LOOKBACK_WINDOW == 30, "Lookback window must be 30"
    print("✓ Check 1: 17 features and 30-day lookback configured.")

    # 2. Model & Scalers Check
    model, scaler_X, scaler_y, meta = load_model_artifacts()
    assert model.input_shape == (None, 30, 17), f"Model input shape mismatch: {model.input_shape}"
    assert scaler_X.n_features_in_ == 17, f"Scaler X features mismatch: {scaler_X.n_features_in_}"
    assert scaler_y.n_features_in_ == 1, f"Scaler y features mismatch: {scaler_y.n_features_in_}"
    print("✓ Check 2: Model & Scalers verified with input shape (30, 17).")

    # 3. Data Ingestion & Features Check
    df_raw, status = fetch_all_market_data()
    assert not df_raw.empty, "Market data dataframe is empty"
    df_feat = build_feature_dataframe(df_raw)
    assert len(df_feat) >= 30, "Insufficient feature rows"
    for f in MODEL_FEATURES:
        assert f in df_feat.columns, f"Missing feature: {f}"
    latest_date_str = str(df_feat.iloc[-1]['Date'])
    print(f"✓ Check 3: Market data fetched & engineered ({len(df_feat)} rows, latest: {latest_date_str}).")

    # 4. Prediction Engine Check
    pred_res = predict_next_day(df_feat)
    assert pred_res['status'] == 'SUCCESS', "Prediction failed"
    assert pred_res['predicted_price'] > 10000, "Unrealistic prediction"
    assert pred_res['current_price'] > 10000, "Unrealistic current price"
    print(f"✓ Check 4: Prediction: Current=${pred_res['current_price']:,.2f}, Next-Day=${pred_res['predicted_price']:,.2f} ({pred_res['expected_change_pct']:+.2f}%).")

    # 5. What-If Simulation Check
    sim_res = predict_what_if(df_feat, {'btc_price_change_pct': 5.0, 'sp500_change_pct': 1.0})
    assert sim_res['status'] == 'SIMULATION'
    print(f"✓ Check 5: What-If simulation: Base=${pred_res['predicted_price']:,.2f} -> Sim=${sim_res['predicted_price']:,.2f}.")

    # 6. SQLite History & KPIs Check
    last_5 = get_latest_completed_predictions(5)
    assert len(last_5) == 5, f"Expected 5 completed predictions, got {len(last_5)}"
    for p in last_5:
        assert p['predicted_price'] is not None and p['actual_price'] is not None
        assert p['error_pct'] is not None

    kpis = get_history_kpis(30)
    assert kpis['status'] == 'OK'
    assert kpis['avg_error_pct'] is not None
    print(f"✓ Check 6: SQLite history verified ({len(last_5)} completed predictions, 30-day mean error: {kpis['avg_error_pct']:.2f}%).")

    # 7. Automated Update Pipeline Check
    pipe_res = run_pipeline(force_refresh=False)
    assert pipe_res['status'] == 'SUCCESS'
    print("✓ Check 7: Automated update pipeline completed end-to-end.")

    print("\n🎉 ALL 7 SYSTEM VERIFICATION CHECKS PASSED SUCCESSFULLY! 🎉")

if __name__ == '__main__':
    run_all_tests()
