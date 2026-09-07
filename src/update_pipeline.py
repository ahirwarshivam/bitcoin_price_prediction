"""
Automated Data Synchronization and Prediction Pipeline.
Fetches the latest market feeds, synchronizes historical evaluation errors,
executes the Standalone LSTM model to generate the next-day forecast, and updates SQLite storage.
Can be triggered manually, via CLI, or through GitHub Actions.
"""

import sys
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import MODEL_VERSION
from src.data_fetch import fetch_all_market_data
from src.feature_engineering import build_feature_dataframe
from src.prediction import predict_next_day
from src.history import (
    sync_history_with_market_data,
    record_prediction,
    get_latest_completed_predictions,
    get_history_kpis
)


def run_pipeline(force_refresh: bool = False) -> dict:
    """Executes the end-to-end update pipeline."""
    print("=" * 60)
    print(f"BITCOIN PREDICTION PIPELINE EXECUTION [{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}]")
    print("=" * 60)

    # 1. Fetch market data
    print("\n[Step 1/4] Fetching latest market data...")
    df, status_info = fetch_all_market_data(force_refresh=force_refresh)
    print(f"Market data status: {status_info['status']} | BTC Source: {status_info['btc_source']} | Macro: {status_info['macro_source']}")
    print(f"Latest record date: {status_info['latest_date']} | BTC Close: ${status_info['latest_btc_price']:,.2f}")

    # 2. Feature Engineering
    print("\n[Step 2/4] Engineering 17 model features...")
    df_feat = build_feature_dataframe(df)
    print(f"Engineered feature set shape: {df_feat.shape}")

    # 3. Synchronize Historical Predictions and Errors in SQLite
    print("\n[Step 3/4] Synchronizing predictions, actual prices, and errors...")
    sync_history_with_market_data(df_feat)
    print("Historical prediction records synchronized up to latest closed market day.")

    # 4. Generate Next-Day Prediction
    print("\n[Step 4/4] Generating next-day price forecast via Standalone LSTM...")
    pred_res = predict_next_day(df_feat)
    target_date = pred_res['target_date']
    pred_price = pred_res['predicted_price']
    current_price = pred_res['current_price']
    exp_change_pct = pred_res['expected_change_pct']
    direction = pred_res['direction']

    print(f"Prediction Date:        {pred_res['latest_date']}")
    print(f"Target Forecast Date:   {target_date}")
    print(f"Current BTC Price:      ${current_price:,.2f}")
    print(f"Predicted Next-Day BTC: ${pred_price:,.2f}")
    print(f"Expected Movement:      {'+' if exp_change_pct >= 0 else ''}{exp_change_pct:.2f}% ({direction})")

    # Record in history
    record_prediction(
        target_date=target_date,
        prediction_date=pred_res['latest_date'],
        predicted_price=pred_price,
        prev_close=current_price,
        model_version=MODEL_VERSION,
        is_simulated=False
    )
    print(f"Saved prediction for {target_date} into persistent SQLite history.")

    # 5. Summary KPIs
    kpis = get_history_kpis(days=30)
    print("\n==========================================")
    print("30-DAY PREDICTION PERFORMANCE SUMMARY")
    print("==========================================")
    print(f"Evaluated Days:       {kpis.get('available_days')}")
    print(f"Average Error:        {kpis.get('avg_error_pct', 0):.2f}%")
    print(f"Median Error:         {kpis.get('median_error_pct', 0):.2f}%")
    print(f"Best Prediction:      {kpis.get('best_prediction_error_pct', 0):.2f}%")
    print(f"Worst Prediction:     {kpis.get('worst_prediction_error_pct', 0):.2f}%")
    print(f"Directional Accuracy: {kpis.get('directional_accuracy_pct', 0):.2f}%")
    print(f"Date Period:          {kpis.get('start_date')} to {kpis.get('end_date')}")
    print("==========================================\n")

    return {
        "status": "SUCCESS",
        "market_status": status_info,
        "prediction": pred_res,
        "kpis": kpis
    }


if __name__ == '__main__':
    run_pipeline(force_refresh=True)
