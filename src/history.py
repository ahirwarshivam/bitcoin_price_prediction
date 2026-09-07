"""
Persistent Prediction History Module using SQLite.
Tracks daily forecasts, actual closing prices, prediction errors, directional accuracy,
and pre-populates evaluated test historical records for instantaneous 30-day analytics.
"""

import sys
import sqlite3
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import DB_PATH, MODEL_VERSION, LOOKBACK_WINDOW, MODEL_FEATURES, TARGET_COLUMN


def get_db_connection() -> sqlite3.Connection:
    """Returns a SQLite connection configured for row dictionaries."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes the database schema if not already created."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                date TEXT PRIMARY KEY,
                prediction_date TEXT,
                predicted_price REAL NOT NULL,
                actual_price REAL,
                error REAL,
                error_pct REAL,
                prev_close REAL,
                expected_change_pct REAL,
                actual_change_pct REAL,
                direction_predicted TEXT,
                direction_actual TEXT,
                direction_correct INTEGER,
                model_version TEXT NOT NULL,
                is_simulated INTEGER DEFAULT 0,
                created_at TEXT NOT NULL
            );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_date ON predictions(date);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_actual_price ON predictions(actual_price);")
        conn.commit()


def sync_history_with_market_data(df_feat: pd.DataFrame):
    """
    Ensures all historical dates present in df_feat have their model predictions,
    actual closing prices, absolute errors, error percentages, and directional matches
    computed and recorded in the SQLite database.
    """
    init_db()
    if len(df_feat) < LOOKBACK_WINDOW + 1:
        return

    from src.prediction import load_model_artifacts
    model, scaler_X, scaler_y, _ = load_model_artifacts()

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT date FROM predictions WHERE actual_price IS NOT NULL AND is_simulated = 0")
        completed_dates = {row['date'] for row in cursor.fetchall()}

    missing_indices = []
    for i in range(LOOKBACK_WINDOW, len(df_feat)):
        target_date = df_feat.iloc[i]['Date'].strftime('%Y-%m-%d')
        if target_date not in completed_dates:
            missing_indices.append(i)

    if not missing_indices:
        return

    sequences = []
    for idx in missing_indices:
        seq_df = df_feat[MODEL_FEATURES].iloc[idx-LOOKBACK_WINDOW : idx].values
        scaled_seq = scaler_X.transform(seq_df)
        sequences.append(scaled_seq)

    seq_tensor = np.array(sequences)
    preds_scaled = model.predict(seq_tensor, verbose=0)
    preds_usd = scaler_y.inverse_transform(preds_scaled).flatten()

    records_to_insert = []
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    for k, idx in enumerate(missing_indices):
        target_date = df_feat.iloc[idx]['Date'].strftime('%Y-%m-%d')
        pred_date = df_feat.iloc[idx-1]['Date'].strftime('%Y-%m-%d')
        pred_p = float(preds_usd[k])
        act_p = float(df_feat.iloc[idx]['Close'])
        prev_close = float(df_feat.iloc[idx-1]['Close'])

        err = act_p - pred_p
        err_pct = (abs(err) / act_p) * 100.0 if act_p > 0 else 0.0
        exp_change_pct = ((pred_p - prev_close) / prev_close) * 100.0 if prev_close > 0 else 0.0
        act_change_pct = ((act_p - prev_close) / prev_close) * 100.0 if prev_close > 0 else 0.0

        dir_pred = "UP" if pred_p >= prev_close else "DOWN"
        dir_act = "UP" if act_p >= prev_close else "DOWN"
        dir_correct = 1 if dir_pred == dir_act else 0

        records_to_insert.append((
            target_date,
            pred_date,
            pred_p,
            act_p,
            err,
            err_pct,
            prev_close,
            exp_change_pct,
            act_change_pct,
            dir_pred,
            dir_act,
            dir_correct,
            MODEL_VERSION,
            0,
            now_str
        ))

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.executemany("""
            INSERT OR REPLACE INTO predictions (
                date, prediction_date, predicted_price, actual_price,
                error, error_pct, prev_close, expected_change_pct,
                actual_change_pct, direction_predicted, direction_actual,
                direction_correct, model_version, is_simulated, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, records_to_insert)
        conn.commit()


def seed_test_history_if_empty():
    """Initializes schema and runs historical baseline sync."""
    init_db()


def record_prediction(
    target_date: str,
    prediction_date: str,
    predicted_price: float,
    prev_close: float,
    model_version: str = MODEL_VERSION,
    is_simulated: bool = False
):
    """Inserts a new next-day prediction record into SQLite."""
    init_db()
    exp_change_pct = ((predicted_price - prev_close) / prev_close) * 100.0 if prev_close > 0 else 0.0
    dir_pred = "UP" if predicted_price >= prev_close else "DOWN"
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT actual_price FROM predictions WHERE date = ?", (target_date,))
        row = cursor.fetchone()

        if row is None or row['actual_price'] is None:
            cursor.execute("""
                INSERT OR REPLACE INTO predictions (
                    date, prediction_date, predicted_price, prev_close,
                    expected_change_pct, direction_predicted,
                    model_version, is_simulated, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                target_date, prediction_date, float(predicted_price), float(prev_close),
                float(exp_change_pct), dir_pred, model_version, 1 if is_simulated else 0, now_str
            ))
            conn.commit()


def update_actual_price(target_date: str, actual_price: float):
    """
    Updates the actual closing price once available, calculating errors and directional accuracy.
    """
    init_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM predictions WHERE date = ?", (target_date,))
        row = cursor.fetchone()

        if row:
            pred_p = row['predicted_price']
            prev_close = row['prev_close'] or pred_p
            act_p = float(actual_price)

            err = act_p - pred_p
            err_pct = (abs(err) / act_p) * 100.0 if act_p > 0 else 0.0
            act_change_pct = ((act_p - prev_close) / prev_close) * 100.0 if prev_close > 0 else 0.0

            dir_pred = row['direction_predicted']
            dir_act = "UP" if act_p >= prev_close else "DOWN"
            dir_correct = 1 if dir_pred == dir_act else 0

            cursor.execute("""
                UPDATE predictions SET
                    actual_price = ?,
                    error = ?,
                    error_pct = ?,
                    actual_change_pct = ?,
                    direction_actual = ?,
                    direction_correct = ?
                WHERE date = ?
            """, (act_p, err, err_pct, act_change_pct, dir_act, dir_correct, target_date))
            conn.commit()


def get_latest_completed_predictions(limit: int = 5) -> list[dict]:
    """Returns the most recent completed predictions (where actual price is available)."""
    init_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT date, predicted_price, actual_price, error, error_pct,
                   direction_predicted, direction_actual, direction_correct, model_version
            FROM predictions
            WHERE is_simulated = 0 AND actual_price IS NOT NULL
            ORDER BY date DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def get_recent_performance_data(days: int = 30) -> pd.DataFrame:
    """Returns DataFrame of the most recent N completed predictions for Plotly charts."""
    init_db()
    with get_db_connection() as conn:
        query = f"""
            SELECT date, predicted_price, actual_price, error, error_pct,
                   prev_close, expected_change_pct, actual_change_pct,
                   direction_predicted, direction_actual, direction_correct, model_version
            FROM predictions
            WHERE is_simulated = 0 AND actual_price IS NOT NULL
            ORDER BY date DESC
            LIMIT {days}
        """
        df = pd.read_sql_query(query, conn)
        if not df.empty:
            df = df.sort_values('date').reset_index(drop=True)
        return df


def get_history_kpis(days: int = 30) -> dict:
    """Calculates key performance metrics over the last N completed prediction days."""
    df = get_recent_performance_data(days=days)
    if df.empty or len(df) < 5:
        return {
            "status": "Insufficient prediction history",
            "available_days": len(df),
            "avg_error_pct": None,
            "median_error_pct": None,
            "best_prediction_error_pct": None,
            "worst_prediction_error_pct": None,
            "directional_accuracy_pct": None,
            "start_date": None,
            "end_date": None
        }

    errors = df['error_pct'].values
    dir_correct = df['direction_correct'].dropna().values

    return {
        "status": "OK",
        "available_days": len(df),
        "avg_error_pct": float(np.mean(errors)),
        "median_error_pct": float(np.median(errors)),
        "best_prediction_error_pct": float(np.min(errors)),
        "worst_prediction_error_pct": float(np.max(errors)),
        "directional_accuracy_pct": float(np.mean(dir_correct) * 100.0) if len(dir_correct) > 0 else None,
        "start_date": df['date'].min(),
        "end_date": df['date'].max()
    }


def get_all_predictions_df(include_simulated: bool = False) -> pd.DataFrame:
    """Returns complete prediction table for interactive data inspection."""
    init_db()
    with get_db_connection() as conn:
        sim_clause = "" if include_simulated else "WHERE is_simulated = 0"
        query = f"""
            SELECT date, prediction_date, predicted_price, actual_price, error, error_pct,
                   expected_change_pct, direction_predicted, direction_actual,
                   direction_correct, model_version, created_at
            FROM predictions
            {sim_clause}
            ORDER BY date DESC
        """
        return pd.read_sql_query(query, conn)
