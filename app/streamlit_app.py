"""
Bitcoin Price Prediction - Interactive Financial Forecasting Terminal.
An end-to-end ML analytics dashboard powered by the Standalone LSTM model.
Author: Shivam Ahirwar (https://www.linkedin.com/in/shivam-ahirwar-425293311/)
"""

import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Configure Root Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import (
    MODEL_VERSION,
    LOOKBACK_WINDOW,
    MODEL_FEATURES,
    THEME_COLORS,
    BENCHMARK_METRICS
)
from src.data_fetch import fetch_all_market_data
from src.feature_engineering import build_feature_dataframe
from src.prediction import predict_next_day, predict_what_if, load_model_artifacts
from src.history import (
    sync_history_with_market_data,
    get_latest_completed_predictions,
    get_recent_performance_data,
    get_history_kpis,
    get_all_predictions_df,
    record_prediction
)
from src.update_pipeline import run_pipeline

# -----------------------------------------------------------------------------
# Streamlit Page Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Bitcoin Price Prediction | AI Terminal",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Custom CSS Styling (Dark Crypto Financial Terminal)
# -----------------------------------------------------------------------------
st.markdown(f"""
<style>
    /* Dark Terminal Theme */
    .stApp {{
        background-color: {THEME_COLORS['dark_bg']};
        color: {THEME_COLORS['text_primary']};
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }}

    /* Global Headings */
    h1, h2, h3, h4, h5, h6 {{
        color: #FFFFFF !important;
        font-weight: 600;
        letter-spacing: -0.02em;
    }}

    /* Top Hero Card */
    .hero-card {{
        background: linear-gradient(135deg, rgba(247, 147, 26, 0.12) 0%, rgba(21, 26, 36, 0.95) 100%);
        border: 1px solid rgba(247, 147, 26, 0.35);
        border-radius: 14px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }}

    /* Metric KPI Cards */
    .kpi-card {{
        background-color: {THEME_COLORS['card_bg']};
        border: 1px solid {THEME_COLORS['card_border']};
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 16px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }}
    .kpi-card:hover {{
        border-color: {THEME_COLORS['bitcoin_orange']};
        transform: translateY(-2px);
    }}

    .kpi-title {{
        color: {THEME_COLORS['text_secondary']};
        font-size: 13px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }}
    .kpi-value {{
        font-size: 26px;
        font-weight: 700;
        color: #FFFFFF;
    }}
    .kpi-subtext {{
        font-size: 13px;
        margin-top: 4px;
    }}

    /* Transparency Card */
    .transparency-box {{
        background-color: {THEME_COLORS['card_bg']};
        border: 1px solid {THEME_COLORS['card_border']};
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 14px;
    }}

    /* Direction Badges */
    .badge-green {{
        background-color: {THEME_COLORS['profit_green_bg']};
        color: {THEME_COLORS['profit_green']};
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 13px;
        display: inline-block;
        border: 1px solid rgba(0, 200, 83, 0.3);
    }}
    .badge-red {{
        background-color: {THEME_COLORS['loss_red_bg']};
        color: {THEME_COLORS['loss_red']};
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 13px;
        display: inline-block;
        border: 1px solid rgba(255, 59, 48, 0.3);
    }}
    .badge-orange {{
        background-color: {THEME_COLORS['bitcoin_orange_glow']};
        color: {THEME_COLORS['bitcoin_orange']};
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 13px;
        display: inline-block;
        border: 1px solid rgba(247, 147, 26, 0.3);
    }}
    .badge-blue {{
        background-color: rgba(14, 156, 252, 0.15);
        color: {THEME_COLORS['ui_blue']};
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 13px;
        display: inline-block;
        border: 1px solid rgba(14, 156, 252, 0.3);
    }}

    /* Target Date Premium Pill */
    .target-date-pill {{
        background: rgba(247, 147, 26, 0.12);
        border: 1px solid rgba(247, 147, 26, 0.45);
        border-radius: 8px;
        padding: 6px 16px;
        display: inline-flex;
        align-items: center;
        gap: 8px;
    }}
    .target-date-label {{
        color: #8B949E;
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }}
    .target-date-value {{
        color: #FFFFFF;
        font-size: 19px;
        font-weight: 800;
        letter-spacing: -0.01em;
    }}

    /* Buttons */
    .stButton>button {{
        background: linear-gradient(135deg, #F7931A 0%, #E67E00 100%);
        color: #FFFFFF;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 8px 18px;
        transition: all 0.2s ease;
    }}
    .stButton>button:hover {{
        background: linear-gradient(135deg, #FFA733 0%, #F7931A 100%);
        box-shadow: 0 4px 14px rgba(247, 147, 26, 0.4);
        color: #FFFFFF;
    }}

    /* Disclaimer box */
    .disclaimer-box {{
        background-color: rgba(21, 26, 36, 0.6);
        border-left: 3px solid {THEME_COLORS['bitcoin_orange']};
        padding: 14px 18px;
        border-radius: 6px;
        font-size: 13px;
        color: {THEME_COLORS['text_secondary']};
        line-height: 1.5;
        margin-top: 30px;
    }}

    /* Sidebar info card */
    .sidebar-card {{
        background-color: #11151E;
        border: 1px solid #1D2330;
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 16px;
        font-size: 13px;
    }}
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Data Loading & Initialization
# -----------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def get_cached_market_data():
    """Fetches market data with 5-minute caching."""
    df, status_info = fetch_all_market_data(force_refresh=False)
    df_feat = build_feature_dataframe(df)
    return df, df_feat, status_info


with st.spinner("Connecting to live cryptocurrency and market data feeds..."):
    df_raw, df_feat, market_status = get_cached_market_data()

st.write("DEBUG raw shape:", df_raw.shape)
st.write("DEBUG feature shape:", df_feat.shape)
st.write("DEBUG market status:", market_status)
st.write("DEBUG latest raw rows:", df_raw.tail(3))
st.write("DEBUG latest feature rows:", df_feat.tail(3))

# Synchronize historical prediction errors up to the latest closed market day
sync_history_with_market_data(df_feat)

# Load Model Artifacts & Metadata (Source of Truth)
try:
    model, scaler_X, scaler_y, model_meta = load_model_artifacts()
except Exception as e:
    st.error(f"Failed to load model artifacts: {e}")
    st.stop()

# Extract Dynamic Model Metadata Values
meta_version = model_meta.get("model_version", MODEL_VERSION)
meta_type = model_meta.get("model_type", "Standalone LSTM")
meta_lookback = model_meta.get("lookback_window", LOOKBACK_WINDOW)
meta_n_features = model_meta.get("n_features", len(MODEL_FEATURES))
meta_target = model_meta.get("target", "Close")
meta_split_ratio = model_meta.get("train_split_ratio", 0.8)
meta_train_start = model_meta.get("training_data_date_range", {}).get("start", "2018-02-04")
meta_train_end = model_meta.get("training_data_date_range", {}).get("end", "2026-08-20")
meta_total_samples = model_meta.get("total_samples", len(df_raw))
meta_test_samples = model_meta.get("test_samples", 624)

# Format Dates Dynamically
today_dt = datetime.now(timezone.utc)
today_date_formatted = today_dt.strftime('%b %d, %Y')
latest_data_date_raw = market_status.get("latest_date", df_raw['Date'].max().strftime('%Y-%m-%d'))
latest_data_date_formatted = pd.to_datetime(latest_data_date_raw).strftime('%b %d, %Y')
training_end_formatted = pd.to_datetime(meta_train_end).strftime('%b %d, %Y')
training_start_formatted = pd.to_datetime(meta_train_start).strftime('%b %d, %Y')

# Calculate Live Prediction
try:
    pred_info = predict_next_day(df_feat)
    record_prediction(
        target_date=pred_info['target_date'],
        prediction_date=pred_info['latest_date'],
        predicted_price=pred_info['predicted_price'],
        prev_close=pred_info['current_price'],
        model_version=meta_version,
        is_simulated=False
    )
except Exception as e:
    st.error(f"Failed to generate prediction: {e}")
    st.stop()

# Load 30-Day History & KPIs (Computed over completed continuous historical days)
kpis_30d = get_history_kpis(days=30)
recent_perf_df = get_recent_performance_data(days=30)
last_5_preds = get_latest_completed_predictions(limit=5)


# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### **Bitcoin Terminal**")
    st.markdown(f"<span class='badge-orange'>{meta_type} v1.0 Production</span>", unsafe_allow_html=True)

    st.markdown("---")

    # 1. DATA STATUS SECTION
    st.markdown("#### **DATA STATUS**")
    st.markdown(f"""
    <div class='sidebar-card'>
        <b>Today's Date:</b> {today_date_formatted}<br>
        <b>Latest Market Data:</b> {latest_data_date_formatted}<br>
        <b>Last Model Training:</b> {training_end_formatted}<br>
        <b>Data Feed Status:</b> <span style='color:{THEME_COLORS["profit_green"]}'>Connected</span><br>
        <b>BTC Data Source:</b> {market_status.get('btc_source', 'Binance API')}<br>
        <b>Macro Data Source:</b> {market_status.get('macro_source', 'Yahoo Finance')}
    </div>
    """, unsafe_allow_html=True)

    if st.button("Refresh Live Feeds", use_container_width=True):
        with st.spinner("Syncing latest market observations..."):
            pipeline_res = run_pipeline(force_refresh=True)
            st.cache_data.clear()
            st.rerun()

    st.markdown("---")

    # 2. MODEL STATUS SECTION
    st.markdown("#### **MODEL STATUS**")
    st.markdown(f"""
    <div class='sidebar-card'>
        <b>Model Version:</b> {meta_version}<br>
        <b>Model Type:</b> {meta_type}<br>
        <b>Model Status:</b> <span style='color:{THEME_COLORS["ui_teal"]}; font-weight:600;'>Production</span><br>
        <b>Retraining Status:</b> <span style='color:{THEME_COLORS["text_secondary"]}; font-weight:600;'>Scheduled</span><br>
        <b>Lookback Window:</b> {meta_lookback} Days<br>
        <b>Input Features:</b> {meta_n_features}<br>
        <b>Target:</b> BTC {meta_target}<br>
        <b>Training Range:</b> {meta_train_start} &rarr; {meta_train_end}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### **Author & Project**")
    st.markdown(f"""
    <div class='sidebar-card'>
        <b>Developer:</b> Shivam Ahirwar<br>
        <a href="https://www.linkedin.com/in/shivam-ahirwar-425293311/" target="_blank" style="color:{THEME_COLORS['ui_blue']}; text-decoration:none; font-weight:600;">
            View LinkedIn Profile
        </a><br><br>
        <i>Complete experiment logs available in <code>results/Experiments and Results.pdf</code></i>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# MAIN HEADER
# -----------------------------------------------------------------------------
header_col1, header_col2 = st.columns([2.8, 1.2])

with header_col1:
    st.title("Bitcoin Price Prediction")
    st.markdown(
        "<p style='color:#8B949E; font-size:16px; margin-top:-10px;'>"
        "Professional Next-Day Bitcoin Price Forecasting using Deep Recurrent LSTM"
        "</p>",
        unsafe_allow_html=True
    )

with header_col2:
    st.markdown(
        f"""
        <div style='text-align: right; padding-top: 10px;'>
            <span class='badge-blue' style='font-size: 13px; font-weight: 600; padding: 6px 12px;'>Status: LIVE</span> &nbsp;
            <span class='badge-orange' style='font-size: 13px; font-weight: 600; padding: 6px 12px;'>Today's Date: {today_date_formatted}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# DATA & MODEL TRANSPARENCY BANNER
# -----------------------------------------------------------------------------
st.markdown(f"""
<div style='background-color: rgba(14, 156, 252, 0.08); border-left: 3px solid {THEME_COLORS["ui_blue"]}; padding: 10px 16px; border-radius: 6px; font-size: 13px; color: {THEME_COLORS["text_primary"]}; margin-bottom: 20px; line-height: 1.5;'>
    <b>Data & Model Transparency:</b> Latest market data ({latest_data_date_formatted}) may be newer than the data used to train the current production model. The production model (<code>{meta_version}</code>) currently uses training data through <b>{training_end_formatted}</b>. New market data is used for inference; model weights are not automatically updated unless the retraining process is executed. Retraining status: <b>Scheduled</b>.
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# TOP KPI METRICS BAR
# -----------------------------------------------------------------------------
kpi_cols = st.columns(5)

current_p = pred_info['current_price']
pred_p = pred_info['predicted_price']
exp_change_pct = pred_info['expected_change_pct']
exp_change_usd = pred_info['expected_change_usd']
direction = pred_info['direction']

with kpi_cols[0]:
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-title'>Current BTC Price</div>
        <div class='kpi-value'>${current_p:,.2f}</div>
        <div class='kpi-subtext' style='color:{THEME_COLORS["text_secondary"]}'>Market Date: {latest_data_date_formatted}</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_cols[1]:
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-title'>Next-Day Forecast</div>
        <div class='kpi-value' style='color:{THEME_COLORS["bitcoin_orange"]}'>${pred_p:,.2f}</div>
        <div class='kpi-subtext' style='color:#FFFFFF; font-size: 13px; font-weight: 600;'>
            <span style='color:{THEME_COLORS["text_secondary"]}; font-weight: 500;'>Target Date:</span> <b>{pred_info["target_date"]}</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

with kpi_cols[2]:
    badge_class = "badge-green" if exp_change_pct >= 0 else "badge-red"
    sign = "+" if exp_change_pct >= 0 else ""
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-title'>Expected Movement</div>
        <div class='kpi-value' style='color:{"#00C853" if exp_change_pct >= 0 else "#FF3B30"}'>{sign}{exp_change_pct:.2f}%</div>
        <div class='kpi-subtext'><span class='{badge_class}'>{sign}${exp_change_usd:,.2f} ({direction})</span></div>
    </div>
    """, unsafe_allow_html=True)

with kpi_cols[3]:
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-title'>Model R² Benchmark</div>
        <div class='kpi-value' style='color:{THEME_COLORS["ui_teal"]}'>{BENCHMARK_METRICS["R2"]:.4f}</div>
        <div class='kpi-subtext' style='color:{THEME_COLORS["text_secondary"]}'>MAE: ${BENCHMARK_METRICS["MAE"]:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_cols[4]:
    avg_err = kpis_30d.get('avg_error_pct')
    err_display = f"{avg_err:.2f}%" if avg_err is not None else "N/A"
    dir_acc = kpis_30d.get('directional_accuracy_pct')
    dir_display = f"Dir Acc: {dir_acc:.1f}%" if dir_acc is not None else "30-Day Mean"
    st.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-title'>30-Day Mean Error</div>
        <div class='kpi-value' style='color:#FFFFFF'>{err_display}</div>
        <div class='kpi-subtext' style='color:{THEME_COLORS["text_secondary"]}'>{dir_display}</div>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# MAIN PREDICTION HERO CARD
# -----------------------------------------------------------------------------
st.markdown(f"""
<div class='hero-card'>
    <div style='display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 18px;'>
        <div>
            <div style='display: flex; align-items: center; gap: 14px; margin-bottom: 12px; flex-wrap: wrap;'>
                <span class='badge-orange' style='font-size: 14px; font-weight: 700; padding: 6px 14px; letter-spacing: 0.04em;'>MODEL PREDICTION</span>
                <div class='target-date-pill'>
                    <span class='target-date-label'>Target Date:</span>
                    <span class='target-date-value'>{pred_info["target_date"]}</span>
                </div>
            </div>
            <div style='display: flex; align-items: baseline; gap: 20px; flex-wrap: wrap; margin-top: 12px;'>
                <div>
                    <span style='color: {THEME_COLORS["text_secondary"]}; font-size: 14px;'>Current Close</span><br>
                    <span style='font-size: 28px; font-weight: 700;'>${current_p:,.2f}</span>
                </div>
                <div style='font-size: 28px; color: {THEME_COLORS["bitcoin_orange"]}; font-weight: 300;'>&rarr;</div>
                <div>
                    <span style='color: {THEME_COLORS["text_secondary"]}; font-size: 14px;'>Predicted Close</span><br>
                    <span style='font-size: 34px; font-weight: 800; color: {THEME_COLORS["bitcoin_orange"]};'>${pred_p:,.2f}</span>
                </div>
                <div style='margin-left: 10px;'>
                    <span style='color: {THEME_COLORS["text_secondary"]}; font-size: 14px;'>Expected 24h Delta</span><br>
                    <span class='{"badge-green" if exp_change_pct >= 0 else "badge-red"}' style='font-size: 16px; padding: 6px 14px;'>
                        {sign}{exp_change_pct:.2f}% ({sign}${exp_change_usd:,.2f})
                    </span>
                </div>
            </div>
        </div>
        <div style='text-align: right; max-width: 360px;'>
            <div style='font-size: 12px; color: {THEME_COLORS["text_secondary"]}; line-height: 1.45;'>
                <b>How the Forecast Works:</b> Forecast generated using the latest available 30-day sequence of 17 input features and the current production LSTM model (<code>{meta_version}</code>). The model is currently used for inference on newly available market data; new data does not automatically retrain the model.
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# EXPANDABLE DATA & MODEL TRANSPARENCY SECTION
# -----------------------------------------------------------------------------
with st.expander("Model & Data Transparency"):
    trans_col1, trans_col2 = st.columns(2)
    with trans_col1:
        st.markdown(f"""
        <div class='transparency-box'>
            <div style='font-weight:700; color:{THEME_COLORS["bitcoin_orange"]}; margin-bottom:10px; text-transform:uppercase; font-size:13px;'>Data Pipeline Specifications</div>
            <table style='width:100%; font-size:13px; color:{THEME_COLORS["text_primary"]};'>
                <tr><td style='color:{THEME_COLORS["text_secondary"]}; padding:4px 0;'>Today's Date:</td><td><b>{today_date_formatted}</b></td></tr>
                <tr><td style='color:{THEME_COLORS["text_secondary"]}; padding:4px 0;'>Latest Market Data:</td><td><b>{latest_data_date_formatted}</b> ({latest_data_date_raw})</td></tr>
                <tr><td style='color:{THEME_COLORS["text_secondary"]}; padding:4px 0;'>Last Model Training Data:</td><td><b>{training_end_formatted}</b> ({meta_train_end})</td></tr>
                <tr><td style='color:{THEME_COLORS["text_secondary"]}; padding:4px 0;'>Training Date Range:</td><td>{meta_train_start} &rarr; {meta_train_end}</td></tr>
                <tr><td style='color:{THEME_COLORS["text_secondary"]}; padding:4px 0;'>Total Training Samples:</td><td>{meta_total_samples:,} days</td></tr>
                <tr><td style='color:{THEME_COLORS["text_secondary"]}; padding:4px 0;'>Test Evaluation Samples:</td><td>{meta_test_samples:,} days ({int((1-meta_split_ratio)*100)}% split)</td></tr>
                <tr><td style='color:{THEME_COLORS["text_secondary"]}; padding:4px 0;'>Data Feed Health:</td><td><span style='color:{THEME_COLORS["profit_green"]}; font-weight:600;'>Connected</span></td></tr>
                <tr><td style='color:{THEME_COLORS["text_secondary"]}; padding:4px 0;'>Bitcoin Data Feed:</td><td>Binance API (BTCUSDT)</td></tr>
                <tr><td style='color:{THEME_COLORS["text_secondary"]}; padding:4px 0;'>Macroeconomic Feed:</td><td>Yahoo Finance (^GSPC, DX-Y.NYB, GC=F, ^VIX)</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
    with trans_col2:
        st.markdown(f"""
        <div class='transparency-box'>
            <div style='font-weight:700; color:{THEME_COLORS["bitcoin_orange"]}; margin-bottom:10px; text-transform:uppercase; font-size:13px;'>Model Artifact & Retraining Status</div>
            <table style='width:100%; font-size:13px; color:{THEME_COLORS["text_primary"]};'>
                <tr><td style='color:{THEME_COLORS["text_secondary"]}; padding:4px 0;'>Model Version:</td><td><code>{meta_version}</code></td></tr>
                <tr><td style='color:{THEME_COLORS["text_secondary"]}; padding:4px 0;'>Model Type:</td><td>{meta_type}</td></tr>
                <tr><td style='color:{THEME_COLORS["text_secondary"]}; padding:4px 0;'>Deployment State:</td><td><span style='color:{THEME_COLORS["ui_teal"]}; font-weight:600;'>Production</span></td></tr>
                <tr><td style='color:{THEME_COLORS["text_secondary"]}; padding:4px 0;'>Retraining Status:</td><td><span style='color:{THEME_COLORS["text_secondary"]}; font-weight:600;'>Scheduled</span></td></tr>
                <tr><td style='color:{THEME_COLORS["text_secondary"]}; padding:4px 0;'>Lookback Window:</td><td>{meta_lookback} Days (Temporal Steps)</td></tr>
                <tr><td style='color:{THEME_COLORS["text_secondary"]}; padding:4px 0;'>Input Features:</td><td>{meta_n_features} Engineered Features</td></tr>
                <tr><td style='color:{THEME_COLORS["text_secondary"]}; padding:4px 0;'>Target Variable:</td><td>BTC {meta_target}</td></tr>
                <tr><td style='color:{THEME_COLORS["text_secondary"]}; padding:4px 0;'>Train / Test Split:</td><td>{int(meta_split_ratio*100)}% / {int((1-meta_split_ratio)*100)}% Chronological</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style='font-size:12px; color:{THEME_COLORS["text_secondary"]}; background-color:#11151E; border:1px solid #1D2330; padding:10px 14px; border-radius:6px; margin-top:8px;'>
        <b>Operational Note:</b> New market data is used for inference by the current production model. The model weights are not automatically updated unless a retraining process is executed.
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# TABS INTERFACE
# -----------------------------------------------------------------------------
tab_overview, tab_performance, tab_macro, tab_whatif, tab_research = st.tabs([
    "Price Charts & Trends",
    "30-Day Model Performance & Error Analysis",
    "Macroeconomic Indicators",
    "What-If Scenario Simulation",
    "Model Architecture & Experiments"
])


# -----------------------------------------------------------------------------
# TAB 1: PRICE CHARTS & TRENDS
# -----------------------------------------------------------------------------
with tab_overview:
    col_ctrl, _ = st.columns([2, 4])
    with col_ctrl:
        timeframe = st.radio(
            "Select Historical Timeframe:",
            ["7 Days", "30 Days", "90 Days", "1 Year", "All History"],
            index=1,
            horizontal=True
        )

    # Filter data based on selection
    now_dt = df_feat['Date'].max()
    if timeframe == "7 Days":
        chart_df = df_feat[df_feat['Date'] >= (now_dt - timedelta(days=7))].copy()
    elif timeframe == "30 Days":
        chart_df = df_feat[df_feat['Date'] >= (now_dt - timedelta(days=30))].copy()
    elif timeframe == "90 Days":
        chart_df = df_feat[df_feat['Date'] >= (now_dt - timedelta(days=90))].copy()
    elif timeframe == "1 Year":
        chart_df = df_feat[df_feat['Date'] >= (now_dt - timedelta(days=365))].copy()
    else:
        chart_df = df_feat.copy()

    # Candlestick / Line Chart with Bollinger Bands
    fig_price = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.75, 0.25]
    )

    # Main Price Trace
    fig_price.add_trace(
        go.Scatter(
            x=chart_df['Date'],
            y=chart_df['Close'],
            name='Bitcoin Price (USD)',
            line=dict(color=THEME_COLORS['bitcoin_orange'], width=2.5),
            hovertemplate='<b>Date</b>: %{x|%b %d, %Y}<br><b>Price</b>: $%{y:,.2f}<extra></extra>'
        ),
        row=1, col=1
    )

    # 7-Day SMA
    fig_price.add_trace(
        go.Scatter(
            x=chart_df['Date'],
            y=chart_df['SMA_7'],
            name='7-Day SMA',
            line=dict(color=THEME_COLORS['ui_blue'], width=1.5, dash='dot'),
            hovertemplate='<b>7-Day SMA</b>: $%{y:,.2f}<extra></extra>'
        ),
        row=1, col=1
    )

    # Bollinger Bands
    fig_price.add_trace(
        go.Scatter(
            x=chart_df['Date'],
            y=chart_df['BB_Upper'],
            name='Bollinger Upper',
            line=dict(color='rgba(255, 255, 255, 0.2)', width=1),
            showlegend=False,
            hoverinfo='skip'
        ),
        row=1, col=1
    )
    fig_price.add_trace(
        go.Scatter(
            x=chart_df['Date'],
            y=chart_df['BB_Lower'],
            name='Bollinger Lower',
            line=dict(color='rgba(255, 255, 255, 0.2)', width=1),
            fill='tonexty',
            fillcolor='rgba(247, 147, 26, 0.04)',
            showlegend=False,
            hoverinfo='skip'
        ),
        row=1, col=1
    )

    # Volume Bar Chart
    colors_vol = [
        THEME_COLORS['profit_green'] if ret >= 0 else THEME_COLORS['loss_red']
        for ret in chart_df['Daily_Return']
    ]
    fig_price.add_trace(
        go.Bar(
            x=chart_df['Date'],
            y=chart_df['Volume'],
            name='Volume',
            marker_color=colors_vol,
            opacity=0.6,
            hovertemplate='<b>Volume</b>: %{y:,.0f}<extra></extra>'
        ),
        row=2, col=1
    )

    fig_price.update_layout(
        title=f"Bitcoin Price Movement & Trading Volume ({timeframe})",
        paper_bgcolor=THEME_COLORS['dark_bg'],
        plot_bgcolor=THEME_COLORS['card_bg'],
        font=dict(color=THEME_COLORS['text_primary'], family="-apple-system, sans-serif"),
        hovermode='x unified',
        height=520,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(0,0,0,0)'
        )
    )
    fig_price.update_xaxes(showgrid=True, gridcolor=THEME_COLORS['grid_color'], gridwidth=0.5)
    fig_price.update_yaxes(title_text="Price (USD)", row=1, col=1, showgrid=True, gridcolor=THEME_COLORS['grid_color'], tickformat="$,.0f")
    fig_price.update_yaxes(title_text="Volume", row=2, col=1, showgrid=True, gridcolor=THEME_COLORS['grid_color'])

    st.plotly_chart(fig_price, use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 2: 30-DAY PERFORMANCE & ERROR ANALYSIS
# -----------------------------------------------------------------------------
with tab_performance:
    st.markdown("### **Model Evaluation & Prediction History**")
    st.markdown(
        "<p style='color:#8B949E; font-size:14px; margin-top:-8px;'>"
        "Tracking actual market realizations against LSTM predictions over recent trading days."
        "</p>",
        unsafe_allow_html=True
    )

    # 30-Day KPI Row
    perf_kpi_cols = st.columns(5)
    with perf_kpi_cols[0]:
        st.metric("30-Day Mean Error", f"{kpis_30d.get('avg_error_pct', 0):.2f}%")
    with perf_kpi_cols[1]:
        st.metric("30-Day Median Error", f"{kpis_30d.get('median_error_pct', 0):.2f}%")
    with perf_kpi_cols[2]:
        st.metric("Best Prediction Error", f"{kpis_30d.get('best_prediction_error_pct', 0):.2f}%")
    with perf_kpi_cols[3]:
        st.metric("Worst Prediction Error", f"{kpis_30d.get('worst_prediction_error_pct', 0):.2f}%")
    with perf_kpi_cols[4]:
        dir_acc_val = kpis_30d.get('directional_accuracy_pct')
        st.metric("Directional Accuracy", f"{dir_acc_val:.1f}%" if dir_acc_val is not None else "N/A")

    if not recent_perf_df.empty:
        # Chart 1: Actual vs Predicted (Two-Line Plotly Chart)
        fig_actual_pred = go.Figure()

        # Format customdata for tooltips
        custom_data = np.stack((
            recent_perf_df['actual_price'],
            recent_perf_df['predicted_price'],
            recent_perf_df['error'].abs(),
            recent_perf_df['error_pct']
        ), axis=-1)

        # Actual Price Line
        fig_actual_pred.add_trace(
            go.Scatter(
                x=pd.to_datetime(recent_perf_df['date']),
                y=recent_perf_df['actual_price'],
                name='Actual Bitcoin Price',
                mode='lines+markers',
                line=dict(color=THEME_COLORS['ui_blue'], width=3),
                marker=dict(size=6, color=THEME_COLORS['ui_blue']),
                customdata=custom_data,
                hovertemplate=(
                    "<b>Date:</b> %{x|%b %d, %Y}<br>"
                    "<b>Actual Price:</b> $%{customdata[0]:,.2f}<br>"
                    "<b>Predicted Price:</b> $%{customdata[1]:,.2f}<br>"
                    "<b>Absolute Error:</b> $%{customdata[2]:,.2f}<br>"
                    "<b>Error Percentage:</b> %{customdata[3]:.2f}%<extra></extra>"
                )
            )
        )

        # Predicted Price Line
        fig_actual_pred.add_trace(
            go.Scatter(
                x=pd.to_datetime(recent_perf_df['date']),
                y=recent_perf_df['predicted_price'],
                name='Predicted Bitcoin Price',
                mode='lines+markers',
                line=dict(color=THEME_COLORS['bitcoin_orange'], width=3, dash='dash'),
                marker=dict(size=6, color=THEME_COLORS['bitcoin_orange']),
                customdata=custom_data,
                hovertemplate=(
                    "<b>Date:</b> %{x|%b %d, %Y}<br>"
                    "<b>Predicted Price:</b> $%{customdata[1]:,.2f}<br>"
                    "<b>Actual Price:</b> $%{customdata[0]:,.2f}<br>"
                    "<b>Error:</b> %{customdata[3]:.2f}%<extra></extra>"
                )
            )
        )

        fig_actual_pred.update_layout(
            title="30-Day Completed Predictions: Actual Price vs LSTM Forecast",
            paper_bgcolor=THEME_COLORS['dark_bg'],
            plot_bgcolor=THEME_COLORS['card_bg'],
            font=dict(color=THEME_COLORS['text_primary']),
            hovermode='x unified',
            height=460,
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor='rgba(0,0,0,0)'
            )
        )
        fig_actual_pred.update_xaxes(showgrid=True, gridcolor=THEME_COLORS['grid_color'], gridwidth=0.5)
        fig_actual_pred.update_yaxes(title_text="Bitcoin Price (USD)", showgrid=True, gridcolor=THEME_COLORS['grid_color'], tickformat="$,.0f")

        st.plotly_chart(fig_actual_pred, use_container_width=True)

        # Chart 2: Prediction Error Percentage Analysis
        fig_err = go.Figure()

        # Dynamic bar colors based on error thresholds
        err_colors = [
            THEME_COLORS['profit_green'] if err <= 3.0 else (
                THEME_COLORS['bitcoin_orange'] if err <= 6.0 else THEME_COLORS['loss_red']
            ) for err in recent_perf_df['error_pct']
        ]

        fig_err.add_trace(
            go.Bar(
                x=pd.to_datetime(recent_perf_df['date']),
                y=recent_perf_df['error_pct'],
                name='Error %',
                marker_color=err_colors,
                hovertemplate='<b>Date:</b> %{x|%b %d, %Y}<br><b>Error Percentage:</b> %{y:.2f}%<extra></extra>'
            )
        )

        # Horizontal Average Error Line
        if kpis_30d.get('avg_error_pct'):
            fig_err.add_hline(
                y=kpis_30d['avg_error_pct'],
                line_dash="dash",
                line_color=THEME_COLORS['ui_teal'],
                annotation_text=f"30-Day Mean: {kpis_30d['avg_error_pct']:.2f}%",
                annotation_position="top left"
            )

        fig_err.update_layout(
            title="30-Day Forecast Error Percentage Distribution",
            paper_bgcolor=THEME_COLORS['dark_bg'],
            plot_bgcolor=THEME_COLORS['card_bg'],
            font=dict(color=THEME_COLORS['text_primary']),
            height=320,
            margin=dict(l=20, r=20, t=50, b=20),
            showlegend=False
        )
        fig_err.update_xaxes(showgrid=True, gridcolor=THEME_COLORS['grid_color'])
        fig_err.update_yaxes(title_text="Absolute Error (%)", showgrid=True, gridcolor=THEME_COLORS['grid_color'])

        st.plotly_chart(fig_err, use_container_width=True)

    # Previous 5 Completed Predictions Table
    st.markdown("#### **Recent Completed Predictions (Last 5 Days)**")
    if last_5_preds:
        table_rows = []
        for p in last_5_preds:
            p_date = p['date']
            pred_val = f"${p['predicted_price']:,.2f}"
            act_val = f"${p['actual_price']:,.2f}" if p['actual_price'] else "Pending"
            err_val = f"${abs(p['error']):,.2f}" if p['error'] is not None else "—"
            err_pct_val = f"{p['error_pct']:.2f}%" if p['error_pct'] is not None else "—"
            dir_pred_badge = p['direction_predicted']
            dir_act_badge = p['direction_actual'] if p['direction_actual'] else "—"
            match_badge = "Match" if p['direction_correct'] == 1 else ("Diverged" if p['direction_correct'] == 0 else "Pending")

            table_rows.append({
                "Forecast Date": p_date,
                "Predicted Price": pred_val,
                "Actual Price": act_val,
                "Absolute Error": err_val,
                "Error %": err_pct_val,
                "Predicted Dir": dir_pred_badge,
                "Actual Dir": dir_act_badge,
                "Direction Match": match_badge,
                "Model Version": p['model_version']
            })

        st.dataframe(
            pd.DataFrame(table_rows),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Insufficient completed predictions available.")

    # Collapsible Full History Table
    with st.expander("View Complete Prediction History Database"):
        all_preds = get_all_predictions_df(include_simulated=False)
        st.dataframe(all_preds, use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 3: MACROECONOMIC INDICATORS
# -----------------------------------------------------------------------------
with tab_macro:
    st.markdown("### **Macroeconomic Indicators & Correlation Dynamics**")
    st.markdown(
        "<p style='color:#8B949E; font-size:14px; margin-top:-8px;'>"
        "Tracking external market drivers integrated into the model's feature space."
        "</p>",
        unsafe_allow_html=True
    )

    latest_row = df_feat.iloc[-1]
    prev_row = df_feat.iloc[-2] if len(df_feat) > 1 else latest_row

    macro_kpi_cols = st.columns(4)

    # 1. S&P 500
    sp_curr = latest_row.get('SP500', 0)
    sp_prev = prev_row.get('SP500', sp_curr)
    sp_chg = ((sp_curr - sp_prev) / sp_prev * 100) if sp_prev > 0 else 0
    with macro_kpi_cols[0]:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-title'>S&P 500 Index (^GSPC)</div>
            <div class='kpi-value'>{sp_curr:,.2f}</div>
            <div class='kpi-subtext'>
                <span class='{"badge-green" if sp_chg >= 0 else "badge-red"}'>{"+" if sp_chg >= 0 else ""}{sp_chg:.2f}% (Corr: 0.90)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 2. DXY
    dxy_curr = latest_row.get('DXY', 0)
    dxy_prev = prev_row.get('DXY', dxy_curr)
    dxy_chg = ((dxy_curr - dxy_prev) / dxy_prev * 100) if dxy_prev > 0 else 0
    with macro_kpi_cols[1]:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-title'>US Dollar Index (DXY)</div>
            <div class='kpi-value'>{dxy_curr:,.2f}</div>
            <div class='kpi-subtext'>
                <span class='{"badge-green" if dxy_chg >= 0 else "badge-red"}'>{"+" if dxy_chg >= 0 else ""}{dxy_chg:.2f}% (Macro Baseline)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 3. Gold
    gold_curr = latest_row.get('Gold', 0)
    gold_prev = prev_row.get('Gold', gold_curr)
    gold_chg = ((gold_curr - gold_prev) / gold_prev * 100) if gold_prev > 0 else 0
    with macro_kpi_cols[2]:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-title'>Gold (GC=F)</div>
            <div class='kpi-value'>${gold_curr:,.2f}</div>
            <div class='kpi-subtext'>
                <span class='{"badge-green" if gold_chg >= 0 else "badge-red"}'>{"+" if gold_chg >= 0 else ""}{gold_chg:.2f}% (Corr: 0.79)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 4. VIX
    vix_curr = latest_row.get('VIX', 0)
    vix_prev = prev_row.get('VIX', vix_curr)
    vix_chg = ((vix_curr - vix_prev) / vix_prev * 100) if vix_prev > 0 else 0
    with macro_kpi_cols[3]:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-title'>Volatility Index (VIX)</div>
            <div class='kpi-value'>{vix_curr:,.2f}</div>
            <div class='kpi-subtext'>
                <span class='{"badge-green" if vix_chg <= 0 else "badge-red"}'>{"+" if vix_chg >= 0 else ""}{vix_chg:.2f}% (Risk Gauge)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Correlation Plot: BTC vs S&P 500
    fig_macro = make_subplots(specs=[[{"secondary_y": True}]])
    recent_macro_df = df_feat.tail(180)

    fig_macro.add_trace(
        go.Scatter(
            x=recent_macro_df['Date'],
            y=recent_macro_df['Close'],
            name='Bitcoin Price (USD)',
            line=dict(color=THEME_COLORS['bitcoin_orange'], width=2.5)
        ),
        secondary_y=False
    )
    fig_macro.add_trace(
        go.Scatter(
            x=recent_macro_df['Date'],
            y=recent_macro_df['SP500'],
            name='S&P 500 Index',
            line=dict(color=THEME_COLORS['ui_teal'], width=2, dash='dash')
        ),
        secondary_y=True
    )

    corr_val = recent_macro_df['Close'].corr(recent_macro_df['SP500'])
    fig_macro.update_layout(
        title=f"Bitcoin vs S&P 500 Co-Movement (180-Day Correlation: {corr_val:.3f})",
        paper_bgcolor=THEME_COLORS['dark_bg'],
        plot_bgcolor=THEME_COLORS['card_bg'],
        font=dict(color=THEME_COLORS['text_primary']),
        hovermode='x unified',
        height=450,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    fig_macro.update_xaxes(showgrid=True, gridcolor=THEME_COLORS['grid_color'])
    fig_macro.update_yaxes(title_text="Bitcoin Price (USD)", secondary_y=False, showgrid=True, gridcolor=THEME_COLORS['grid_color'], tickformat="$,.0f")
    fig_macro.update_yaxes(title_text="S&P 500 Index", secondary_y=True, showgrid=False)

    st.plotly_chart(fig_macro, use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 4: WHAT-IF SCENARIO SIMULATION
# -----------------------------------------------------------------------------
with tab_whatif:
    st.markdown("### **What-If Scenario Simulation (Hypothetical Mode)**")
    st.markdown(
        "<p style='color:#8B949E; font-size:14px; margin-top:-8px;'>"
        "Adjust market conditions manually to evaluate how the LSTM model's forecast reacts to hypothetical shocks. "
        "<b>Note:</b> These adjustments are strictly hypothetical and do not alter live prediction records."
        "</p>",
        unsafe_allow_html=True
    )

    col_sim_controls, col_sim_results = st.columns([1, 1], gap="large")

    with col_sim_controls:
        st.markdown("#### **Hypothetical Market Shocks**")

        sim_btc_chg = st.slider(
            "Hypothetical Bitcoin 24h Price Shock (%):",
            min_value=-15.0,
            max_value=15.0,
            value=0.0,
            step=0.5,
            format="%.1f%%"
        )

        sim_sp_chg = st.slider(
            "Hypothetical S&P 500 Index Shock (%):",
            min_value=-8.0,
            max_value=8.0,
            value=0.0,
            step=0.5,
            format="%.1f%%"
        )

        sim_vol_mult = st.slider(
            "Trading Volume Multiplier:",
            min_value=0.5,
            max_value=3.0,
            value=1.0,
            step=0.1,
            format="%.1fx"
        )

        sim_rsi = st.slider(
            "Override Relative Strength Index (RSI):",
            min_value=15.0,
            max_value=90.0,
            value=float(round(latest_row['RSI'], 1)),
            step=1.0
        )

        run_sim_btn = st.button("Compute What-If Forecast", use_container_width=True)

    with col_sim_results:
        st.markdown("#### **Simulation Result**")

        adjustments = {
            "btc_price_change_pct": sim_btc_chg,
            "sp500_change_pct": sim_sp_chg,
            "volume_multiplier": sim_vol_mult,
            "rsi_override": sim_rsi
        }

        sim_res = predict_what_if(df_feat, adjustments)

        base_pred = pred_info['predicted_price']
        sim_pred = sim_res['predicted_price']
        delta_sim = sim_pred - base_pred
        delta_sim_pct = (delta_sim / base_pred) * 100.0

        st.markdown(f"""
        <div class='hero-card' style='margin-top: 10px;'>
            <span class='badge-orange'>WHAT-IF SIMULATION</span>
            <div style='margin-top: 14px;'>
                <span style='color: {THEME_COLORS["text_secondary"]}; font-size: 13px;'>Hypothetical Input Price</span><br>
                <span style='font-size: 22px; font-weight: 700;'>${sim_res["hypothetical_current_price"]:,.2f}</span>
                <span style='font-size: 13px; color: {"#00C853" if sim_btc_chg >= 0 else "#FF3B30"}'>({"+" if sim_btc_chg >= 0 else ""}{sim_btc_chg:.1f}%)</span>
            </div>
            <div style='margin-top: 14px;'>
                <span style='color: {THEME_COLORS["text_secondary"]}; font-size: 13px;'>Simulated Next-Day Forecast</span><br>
                <span style='font-size: 30px; font-weight: 800; color: {THEME_COLORS["bitcoin_orange"]};'>${sim_pred:,.2f}</span>
            </div>
            <div style='margin-top: 12px;'>
                <span style='color: {THEME_COLORS["text_secondary"]}; font-size: 13px;'>Divergence from Base Forecast:</span><br>
                <span class='{"badge-green" if delta_sim >= 0 else "badge-red"}' style='font-size: 14px;'>
                    {"+" if delta_sim >= 0 else ""}${delta_sim:,.2f} ({"+" if delta_sim_pct >= 0 else ""}{delta_sim_pct:.2f}%)
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# TAB 5: MODEL ARCHITECTURE & RESEARCH SUMMARY
# -----------------------------------------------------------------------------
with tab_research:
    st.markdown("### **Deep Learning Architecture & Experiment Journey**")
    st.markdown(
        "<p style='color:#8B949E; font-size:14px; margin-top:-8px;'>"
        "Comprehensive documentation of model engineering, feature selection, and comparative benchmarks."
        "</p>",
        unsafe_allow_html=True
    )

    st.markdown("#### **Final Model Selection: Standalone LSTM**")
    st.markdown("""
    During research across 17 iterative experiments, the **Standalone Deep LSTM** was compared against a hybrid **LSTM + XGBoost** model. 
    The Standalone LSTM decisively outperformed the hybrid approach by capturing long-term temporal dependencies without introducing gradient-boosting variance:
    """)

    # Model comparison table
    comp_data = {
        "Metric": ["Mean Absolute Percentage Error (MAPE)", "Root Mean Squared Error (RMSE)", "Coefficient of Determination (R²)", "Selected for Production"],
        "Standalone LSTM (Final Model)": ["4.35% (or 3.20% on test run)", "$5,537.18 (or $3,812.59)", "0.8889 (or 0.9539)", "Yes (Final Deployed Model)"],
        "LSTM + XGBoost Hybrid": ["7.19%", "$10,174.23", "0.6718", "No (Inferior generalization)"]
    }
    st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### **The 17 Input Features (In Exact Order)**")

    feat_col1, feat_col2 = st.columns(2)
    with feat_col1:
        st.markdown("""
        **Price & Momentum Features:**
        1. `Open` — Daily opening price
        2. `High` — Daily high price
        3. `Low` — Daily low price
        4. `Volume` — 24h trading volume
        5. `Daily_Return` — Daily percentage price change
        6. `ROC` — Rate of Change (12-period momentum)
        7. `Close_Lag_7` — 7-day lagged closing price
        8. `SMA_7` — 7-day simple moving average
        9. `Price_SMA7_Ratio` — Price to 7-day SMA ratio
        """)
    with feat_col2:
        st.markdown("""
        **Technical Indicators & Macro Signals:**
        10. `RSI` — Relative Strength Index (14-period)
        11. `MACD` — Moving Average Convergence Divergence line
        12. `MACD_Signal` — MACD 9-period signal line
        13. `BB_Middle` — Bollinger Bands 20-period middle line
        14. `BB_Upper` — Bollinger Bands upper volatility band
        15. `BB_Lower` — Bollinger Bands lower volatility band
        16. `BB_Width` — Bollinger Bands bandwidth
        17. `SP500` — S&P 500 Index closing value
        """)

    st.markdown("---")
    st.markdown("#### **Experiment Reference**")
    st.info("Complete step-by-step experimentation and ablation history is preserved in `results/Experiments and Results.pdf`.")


# -----------------------------------------------------------------------------
# DISCLAIMER & AUTHOR FOOTER
# -----------------------------------------------------------------------------
st.markdown(f"""
<div class='disclaimer-box'>
    <b>Financial Disclaimer:</b><br>
    This application provides experimental machine-learning forecasts based on historical market data. 
    Bitcoin prices are highly volatile and affected by events that may not be captured by the model. 
    Predictions are not financial advice and should not be treated as guaranteed future prices.
</div>

<div style='text-align: center; margin-top: 35px; padding-bottom: 20px; color: {THEME_COLORS["text_secondary"]}; font-size: 13px;'>
    Developed by <b>Shivam Ahirwar</b> | 
    <a href="https://www.linkedin.com/in/shivam-ahirwar-425293311/" target="_blank" style="color: {THEME_COLORS['bitcoin_orange']}; text-decoration: none; font-weight: 600;">
        LinkedIn Profile
    </a> | Bitcoin Price Prediction System v1.0
</div>
""", unsafe_allow_html=True)
