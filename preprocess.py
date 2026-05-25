"""
preprocess.py
─────────────
Feature engineering for the UCI Household Electric Power Consumption dataset.

LSTM input features use the rich UCI columns:
  consumption_kwh, voltage, intensity, reactive_power,
  sub-metering (kitchen/laundry/hvac), plus cyclic time encodings.

XGBoost price features add lag and rolling statistics.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ── Feature lists ─────────────────────────────────────────────────

# Index 0 MUST be consumption_kwh (target) — used in forecast function
LSTM_FEATURES = [
    "consumption_kwh",   # [0] target
    "voltage",
    "intensity",
    "reactive_power",
    "kitchen_kwh",
    "laundry_kwh",
    "hvac_kwh",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
    "day_of_week",
]

XGB_FEATURES = [
    "consumption_kwh",
    "voltage",
    "intensity",
    "reactive_power",
    "kitchen_kwh",
    "laundry_kwh",
    "hvac_kwh",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "lag_1",
    "lag_2",
    "lag_24",
    "lag_168",            # 1 week
    "rolling_mean_24",
    "rolling_std_24",
    "hour_sin",
    "hour_cos",
]


# ════════════════════════════════════════════════════════════════════
# Feature engineering
# ════════════════════════════════════════════════════════════════════

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag features, rolling statistics, and cyclic time encodings.
    Drops any rows with NaN introduced by shifting/rolling.
    """
    df = df.copy()

    # Lag features
    df["lag_1"]  = df["consumption_kwh"].shift(1)
    df["lag_2"]  = df["consumption_kwh"].shift(2)
    df["lag_24"] = df["consumption_kwh"].shift(24) if len(df) > 24 else 0.0

    # 1-week lag — only when the dataset is long enough (full UCI dataset has ~35 000 rows)
    df["lag_168"] = df["consumption_kwh"].shift(168) if len(df) > 200 else df["lag_24"]

    # Rolling statistics (24 h window)
    roll_window = min(24, len(df) - 1)
    df["rolling_mean_24"] = df["consumption_kwh"].rolling(roll_window).mean()
    df["rolling_std_24"]  = df["consumption_kwh"].rolling(roll_window).std()

    # Cyclic encodings (avoids the 23 → 0 and Dec → Jan discontinuities)
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]  / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]  / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df = df.dropna().reset_index(drop=True)
    return df


# ════════════════════════════════════════════════════════════════════
# LSTM sequence builder
# ════════════════════════════════════════════════════════════════════

def create_lstm_sequences(df: pd.DataFrame, seq_len: int = 24):
    """
    Build sliding-window (look-back) sequences for LSTM training.

    Parameters
    ----------
    df      : preprocessed DataFrame (after add_features)
    seq_len : look-back window length (default 24 = 24 hours)

    Returns
    -------
    X              : ndarray  (n_samples, seq_len, n_features)
    y              : ndarray  (n_samples, 1)
    feature_scaler : MinMaxScaler fitted on LSTM_FEATURES
    target_scaler  : MinMaxScaler fitted on consumption_kwh only
    """
    missing = [c for c in LSTM_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing LSTM feature columns: {missing}")

    feature_scaler = MinMaxScaler()
    target_scaler  = MinMaxScaler()

    scaled_X = feature_scaler.fit_transform(df[LSTM_FEATURES].values)
    scaled_y = target_scaler.fit_transform(df[["consumption_kwh"]].values)

    X, y = [], []
    for i in range(seq_len, len(scaled_X)):
        X.append(scaled_X[i - seq_len: i])
        y.append(scaled_y[i])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), \
           feature_scaler, target_scaler


# ════════════════════════════════════════════════════════════════════
# XGBoost data builder
# ════════════════════════════════════════════════════════════════════

def prepare_xgb_data(df: pd.DataFrame):
    """
    Return tabular features + price target for XGBoost.

    Returns
    -------
    X            : ndarray
    y            : ndarray  (price_per_unit)
    feat_names   : list[str]
    """
    available = [c for c in XGB_FEATURES if c in df.columns]
    X = df[available].values.astype(np.float32)
    y = df["price_per_unit"].values.astype(np.float32)
    return X, y, available
