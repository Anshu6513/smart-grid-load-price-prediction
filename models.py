"""
models.py
─────────
Two models:
  1. CNN-LSTM  — electricity consumption forecasting (time-series)
  2. XGBoost   — dynamic price prediction (tabular regression)

TensorFlow/Keras is used when available.
Falls back to sklearn MLPRegressor if TF is not installed —
the dashboard still works, just with a lighter model.
"""

import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── TensorFlow availability ───────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        LSTM, Dense, Dropout,
        Conv1D, MaxPooling1D, Input, BatchNormalization,
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    from sklearn.neural_network import MLPRegressor


# ════════════════════════════════════════════════════════════════════
# CNN-LSTM MODEL
# ════════════════════════════════════════════════════════════════════

def _build_cnn_lstm(input_shape: tuple):
    """
    Architecture
    ────────────
    Conv1D  → extract local temporal patterns
    LSTM ×2 → learn long-range dependencies
    Dense   → regression output
    """
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(64, kernel_size=3, activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        LSTM(128, return_sequences=True),
        Dropout(0.20),
        LSTM(64),
        Dropout(0.20),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(
        optimizer=Adam(learning_rate=5e-4),
        loss="huber",          # robust to outliers vs plain MSE
        metrics=["mae"],
    )
    return model


def train_lstm_model(X: np.ndarray, y: np.ndarray,
                     epochs: int = 30, batch: int = 128):
    """
    Train the consumption model.

    Returns
    -------
    model         : trained Keras / sklearn model
    history_dict  : dict with loss curves (or empty dict for sklearn)
    """
    split       = int(0.85 * len(X))
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    if TF_AVAILABLE:
        model = _build_cnn_lstm((X.shape[1], X.shape[2]))
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=6,
                          restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                              patience=3, verbose=0),
        ]
        hist = model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch,
            callbacks=callbacks,
            verbose=0,
        )
        return model, hist.history

    else:
        # Flatten sequences for MLP
        Xf_tr  = X_tr.reshape(len(X_tr),  -1)
        Xf_val = X_val.reshape(len(X_val), -1)
        model  = MLPRegressor(
            hidden_layer_sizes=(256, 128, 64),
            max_iter=300, random_state=42, early_stopping=True,
        )
        model.fit(Xf_tr, y_tr.ravel())
        val_mse = mean_squared_error(y_val.ravel(), model.predict(Xf_val))
        return model, {"val_loss": [val_mse]}


def _predict_raw(model, X: np.ndarray, target_scaler) -> np.ndarray:
    """Run inference and inverse-transform back to original kWh scale."""
    if model is None:
        # Fallback: return mean of the last row's consumption feature (index 0)
        # X shape: (n, seq_len, n_features) — feature 0 is consumption_kwh (scaled)
        last_vals = X[:, -1, 0:1]                        # (n, 1)
        return target_scaler.inverse_transform(last_vals)[:, 0]

    if TF_AVAILABLE and hasattr(model, "predict"):
        scaled = model.predict(X, verbose=0)              # (n, 1)
    else:
        Xf     = X.reshape(len(X), -1)
        scaled = model.predict(Xf).reshape(-1, 1)
    return target_scaler.inverse_transform(scaled)[:, 0]


def forecast_consumption_24h(model, df, closest_idx: int,
                              feature_scaler, target_scaler,
                              lstm_features: list,
                              seq_len: int = 24) -> np.ndarray:
    """
    Predict consumption for the next 24 hours using a
    direct multi-step strategy (avoids recursive compounding errors).

    For each step i ∈ [0, 23]:
      - Build a sequence of the last seq_len rows ending at closest_idx + i
        (clipped to available data, padded at the front if needed)
      - Run one model forward pass
    """
    preds = []
    for step in range(24):
        end_idx   = min(closest_idx + step, len(df) - 1)
        start_idx = max(0, end_idx - seq_len)

        rows = df[lstm_features].iloc[start_idx: end_idx].values
        # Front-pad if we are near the beginning of the dataset
        if len(rows) < seq_len:
            pad  = np.tile(rows[0], (seq_len - len(rows), 1))
            rows = np.vstack([pad, rows])

        rows_scaled = feature_scaler.transform(rows)     # (seq_len, n_feat)
        X_in        = rows_scaled[np.newaxis, :, :]      # (1, seq_len, n_feat)

        p = _predict_raw(model, X_in, target_scaler)[0]
        preds.append(max(0.02, float(p)))

    return np.array(preds, dtype=np.float32)


# ════════════════════════════════════════════════════════════════════
# XGBOOST PRICE MODEL
# ════════════════════════════════════════════════════════════════════

def train_xgb_model(X: np.ndarray, y: np.ndarray):
    """Train XGBoost. Returns (model, metrics_dict)."""
    split       = int(0.85 * len(X))
    X_tr, X_val = X[:split],  X[split:]
    y_tr, y_val = y[:split],  y[split:]

    model = xgb.XGBRegressor(
        n_estimators     = 400,
        max_depth        = 6,
        learning_rate    = 0.04,
        subsample        = 0.80,
        colsample_bytree = 0.80,
        reg_alpha        = 0.10,
        reg_lambda       = 1.00,
        min_child_weight = 3,
        random_state     = 42,
        n_jobs           = -1,
        verbosity        = 0,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    y_hat   = model.predict(X_val)
    metrics = {
        "MAE":    float(mean_absolute_error(y_val, y_hat)),
        "RMSE":   float(np.sqrt(mean_squared_error(y_val, y_hat))),
        "R²":     float(r2_score(y_val, y_hat)),
        "MAPE %": float(np.mean(np.abs((y_val - y_hat) /
                                        (np.abs(y_val) + 1e-6))) * 100),
    }
    return model, metrics


def forecast_price_24h(xgb_model,
                       cons_fc: np.ndarray,
                       df,
                       closest_idx: int,
                       xgb_features: list,
                       sim_dt) -> np.ndarray:
    """Build feature rows for the next 24 hours and predict prices."""
    from datetime import timedelta
    import pandas as pd

    rows = []
    for step in range(24):
        future_dt = sim_dt + timedelta(hours=step)
        h, dow, m = future_dt.hour, future_dt.weekday(), future_dt.month

        # Pull UCI sensor averages from the reference row
        ref_idx = min(closest_idx, len(df) - 1)
        ref     = df.iloc[ref_idx]

        # Lags: use forecast value for steps already predicted, else historical
        def _lag(k):
            if step - k >= 0:
                return float(cons_fc[step - k])
            idx = max(0, closest_idx - k + step)
            return float(df["consumption_kwh"].iloc[idx])

        lag24_idx = max(0, closest_idx - 24 + step)
        lag168_idx = max(0, closest_idx - 168 + step)

        # Rolling mean from mix of forecast and historical
        hist_vals = list(df["consumption_kwh"].iloc[
            max(0, closest_idx - 24): closest_idx].values)
        fc_so_far  = list(cons_fc[:step])
        window24   = (hist_vals + fc_so_far)[-24:]
        rolling24  = float(np.mean(window24)) if window24 else 1.0
        rollstd24  = float(np.std(window24))  if len(window24) > 1 else 0.0

        row = {
            "consumption_kwh":  float(cons_fc[step]),
            "voltage":          float(ref.get("voltage",   230.0)),
            "intensity":        float(ref.get("intensity",  4.0)),
            "reactive_power":   float(ref.get("reactive_power", 0.1)),
            "kitchen_kwh":      float(ref.get("kitchen_kwh", 0.01)),
            "laundry_kwh":      float(ref.get("laundry_kwh", 0.01)),
            "hvac_kwh":         float(ref.get("hvac_kwh",  0.02)),
            "hour":             h,
            "day_of_week":      dow,
            "month":            m,
            "is_weekend":       1 if dow >= 5 else 0,
            "lag_1":            _lag(1),
            "lag_2":            _lag(2),
            "lag_24":           float(df["consumption_kwh"].iloc[lag24_idx]),
            "lag_168":          float(df["consumption_kwh"].iloc[lag168_idx]),
            "rolling_mean_24":  rolling24,
            "rolling_std_24":   rollstd24,
            "hour_sin":         np.sin(2 * np.pi * h / 24),
            "hour_cos":         np.cos(2 * np.pi * h / 24),
        }
        rows.append(row)

    future_df = pd.DataFrame(rows)
    avail     = [f for f in xgb_features if f in future_df.columns]
    preds     = xgb_model.predict(future_df[avail].values)
    return np.clip(preds, 3.0, 25.0).astype(np.float32)


# ════════════════════════════════════════════════════════════════════
# Shared evaluation helper
# ════════════════════════════════════════════════════════════════════

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "MAE":    round(float(mean_absolute_error(y_true, y_pred)), 4),
        "RMSE":   round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "R²":     round(float(r2_score(y_true, y_pred)), 4),
        "MAPE %": round(
            float(np.mean(np.abs((y_true - y_pred) /
                                  (np.abs(y_true) + 1e-6))) * 100), 2),
    }
