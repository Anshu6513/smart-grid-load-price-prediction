"""
app.py — Smart Grid AI Dashboard (UCI Dataset)
────────────────────────────────────────────────
Run:  streamlit run app.py

Dataset: UCI Household Electric Power Consumption
         (Dec 2006 – Nov 2010, French household)

5 dashboard tabs:
  Overview       — live gauges, 24 h combined forecast, bill summary
  Consumption    — LSTM forecast vs actual, sub-metering breakdown
  Price Forecast — XGBoost price chart, day×hour heat-map, feature importance
  Alerts & Recs  — decision engine, appliance simulator, optimal hours
  Analytics      — long-term trends, voltage, power factor, distributions
"""

import warnings
warnings.filterwarnings("ignore")
import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

from data_loader    import load_uci_data, CACHE_CSV
from preprocess     import add_features, create_lstm_sequences, prepare_xgb_data, LSTM_FEATURES
from models         import (train_lstm_model, train_xgb_model,
                             forecast_consumption_24h, forecast_price_24h,
                             compute_metrics, _predict_raw, TF_AVAILABLE)
from decision_engine import (get_alerts, get_recommendations, estimate_bill,
                              appliance_cost, APPLIANCES, PEAK_KWH)

# ════════════════════════════════════════════════════════════════════
# Page config
# ════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="⚡ Smart Grid AI — UCI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════════════
# Global CSS
# ════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
.mcard {
    background:linear-gradient(135deg,#1a2340,#243060);
    border:1px solid #3a4a7a; border-radius:12px;
    padding:18px 10px; text-align:center; margin:4px 0;
}
.mval  { font-size:1.8rem; font-weight:700; color:#7ec8e3; }
.mlbl  { font-size:.78rem; color:#9ab; margin-top:4px; letter-spacing:.04em; }
.al-high   { background:#2a1010; border-left:4px solid #e53e3e;
             border-radius:6px; padding:10px 14px; margin:6px 0; color:#feb2b2; }
.al-medium { background:#2a1a08; border-left:4px solid #dd6b20;
             border-radius:6px; padding:10px 14px; margin:6px 0; color:#fbd38d; }
.al-low    { background:#0a2a14; border-left:4px solid #38a169;
             border-radius:6px; padding:10px 14px; margin:6px 0; color:#9ae6b4; }
.rec-row   { background:#161c2e; border:1px solid #2d3748;
             border-radius:6px; padding:10px 14px; margin:5px 0; color:#e2e8f0; }
.info-box  { background:#1a2040; border:1px solid #4a5a8a;
             border-radius:8px; padding:14px; margin:8px 0; }
</style>
""", unsafe_allow_html=True)

DARK = "plotly_dark"


# ════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════
def mcard(val, lbl):
    st.markdown(
        f'<div class="mcard"><div class="mval">{val}</div>'
        f'<div class="mlbl">{lbl}</div></div>',
        unsafe_allow_html=True,
    )

def alert_html(a):
    st.markdown(
        f'<div class="al-{a.level}">{a.message}</div>',
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════
# Cached loaders
# ════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    msgs = []
    def cb(m): msgs.append(m)
    df = load_uci_data(use_cache=True, status_cb=cb)
    df = add_features(df)
    return df


@st.cache_resource(show_spinner=False)
def load_models():
    df = load_data()

    # Sub-sample for speed: use last 2 years (≈17 500 rows)
    # The UCI dataset has 4 years; using full data is fine but slower
    df_train = df.tail(min(17520, len(df))).reset_index(drop=True)

    # LSTM sequences
    X_seq, y_seq, f_scaler, t_scaler = create_lstm_sequences(df_train, seq_len=24)

    # Train LSTM
    lstm_model, lstm_hist = train_lstm_model(X_seq, y_seq, epochs=30, batch=128)

    # XGBoost price
    X_xgb, y_xgb, xgb_names = prepare_xgb_data(df_train)
    xgb_model, xgb_met      = train_xgb_model(X_xgb, y_xgb)

    return dict(
        lstm_model = lstm_model,
        xgb_model  = xgb_model,
        f_scaler   = f_scaler,
        t_scaler   = t_scaler,
        xgb_names  = xgb_names,
        xgb_met    = xgb_met,
        lstm_hist  = lstm_hist,
        X_seq      = X_seq,
        y_seq      = y_seq,
        X_xgb      = X_xgb,
        y_xgb      = y_xgb,
        df_train   = df_train,
    )


# ════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚡ Smart Grid AI")
    st.caption("UCI Household Power Consumption Dataset")
    st.divider()

    # ── Dataset setup notice ──────────────────────────────────────
    if not CACHE_CSV.exists():
        st.warning(
            "**Dataset not found.**\n\n"
            "The app will try to auto-download (~20 MB).\n\n"
            "If that fails, download manually:\n"
            "1. Visit [UCI page](https://archive.ics.uci.edu/dataset/235/)\n"
            "2. Unzip → place `.txt` in `data/`\n"
            "3. Refresh"
        )
    else:
        st.success("✅ Dataset loaded from cache")

    st.markdown("### 🗓️ Simulation Window")
    sim_date = st.date_input(
        "Date",
        value=datetime(2009, 7, 15),
        min_value=datetime(2007, 1, 2),
        max_value=datetime(2010, 11, 25),
    )
    sim_hour = st.slider("Hour", 0, 23, 18)

    st.divider()
    st.markdown("### 📐 Model Summary")
    tf_label = "CNN-LSTM (TensorFlow)" if TF_AVAILABLE else "MLP (sklearn fallback)"
    st.markdown(f"""
| Task | Model |
|---|---|
| Consumption | {tf_label} |
| Price | XGBoost |
| Alerts | Rule engine |
""")

    st.divider()
    if st.button("🔄 Clear cache & retrain", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    st.divider()
    st.markdown("### 📋 UCI Dataset Info")
    st.markdown("""
- **Location**: Sceaux, France  
- **Period**: Dec 2006 – Nov 2010  
- **Granularity**: 1 minute → resampled to 1 hour  
- **Sub-meters**: Kitchen · Laundry · HVAC  
- **Price**: Simulated ToU tariff (₹/kWh)
""")


# ════════════════════════════════════════════════════════════════════
# LOAD DATA + MODELS
# ════════════════════════════════════════════════════════════════════
try:
    with st.spinner("⚡ Loading UCI dataset (parsing ~2 M rows on first run)…"):
        df = load_data()

    with st.spinner("🤖 Training models — cached after first run…"):
        M = load_models()

except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# Unpack models
lstm_model = M["lstm_model"]
xgb_model  = M["xgb_model"]
f_scaler   = M["f_scaler"]
t_scaler   = M["t_scaler"]
xgb_names  = M["xgb_names"]
xgb_met    = M["xgb_met"]
X_seq      = M["X_seq"]
y_seq      = M["y_seq"]
df_train   = M["df_train"]

# ── Find simulation index ─────────────────────────────────────────
df["datetime"] = pd.to_datetime(df["datetime"])
sim_dt   = datetime.combine(sim_date, datetime.min.time()) + timedelta(hours=sim_hour)
ci       = int((df["datetime"] - sim_dt).abs().idxmin())
ci       = max(200, min(ci, len(df) - 30))

# ── 24-h forecasts ────────────────────────────────────────────────
cons_fc  = forecast_consumption_24h(lstm_model, df, ci, f_scaler,
                                     t_scaler, LSTM_FEATURES, seq_len=24)
cons_fc  = np.clip(cons_fc, 0.02, 8.0).tolist()

price_fc = forecast_price_24h(xgb_model, np.array(cons_fc),
                               df, ci, xgb_names, sim_dt)
price_fc = np.clip(price_fc, 3.0, 22.0).tolist()

# ── Decision engine ───────────────────────────────────────────────
avg_volt   = float(df["voltage"].iloc[ci]) if "voltage" in df.columns else 230.0
alerts     = get_alerts(cons_fc, price_fc, sim_hour, avg_volt)
recs       = get_recommendations(cons_fc, price_fc, sim_hour)
bill       = estimate_bill(cons_fc, price_fc)

# ── Current readings ──────────────────────────────────────────────
cur_row   = df.iloc[ci]
cur_cons  = float(cur_row["consumption_kwh"])
cur_price = float(cur_row["price_per_unit"])
cur_volt  = float(cur_row.get("voltage", 230.0))
cur_int   = float(cur_row.get("intensity", 3.0))
cur_react = float(cur_row.get("reactive_power", 0.1))
cur_kitch = float(cur_row.get("kitchen_kwh", 0.0))
cur_laund = float(cur_row.get("laundry_kwh", 0.0))
cur_hvac  = float(cur_row.get("hvac_kwh", 0.0))

fc_dts    = [sim_dt + timedelta(hours=i) for i in range(24)]
fc_lbl    = [dt.strftime("%H:00") for dt in fc_dts]
pf        = round(cur_cons / max(np.sqrt(cur_cons**2 + cur_react**2), 1e-6), 3)


# ════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Overview", "📈 Consumption", "💰 Price Forecast",
    "🚨 Alerts & Recs", "📊 Analytics",
])


# ════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(
        f"## ⚡ Smart Grid Overview  ·  "
        f"{sim_date.strftime('%d %b %Y')}  {sim_hour:02d}:00"
    )

    # KPI row
    c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
    with c1: mcard(f"{cur_cons:.3f} kWh",    "Active Load")
    with c2: mcard(f"₹{cur_price:.2f}",      "Price / Unit")
    with c3: mcard(f"{cur_volt:.1f} V",       "Voltage")
    with c4: mcard(f"{cur_int:.2f} A",        "Intensity")
    with c5: mcard(f"{pf:.3f}",               "Power Factor")
    with c6: mcard(f"{bill['24h_units']} kWh","24 h Forecast")
    with c7: mcard(f"₹{bill['monthly_estimate']:.0f}", "Est. Monthly")

    st.divider()

    col_left, col_right = st.columns([3, 1])

    with col_left:
        # Dual-axis 24 h forecast
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fc_lbl, y=cons_fc,
            name="Consumption (kWh)",
            line=dict(color="#63b3ed", width=2.5),
            fill="tozeroy", fillcolor="rgba(99,179,237,0.10)",
        ))
        fig.add_trace(go.Scatter(
            x=fc_lbl, y=price_fc,
            name="Price (₹/unit)",
            line=dict(color="#f6ad55", width=2, dash="dash"),
            yaxis="y2",
        ))
        fig.add_hline(y=PEAK_KWH, line_dash="dot", line_color="#fc8181",
                      annotation_text="Peak threshold", yref="y")
        fig.update_layout(
            title="24-Hour Forecast: Consumption & Dynamic Price",
            xaxis_title="Hour",
            yaxis =dict(title="kWh", side="left",  color="#63b3ed"),
            yaxis2=dict(title="₹/unit", side="right", overlaying="y", color="#f6ad55"),
            template=DARK, height=370,
            margin=dict(t=40, b=40),
            legend=dict(orientation="h", y=1.06),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        # Gauge
        gauge_col = ("#e53e3e" if cur_cons > PEAK_KWH
                     else "#dd6b20" if cur_cons > PEAK_KWH * 0.7
                     else "#38a169")
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=cur_cons,
            delta={"reference": float(df["consumption_kwh"].mean()),
                   "valueformat": ".3f"},
            title={"text": "Current Load<br><sup>kWh</sup>"},
            number={"valueformat": ".3f"},
            gauge={
                "axis":  {"range": [0, 3], "tickcolor": "white"},
                "bar":   {"color": gauge_col},
                "steps": [
                    {"range": [0.0, PEAK_KWH * 0.6], "color": "#1a3328"},
                    {"range": [PEAK_KWH * 0.6, PEAK_KWH], "color": "#3a2a0a"},
                    {"range": [PEAK_KWH, 3.0],            "color": "#3a1010"},
                ],
                "threshold": {"line": {"color":"white","width":2},
                              "thickness": 0.75, "value": PEAK_KWH},
            },
        ))
        fig_g.update_layout(template=DARK, height=230,
                            margin=dict(t=40,b=5,l=5,r=5))
        st.plotly_chart(fig_g, use_container_width=True)

        # Sub-metering mini pie
        sm_vals = [max(0, cur_kitch), max(0, cur_laund), max(0, cur_hvac)]
        other   = max(0, cur_cons - sum(sm_vals))
        labels  = ["Kitchen", "Laundry", "HVAC", "Other"]
        values  = sm_vals + [other]
        if sum(values) > 0:
            fig_pie = go.Figure(go.Pie(
                labels=labels, values=values,
                hole=0.45,
                marker_colors=["#f6ad55","#63b3ed","#fc8181","#68d391"],
            ))
            fig_pie.update_layout(
                title="Current Sub-metering",
                template=DARK, height=230,
                margin=dict(t=40,b=5,l=5,r=5),
                showlegend=True,
                legend=dict(font=dict(size=10)),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    # Top alerts row
    for a in alerts[:3]:
        alert_html(a)

    # Bill summary row
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("24 h Units",        f"{bill['24h_units']} kWh")
    b2.metric("24 h Cost",         f"₹{bill['24h_cost']}")
    b3.metric("Avg Tariff",        f"₹{bill['avg_price']}/unit")
    b4.metric("Peak Hours (24 h)", bill["peak_hours"])


# ════════════════════════════════════════════════════════════════════
# TAB 2 — CONSUMPTION
# ════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 📈 Electricity Consumption Prediction")
    st.caption(f"Model: {'CNN-LSTM (TensorFlow)' if TF_AVAILABLE else 'MLP (sklearn)'}")

    col_l, col_r = st.columns(2)

    with col_l:
        # Actual vs Predicted (last 400 h in training set)
        n_back     = min(400, len(X_seq))
        y_pred_kw  = _predict_raw(lstm_model, X_seq[-n_back:], t_scaler)
        y_true_kw  = t_scaler.inverse_transform(y_seq[-n_back:])[:, 0]
        pred_times = df_train["datetime"].iloc[len(df_train) - n_back:].values

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pred_times, y=y_true_kw,
            name="Actual", line=dict(color="#48bb78", width=1.2),
        ))
        fig.add_trace(go.Scatter(
            x=pred_times, y=y_pred_kw,
            name="LSTM Predicted",
            line=dict(color="#63b3ed", width=1.2, dash="dash"),
        ))
        fig.update_layout(
            title="Actual vs LSTM Predicted (last 400 h — test split)",
            xaxis_title="Time", yaxis_title="kWh",
            template=DARK, height=350, margin=dict(t=40,b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        # 24-h forecast with confidence band
        hist_12 = df.iloc[ci - 12: ci]
        c_up = [v * 1.13 for v in cons_fc]
        c_dn = [v * 0.87 for v in cons_fc]

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=hist_12["datetime"], y=hist_12["consumption_kwh"],
            name="Historical (12 h)", line=dict(color="#48bb78", width=2),
        ))
        fig2.add_trace(go.Scatter(
            x=fc_dts + fc_dts[::-1], y=c_up + c_dn[::-1],
            fill="toself", fillcolor="rgba(99,179,237,0.12)",
            line=dict(color="rgba(0,0,0,0)"), name="±13% band",
        ))
        fig2.add_trace(go.Scatter(
            x=fc_dts, y=cons_fc,
            name="Forecast", line=dict(color="#63b3ed", width=2.5),
            mode="lines+markers", marker=dict(size=5),
        ))
        fig2.add_hline(y=PEAK_KWH, line_dash="dot", line_color="#fc8181",
                       annotation_text="Peak threshold")
        fig2.update_layout(
            title="Next 24 h Forecast (with uncertainty band)",
            xaxis_title="Time", yaxis_title="kWh",
            template=DARK, height=350, margin=dict(t=40,b=40),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Model metrics
    st.markdown("### 🎯 LSTM Metrics (held-out test split)")
    m = compute_metrics(y_true_kw, y_pred_kw)
    mc1,mc2,mc3,mc4 = st.columns(4)
    mc1.metric("MAE",    m["MAE"])
    mc2.metric("RMSE",   m["RMSE"])
    mc3.metric("R²",     m["R²"])
    mc4.metric("MAPE %", m["MAPE %"])

    st.divider()

    # ── Sub-metering time-series ──────────────────────────────────
    st.markdown("### 🔌 Sub-Metering Breakdown (actual data)")
    win = df.iloc[max(0, ci - 168): ci]   # last 7 days

    fig3 = go.Figure()
    for col, color, name in [
        ("kitchen_kwh", "#f6ad55", "Kitchen"),
        ("laundry_kwh", "#63b3ed", "Laundry"),
        ("hvac_kwh",    "#fc8181", "HVAC / Water Heater"),
    ]:
        if col in win.columns:
            fig3.add_trace(go.Scatter(
                x=win["datetime"], y=win[col],
                name=name, line=dict(color=color, width=1.5),
                stackgroup="one",
            ))
    fig3.update_layout(
        title="Stacked Sub-metering — last 7 days (kWh/h)",
        xaxis_title="Time", yaxis_title="kWh",
        template=DARK, height=300, margin=dict(t=40,b=20),
    )
    st.plotly_chart(fig3, use_container_width=True)

    # ── Hourly profile ────────────────────────────────────────────
    st.markdown("### ⏰ Average Consumption by Hour")
    hrly = df.groupby("hour")["consumption_kwh"].mean().reset_index()
    fig4 = px.bar(
        hrly, x="hour", y="consumption_kwh",
        color="consumption_kwh", color_continuous_scale="Blues",
        labels={"consumption_kwh": "Avg kWh", "hour": "Hour of Day"},
        template=DARK,
    )
    fig4.update_layout(height=260, margin=dict(t=20,b=10), showlegend=False)
    st.plotly_chart(fig4, use_container_width=True)


# ════════════════════════════════════════════════════════════════════
# TAB 3 — PRICE FORECAST
# ════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## 💰 Dynamic Price Prediction (XGBoost)")

    col_l, col_r = st.columns(2)

    with col_l:
        hist_24 = df.iloc[ci - 24: ci]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist_24["datetime"], y=hist_24["price_per_unit"],
            name="Historical Price", line=dict(color="#f6ad55", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=fc_dts, y=price_fc,
            name="XGBoost Forecast",
            line=dict(color="#fc8181", width=2.5, dash="dash"),
            mode="lines+markers", marker=dict(size=6),
        ))
        fig.add_hline(y=9.5,  line_dash="dot", line_color="#e53e3e",
                      annotation_text="High tariff ₹9.5")
        fig.add_hline(y=5.8,  line_dash="dot", line_color="#38a169",
                      annotation_text="Off-peak ₹5.8")
        fig.update_layout(
            title="Electricity Price Forecast (₹/kWh)",
            xaxis_title="Time", yaxis_title="₹/unit",
            template=DARK, height=360, margin=dict(t=40,b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        # Price heat-map: day-of-week × hour (from training data)
        pivot = (df_train
                 .pivot_table(values="price_per_unit",
                              index="day_of_week", columns="hour",
                              aggfunc="mean"))
        pivot.index = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        fig2 = px.imshow(
            pivot, color_continuous_scale="RdYlGn_r",
            labels={"color": "₹/unit"},
            title="Average Price Heatmap (Day of Week × Hour)",
            template=DARK,
        )
        fig2.update_layout(height=360, margin=dict(t=40,b=40))
        st.plotly_chart(fig2, use_container_width=True)

    # XGBoost metrics
    st.markdown("### 🎯 XGBoost Metrics")
    xm = xgb_met
    xc1,xc2,xc3,xc4 = st.columns(4)
    xc1.metric("MAE",    f"{xm['MAE']:.4f}")
    xc2.metric("RMSE",   f"{xm['RMSE']:.4f}")
    xc3.metric("R²",     f"{xm['R²']:.4f}")
    xc4.metric("MAPE %", f"{xm['MAPE %']:.2f}")

    # Feature importance
    st.markdown("### 🏆 Top Feature Importances")
    imps = xgb_model.feature_importances_
    n    = min(len(xgb_names), len(imps))
    fi   = (pd.DataFrame({"Feature": xgb_names[:n], "Importance": imps[:n]})
              .sort_values("Importance", ascending=True)
              .tail(12))
    fig3 = px.bar(
        fi, x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale="Blues",
        template=DARK,
    )
    fig3.update_layout(height=340, margin=dict(t=20,b=10), showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

    # Price forecast table
    st.markdown("### 📋 Hourly Price Forecast Table")
    fc_table = pd.DataFrame({
        "Hour":             fc_lbl,
        "Forecast (₹/unit)": [round(p, 2) for p in price_fc],
        "Consumption (kWh)": [round(c, 4) for c in cons_fc],
        "Hourly Cost (₹)":   [round(c*p, 2) for c, p in zip(cons_fc, price_fc)],
        "Tariff Band": [
            "🔴 Peak" if p >= 9.5
            else "🟡 Standard" if p >= 5.8
            else "🟢 Off-peak"
            for p in price_fc
        ],
    })
    st.dataframe(fc_table, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════
# TAB 4 — ALERTS & RECOMMENDATIONS
# ════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## 🚨 Smart Grid Decision Engine")

    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown("### ⚠️ Active Alerts")
        for a in alerts:
            alert_html(a)

        st.divider()
        st.markdown("### 💡 Smart Recommendations")
        for r in recs:
            st.markdown(f'<div class="rec-row">{r}</div>',
                        unsafe_allow_html=True)

    with col_b:
        # Hourly cost bar + load overlay
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=fc_lbl, y=cons_fc,
            name="Consumption (kWh)", marker_color="#63b3ed",
            opacity=0.85,
        ))
        hourly_costs = [c * p for c, p in zip(cons_fc, price_fc)]
        fig.add_trace(go.Scatter(
            x=fc_lbl, y=hourly_costs,
            name="Hourly Cost (₹)",
            line=dict(color="#f6ad55", width=2.5),
            yaxis="y2",
        ))
        fig.update_layout(
            title="Load vs Hourly Cost",
            yaxis =dict(title="kWh",   side="left",  color="#63b3ed"),
            yaxis2=dict(title="₹",     side="right", color="#f6ad55",
                        overlaying="y"),
            template=DARK, height=300,
            margin=dict(t=40,b=40),
            legend=dict(orientation="h", y=1.08),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Optimal hours highlight
        sorted_hours = sorted(range(24), key=lambda i: price_fc[i])
        best3   = sorted_hours[:3]
        worst3  = sorted_hours[-3:]

        c_best, c_worst = st.columns(2)
        with c_best:
            st.markdown("**✅ Cheapest Hours**")
            for i in best3:
                h = (sim_hour + i) % 24
                st.success(f"{h:02d}:00 — ₹{price_fc[i]:.2f}/unit")
        with c_worst:
            st.markdown("**❌ Costliest Hours**")
            for i in worst3:
                h = (sim_hour + i) % 24
                st.error(f"{h:02d}:00 — ₹{price_fc[i]:.2f}/unit")

        st.divider()

        # Bill estimator
        st.markdown("### 🧮 Bill Estimator")
        b1, b2 = st.columns(2)
        b1.metric("24h Consumption",  f"{bill['24h_units']} kWh")
        b1.metric("24h Cost",         f"₹{bill['24h_cost']}")
        b2.metric("Avg Price",        f"₹{bill['avg_price']}/unit")
        b2.metric("Monthly Estimate", f"₹{bill['monthly_estimate']:.0f}")

        st.divider()

        # Appliance simulator
        st.markdown("### 🏠 Appliance Cost Simulator")
        sel_app  = st.selectbox("Select Appliance", list(APPLIANCES.keys()), key="app_sel")
        hrs_used = st.slider("Hours of use", 0.5, 12.0, 3.0, step=0.5, key="hrs_app")
        res      = appliance_cost(sel_app, hrs_used, bill["avg_price"])
        st.info(
            f"**{sel_app}** ({res['watts']} W) × {hrs_used} h "
            f"→ **{res['kwh']} kWh** → **₹{res['cost']}**"
        )
        if bill["cheapest_hour"] is not None:
            ch = (sim_hour + bill["cheapest_hour"]) % 24
            res_cheap = appliance_cost(sel_app, hrs_used, min(price_fc))
            st.success(
                f"🕐 If you run at {ch:02d}:00 (cheapest): "
                f"₹{res_cheap['cost']} vs ₹{res['cost']} at avg price"
            )


# ════════════════════════════════════════════════════════════════════
# TAB 5 — ANALYTICS
# ════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("## 📊 Historical Analytics — UCI Dataset")

    col1, col2 = st.columns(2)

    with col1:
        # Monthly avg consumption
        mly = df.groupby("month")["consumption_kwh"].mean().reset_index()
        mly["month_name"] = mly["month"].map(
            {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
             7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
        )
        fig = px.bar(
            mly, x="month_name", y="consumption_kwh",
            color="consumption_kwh", color_continuous_scale="Blues",
            title="Monthly Average Consumption (kWh/h)",
            labels={"consumption_kwh": "Avg kWh", "month_name": "Month"},
            template=DARK,
        )
        fig.update_layout(height=300, margin=dict(t=40,b=20), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Weekday vs Weekend profile
        df["day_type"] = df["is_weekend"].map({0: "Weekday", 1: "Weekend"})
        hrly_type = df.groupby(["hour","day_type"])["consumption_kwh"].mean().reset_index()
        fig2 = px.line(
            hrly_type, x="hour", y="consumption_kwh", color="day_type",
            color_discrete_map={"Weekday":"#63b3ed","Weekend":"#f6ad55"},
            markers=True,
            title="Avg Hourly Profile: Weekday vs Weekend",
            labels={"consumption_kwh":"Avg kWh","hour":"Hour"},
            template=DARK,
        )
        fig2.update_layout(height=300, margin=dict(t=40,b=20))
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # Voltage trend (rolling daily avg)
        if "voltage" in df.columns:
            vdf = df[["datetime","voltage"]].set_index("datetime")
            vroll = vdf.resample("D").mean().reset_index()
            fig3 = px.line(
                vroll, x="datetime", y="voltage",
                title="Daily Average Voltage (V)",
                labels={"voltage":"V","datetime":"Date"},
                template=DARK, color_discrete_sequence=["#68d391"],
            )
            fig3.add_hline(y=242, line_dash="dot", line_color="#fc8181",
                           annotation_text="Swell 242V")
            fig3.add_hline(y=218, line_dash="dot", line_color="#fc8181",
                           annotation_text="Sag 218V")
            fig3.update_layout(height=300, margin=dict(t=40,b=20))
            st.plotly_chart(fig3, use_container_width=True)

    with col4:
        # Sub-metering breakdown across years
        df["year_str"] = df["year"].astype(str)
        yrly = df.groupby("year")[
            ["kitchen_kwh","laundry_kwh","hvac_kwh"]
        ].sum().reset_index()
        yrly_melt = yrly.melt(id_vars="year",
                              value_vars=["kitchen_kwh","laundry_kwh","hvac_kwh"],
                              var_name="Sub-meter", value_name="Total kWh")
        yrly_melt["Sub-meter"] = yrly_melt["Sub-meter"].map({
            "kitchen_kwh":"Kitchen",
            "laundry_kwh":"Laundry",
            "hvac_kwh":   "HVAC"
        })
        fig4 = px.bar(
            yrly_melt, x="year", y="Total kWh", color="Sub-meter",
            barmode="group",
            color_discrete_map={"Kitchen":"#f6ad55","Laundry":"#63b3ed","HVAC":"#fc8181"},
            title="Annual Sub-metering Totals by Year",
            template=DARK,
        )
        fig4.update_layout(height=300, margin=dict(t=40,b=20))
        st.plotly_chart(fig4, use_container_width=True)

    # ── Scatter: voltage vs consumption ──────────────────────────
    col5, col6 = st.columns(2)

    with col5:
        if "voltage" in df.columns:
            sample = df.sample(min(800, len(df)), random_state=7)
            fig5 = px.scatter(
                sample, x="voltage", y="consumption_kwh",
                color="hour", color_continuous_scale="Plasma",
                opacity=0.6,
                title="Voltage vs Consumption (coloured by hour)",
                labels={"voltage":"V","consumption_kwh":"kWh","hour":"Hour"},
                template=DARK,
            )
            fig5.update_layout(height=300, margin=dict(t=40,b=20))
            st.plotly_chart(fig5, use_container_width=True)

    with col6:
        # Consumption distribution
        fig6 = px.histogram(
            df, x="consumption_kwh", nbins=80,
            color_discrete_sequence=["#63b3ed"],
            title="Consumption Distribution (kWh/h)",
            labels={"consumption_kwh":"kWh"},
            template=DARK,
        )
        fig6.update_layout(height=300, margin=dict(t=40,b=20))
        st.plotly_chart(fig6, use_container_width=True)

    # ── Raw data table ────────────────────────────────────────────
    st.markdown("### 📋 Last 24 h Smart Meter Readings")
    cols_show = ["datetime","consumption_kwh","voltage","intensity",
                 "reactive_power","kitchen_kwh","laundry_kwh",
                 "hvac_kwh","price_per_unit"]
    recent = df.iloc[ci - 24: ci][[c for c in cols_show if c in df.columns]].copy()
    recent["datetime"] = recent["datetime"].dt.strftime("%Y-%m-%d %H:%M")
    st.dataframe(
        recent.sort_values("datetime", ascending=False),
        use_container_width=True, hide_index=True,
    )


# ════════════════════════════════════════════════════════════════════
# Footer
# ════════════════════════════════════════════════════════════════════
st.divider()
st.markdown(
    "<div style='text-align:center;color:#4a5568;font-size:.8rem;'>"
    "⚡ Smart Grid AI  ·  UCI Household Electric Power Consumption Dataset  ·  "
    "CNN-LSTM + XGBoost + Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
