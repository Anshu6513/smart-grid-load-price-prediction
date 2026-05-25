"""
decision_engine.py
──────────────────
Smart Grid Decision Engine.

Converts 24-hour consumption + price forecasts into:
  • Colour-coded alerts           (high / medium / low)
  • Actionable recommendations
  • Bill estimates
  • Appliance cost calculation
"""

from dataclasses import dataclass
from typing import List
import numpy as np

# ── Thresholds ────────────────────────────────────────────────────
PEAK_KWH        = 1.20    # kWh/h — UCI household scale (smaller than industrial)
HIGH_PRICE      = 9.50    # ₹/unit
CHEAP_PRICE     = 5.80    # ₹/unit
HIGH_VOLT       = 242.0   # V  — above this = voltage swell warning
LOW_VOLT        = 218.0   # V  — below this = voltage sag warning


# ── Data class ────────────────────────────────────────────────────
@dataclass
class Alert:
    level:   str    # "high" | "medium" | "low"
    message: str


# ════════════════════════════════════════════════════════════════════
# ALERTS
# ════════════════════════════════════════════════════════════════════

def get_alerts(consumption_fc: list,
               price_fc: list,
               current_hour: int,
               avg_voltage: float = 230.0) -> List[Alert]:
    alerts: List[Alert] = []

    max_cons   = max(consumption_fc)
    max_price  = max(price_fc)
    avg_price  = float(np.mean(price_fc))
    peak_count = sum(1 for c in consumption_fc if c > PEAK_KWH)

    # ── High load ─────────────────────────────────────────────────
    if max_cons > PEAK_KWH:
        idx = consumption_fc.index(max_cons)
        alerts.append(Alert(
            "high",
            f"⚡ Peak load forecast: {max_cons:.3f} kWh at "
            f"{(current_hour + idx) % 24:02d}:00 "
            f"(threshold {PEAK_KWH} kWh)"
        ))

    # ── High price ────────────────────────────────────────────────
    if max_price > HIGH_PRICE:
        idx = price_fc.index(max_price)
        alerts.append(Alert(
            "medium",
            f"💰 High tariff: ₹{max_price:.2f}/unit at "
            f"{(current_hour + idx) % 24:02d}:00"
        ))

    # ── Extended peak period ──────────────────────────────────────
    if peak_count >= 4:
        alerts.append(Alert(
            "medium",
            f"📈 Extended high-demand: {peak_count} peak hours "
            f"forecast in next 24h"
        ))

    # ── Voltage swell/sag ─────────────────────────────────────────
    if avg_voltage > HIGH_VOLT:
        alerts.append(Alert(
            "medium",
            f"⚠️  Voltage swell detected: {avg_voltage:.1f} V "
            f"(normal < {HIGH_VOLT} V). Check sensitive appliances."
        ))
    elif avg_voltage < LOW_VOLT:
        alerts.append(Alert(
            "medium",
            f"⚠️  Voltage sag detected: {avg_voltage:.1f} V "
            f"(normal > {LOW_VOLT} V). Risk of under-voltage trips."
        ))

    # ── Off-peak opportunity ──────────────────────────────────────
    if avg_price < CHEAP_PRICE:
        alerts.append(Alert(
            "low",
            f"✅ Low-tariff window: avg ₹{avg_price:.2f}/unit — "
            f"ideal for high-consumption tasks"
        ))

    if not alerts:
        alerts.append(Alert("low",
                            "✅ Grid status normal — no alerts for next 24h"))

    return alerts


# ════════════════════════════════════════════════════════════════════
# RECOMMENDATIONS
# ════════════════════════════════════════════════════════════════════

def get_recommendations(consumption_fc: list,
                        price_fc: list,
                        current_hour: int) -> List[str]:
    recs: List[str] = []

    # Cheapest 6 hours
    indexed   = sorted(enumerate(price_fc), key=lambda x: x[1])
    cheap_idx = sorted([i for i, _ in indexed[:6]])
    cheap_hrs = [(current_hour + i) % 24 for i in cheap_idx[:3]]
    avg_cheap = float(np.mean([price_fc[i] for i in cheap_idx]))
    avg_price = float(np.mean(price_fc))

    if avg_cheap < avg_price * 0.78:
        hstr = ", ".join(f"{h:02d}:00" for h in cheap_hrs)
        recs.append(
            f"🕐 Cheapest hours to run high-load appliances: {hstr} "
            f"(avg ₹{avg_cheap:.2f}/unit)"
        )

    # Peak avoidance
    peak_hrs = [(current_hour + i) % 24
                for i, c in enumerate(consumption_fc) if c > PEAK_KWH]
    if peak_hrs:
        hstr = ", ".join(f"{h:02d}:00" for h in peak_hrs[:3])
        recs.append(f"⚠️  Reduce usage during: {hstr} — peak demand period")

    # Load shift saving
    saving = (avg_price - avg_cheap) * sum(consumption_fc[:6])
    if saving > 0.5:
        recs.append(
            f"💡 Shifting 6h of load to off-peak could save ≈₹{saving:.1f} today"
        )

    # EV / battery charging
    min_price = min(price_fc)
    if min_price < CHEAP_PRICE:
        mi = price_fc.index(min_price)
        recs.append(
            f"🔌 Best EV / storage charging: "
            f"{(current_hour + mi) % 24:02d}:00 at ₹{min_price:.2f}/unit"
        )

    # Pre-conditioning
    first_peak = next(
        (i for i, c in enumerate(consumption_fc) if c > PEAK_KWH), None
    )
    if first_peak and first_peak > 1:
        recs.append(
            f"❄️  Pre-cool/heat {first_peak}h before peak to cut "
            f"HVAC demand during high-tariff period"
        )

    # Kitchen sub-metering tip (based on historical pattern)
    recs.append("🍳 Kitchen peak usually 07–09 h and 19–21 h — "
                "pre-heat oven / kettle in off-peak window if possible")
    recs.append("🌙 Schedule deferred loads (washing, dishwasher) "
                "via smart-plug timers for overnight off-peak hours")

    return recs


# ════════════════════════════════════════════════════════════════════
# BILL ESTIMATOR
# ════════════════════════════════════════════════════════════════════

def estimate_bill(consumption_fc: list, price_fc: list) -> dict:
    hourly_cost = [c * p for c, p in zip(consumption_fc, price_fc)]
    total_kwh   = sum(consumption_fc)
    total_inr   = sum(hourly_cost)
    avg_price   = total_inr / max(total_kwh, 1e-6)

    return {
        "24h_units":        round(total_kwh, 3),
        "24h_cost":         round(total_inr, 2),
        "avg_price":        round(avg_price, 2),
        "monthly_estimate": round(total_inr * 30, 0),
        "peak_hours":       sum(1 for c in consumption_fc if c > PEAK_KWH),
        "cheapest_hour":    int(np.argmin(price_fc)),
        "costliest_hour":   int(np.argmax(price_fc)),
    }


# ════════════════════════════════════════════════════════════════════
# APPLIANCE SIMULATOR
# ════════════════════════════════════════════════════════════════════

APPLIANCES = {
    "AC / Heat Pump (1.5 kW)":  1500,
    "Washing Machine":           500,
    "Tumble Dryer":             2500,
    "Dishwasher":               1200,
    "Water Heater / Boiler":    2000,
    "Induction Hob":            1800,
    "Refrigerator (always on)":  150,
    "Television (LED 55\")":      100,
    "Ceiling Fan":                 75,
    "Microwave Oven":            1000,
    "Desktop Computer":           300,
    "Laptop":                      65,
    "Iron":                      1000,
    "Vacuum Cleaner":            1400,
    "EV Charger (7 kW)":        7000,
}


def appliance_cost(name: str, hours: float, price_per_unit: float) -> dict:
    watts = APPLIANCES.get(name, 500)
    kwh   = watts * hours / 1000
    cost  = kwh * price_per_unit
    return {
        "watts": watts,
        "kwh":   round(kwh, 3),
        "cost":  round(cost, 2),
    }
