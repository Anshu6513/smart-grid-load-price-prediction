"""
data_loader.py
──────────────
Handles downloading, parsing, and preprocessing the
UCI Household Electric Power Consumption dataset.

Source : https://archive.ics.uci.edu/dataset/235/
         individual+household+electric+power+consumption

Raw data: ~2 million minute-level readings from a single
household in Sceaux, France (Dec 2006 – Nov 2010).

Columns in raw file
───────────────────
  Date, Time
  Global_active_power   (kW)
  Global_reactive_power (kW)
  Voltage               (V)
  Global_intensity      (A)
  Sub_metering_1        (Wh) — kitchen
  Sub_metering_2        (Wh) — laundry room
  Sub_metering_3        (Wh) — water-heater + AC

This module outputs a clean hourly DataFrame ready for ML.
"""

import os
import io
import zipfile
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np
import pandas as pd

# ── File paths ────────────────────────────────────────────────────
DATA_DIR     = Path("data")
RAW_TXT      = DATA_DIR / "household_power_consumption.txt"
RAW_ZIP      = DATA_DIR / "household_power_consumption.zip"
CACHE_CSV    = DATA_DIR / "uci_hourly.csv"

UCI_URLS = [
    ("https://archive.ics.uci.edu/ml/machine-learning-databases"
     "/00235/household_power_consumption.zip"),
    ("https://archive.ics.uci.edu/static/public/235/"
     "individual+household+electric+power+consumption.zip"),
]

RAW_COLS = [
    "Date", "Time",
    "Global_active_power", "Global_reactive_power",
    "Voltage", "Global_intensity",
    "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
]


# ════════════════════════════════════════════════════════════════════
# 1. DOWNLOAD
# ════════════════════════════════════════════════════════════════════

def _download_zip(progress_cb=None) -> bool:
    """Try each UCI mirror; return True on success."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for url in UCI_URLS:
        try:
            def _hook(count, block, total):
                if progress_cb and total > 0:
                    progress_cb(min(count * block / total, 1.0))

            urllib.request.urlretrieve(url, RAW_ZIP, reporthook=_hook)
            return True
        except Exception:
            pass
    return False


def _unzip() -> bool:
    """Extract .txt from the zip into DATA_DIR."""
    try:
        with zipfile.ZipFile(RAW_ZIP, "r") as z:
            for name in z.namelist():
                if name.lower().endswith(".txt"):
                    data = z.read(name)
                    RAW_TXT.write_bytes(data)
                    return True
    except Exception:
        pass
    return False


# ════════════════════════════════════════════════════════════════════
# 2. PARSE MINUTE-LEVEL DATA
# ════════════════════════════════════════════════════════════════════

def _parse_raw(status_cb=None) -> pd.DataFrame:
    """
    Parse the semicolon-separated raw file.
    Missing values are marked '?' — replaced with NaN then forward-filled.
    Returns a clean per-minute DataFrame indexed by datetime.
    """
    if status_cb:
        status_cb("Parsing ~2 million rows (≈15–30 s)…")

    df = pd.read_csv(
        RAW_TXT,
        sep=";",
        na_values=["?"],
        dtype=str,
        engine="python",
        on_bad_lines="skip",
    )
    df.columns = RAW_COLS

    # Parse datetime
    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format="%d/%m/%Y %H:%M:%S",
        errors="coerce",
    )

    for c in RAW_COLS[2:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = (df.drop(columns=["Date", "Time"])
            .dropna(subset=["datetime"])
            .sort_values("datetime")
            .set_index("datetime"))

    # Fill gaps: forward-fill then back-fill any leading NaN
    df = df.ffill().bfill().reset_index()
    return df


# ════════════════════════════════════════════════════════════════════
# 3. RESAMPLE → HOURLY
# ════════════════════════════════════════════════════════════════════

def _simulate_price(hour: int, is_weekend: int,
                    consumption: float, rng_val: float) -> float:
    """
    Simulate dynamic electricity price in ₹/kWh using a
    time-of-use tariff (mirrors Indian grid pricing).

    Off-peak night (00-05):  ₹4.5
    Morning peak  (07-10):   ₹9.0
    Standard day  (10-17):   ₹7.0
    Evening peak  (18-22):   ₹11.5
    Weekend flat:            ₹5.5
    """
    if is_weekend:
        base = 5.5
    elif 18 <= hour <= 22:
        base = 11.5
    elif 7 <= hour <= 10:
        base = 9.0
    elif 0 <= hour <= 5:
        base = 4.5
    else:
        base = 7.0

    price = base + 0.35 * consumption + rng_val
    return round(max(3.0, float(price)), 2)


def _resample_to_hourly(df: pd.DataFrame, status_cb=None) -> pd.DataFrame:
    """
    Aggregate minute-level data to hourly.

    • Global_active_power  → mean kW ≡ kWh consumed that hour
    • Sub_metering_X       → sum of Wh/min × (1 kWh / 1000 Wh)
    """
    if status_cb:
        status_cb("Resampling to hourly…")

    df = df.set_index("datetime")

    hourly = df.resample("h").agg(
        consumption_kwh = ("Global_active_power",   "mean"),
        reactive_power  = ("Global_reactive_power", "mean"),
        voltage         = ("Voltage",               "mean"),
        intensity       = ("Global_intensity",      "mean"),
        kitchen_wh      = ("Sub_metering_1",        "sum"),
        laundry_wh      = ("Sub_metering_2",        "sum"),
        hvac_wh         = ("Sub_metering_3",        "sum"),
    ).dropna(subset=["consumption_kwh"]).reset_index()

    # Wh → kWh
    hourly["kitchen_kwh"] = (hourly.pop("kitchen_wh") / 1000).round(4)
    hourly["laundry_kwh"] = (hourly.pop("laundry_wh") / 1000).round(4)
    hourly["hvac_kwh"]    = (hourly.pop("hvac_wh")    / 1000).round(4)

    # Time features
    hourly["hour"]        = hourly["datetime"].dt.hour
    hourly["day_of_week"] = hourly["datetime"].dt.dayofweek
    hourly["month"]       = hourly["datetime"].dt.month
    hourly["year"]        = hourly["datetime"].dt.year
    hourly["is_weekend"]  = (hourly["day_of_week"] >= 5).astype(int)

    # Simulated price with reproducible noise
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.30, len(hourly))
    hourly["price_per_unit"] = [
        _simulate_price(int(r.hour), int(r.is_weekend),
                        float(r.consumption_kwh), float(noise[i]))
        for i, r in enumerate(hourly.itertuples())
    ]

    # Round continuous columns
    for c in ["consumption_kwh", "reactive_power", "voltage",
              "intensity", "kitchen_kwh", "laundry_kwh", "hvac_kwh",
              "price_per_unit"]:
        hourly[c] = hourly[c].round(4)

    return hourly


# ════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ════════════════════════════════════════════════════════════════════

def load_uci_data(use_cache: bool = True,
                  status_cb=None,
                  progress_cb=None) -> pd.DataFrame:
    """
    Load the UCI dataset as a clean hourly DataFrame.

    Resolution priority:
      1. Cached CSV  →  data/uci_hourly.csv
      2. Raw TXT     →  data/household_power_consumption.txt
      3. Zip file    →  data/household_power_consumption.zip
      4. Auto-download from UCI
      5. Raise FileNotFoundError with manual instructions

    Parameters
    ----------
    use_cache   : load from cache CSV if it exists
    status_cb   : callable(str) — receives status messages
    progress_cb : callable(float 0–1) — download progress
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    def _info(msg):
        if status_cb:
            status_cb(msg)
        else:
            print(msg)

    # ── 1. Cached CSV ────────────────────────────────────────────
    if use_cache and CACHE_CSV.exists():
        _info("Loading cached hourly data…")
        df = pd.read_csv(CACHE_CSV, parse_dates=["datetime"])
        _info(f"Loaded {len(df):,} hourly rows from cache.")
        return df

    # ── 2 / 3. Need raw file ─────────────────────────────────────
    if not RAW_TXT.exists():
        if RAW_ZIP.exists():
            _info("Extracting zip file…")
            if not _unzip():
                raise RuntimeError("Failed to extract zip.")
        else:
            _info("Downloading UCI dataset (~20 MB)…")
            if _download_zip(progress_cb):
                _info("Download complete. Extracting…")
                if not _unzip():
                    raise RuntimeError("Downloaded but failed to extract.")
            else:
                raise FileNotFoundError(
                    "\n\n"
                    "══════════════════════════════════════════════════\n"
                    "  UCI dataset not found. Download it manually:\n\n"
                    "  1. Open this URL in your browser:\n"
                    "     https://archive.ics.uci.edu/ml/machine-learning-\n"
                    "     databases/00235/household_power_consumption.zip\n\n"
                    "  2. Unzip it.\n"
                    "  3. Copy  household_power_consumption.txt\n"
                    "     into the  data/  folder next to app.py.\n"
                    "  4. Run  streamlit run app.py  again.\n"
                    "══════════════════════════════════════════════════\n"
                )

    # ── Parse + resample ─────────────────────────────────────────
    raw    = _parse_raw(status_cb=_info)
    hourly = _resample_to_hourly(raw, status_cb=_info)

    # ── Save cache ───────────────────────────────────────────────
    hourly.to_csv(CACHE_CSV, index=False)
    _info(f"Saved {len(hourly):,} rows → {CACHE_CSV}")

    return hourly


# ════════════════════════════════════════════════════════════════════
# Quick test
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    df = load_uci_data()
    print(f"\nShape  : {df.shape}")
    print(f"Range  : {df['datetime'].min()} → {df['datetime'].max()}")
    print(f"Cols   : {list(df.columns)}")
    print(df[["datetime", "consumption_kwh", "voltage",
               "kitchen_kwh", "price_per_unit"]].tail(5))
