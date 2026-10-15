import pandas as pd
import numpy as np
from pathlib import Path

# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "processed" / "daily_weather.csv"

OUTPUT_FILE = BASE_DIR / "processed" / "rainfall_features.csv"


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date").reset_index(drop=True)


print("Original dataset:")
print(df.head())


# --------------------------------------------------
# RAINFALL LAG FEATURES
# --------------------------------------------------

df["rain_lag_1"] = df["rainfall"].shift(1)

df["rain_lag_2"] = df["rainfall"].shift(2)

df["rain_lag_3"] = df["rainfall"].shift(3)

df["rain_lag_7"] = df["rainfall"].shift(7)


# --------------------------------------------------
# ROLLING RAINFALL
# --------------------------------------------------

df["rain_3day"] = (
    df["rainfall"]
    .shift(1)
    .rolling(3)
    .sum()
)

df["rain_7day"] = (
    df["rainfall"]
    .shift(1)
    .rolling(7)
    .sum()
)


# --------------------------------------------------
# WEATHER CHANGE FEATURES
# --------------------------------------------------

df["pressure_change"] = (
    df["pressure"] - df["pressure"].shift(1)
)

df["temperature_change"] = (
    df["temperature"] - df["temperature"].shift(1)
)

df["dewpoint_change"] = (
    df["dew_point"] - df["dew_point"].shift(1)
)


# --------------------------------------------------
# CALENDAR FEATURES
# --------------------------------------------------

df["month"] = df["date"].dt.month

df["day_of_year"] = df["date"].dt.dayofyear

df["day_of_week"] = df["date"].dt.dayofweek


# --------------------------------------------------
# TARGET
# --------------------------------------------------

# Tomorrow's rainfall

df["target_rainfall"] = df["rainfall"].shift(-1)


# --------------------------------------------------
# REMOVE ROWS WITH MISSING VALUES
# --------------------------------------------------

df = df.dropna().reset_index(drop=True)


# --------------------------------------------------
# SAVE
# --------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# INFORMATION
# --------------------------------------------------

print("\n" + "=" * 60)

print("FEATURE ENGINEERING COMPLETE")

print("=" * 60)

print("\nRows:", len(df))

print("\nColumns:")
print(df.columns.tolist())

print("\nSample:")
print(df.head())

print("\nSaved to:")
print(OUTPUT_FILE)