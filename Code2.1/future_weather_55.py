# ============================================================
# KERALA RAINFALL PREDICTION SYSTEM
# 55-FEATURE PRODUCTION FORECAST VERSION
#
# Models:
#   Classification : Optimized Logistic Regression
#   Regression     : Optimized Extra Trees
#
# Data:
#   Historical ERA5-based daily_weather.csv
#   Recent weather history from Open-Meteo
#   Forecast weather from ECMWF IFS HRES
#
# IMPORTANT:
#   Existing model files are NEVER modified.
# ============================================================

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import requests


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = BASE_DIR / "processed"
MODEL_DIR = PROCESSED_DIR / "models"

HISTORICAL_FILE = PROCESSED_DIR / "daily_weather.csv"

CLASSIFICATION_MODEL_FILE = (
    MODEL_DIR / "best_optimized_logistic_regression.joblib"
)

CLASSIFICATION_FEATURES_FILE = (
    MODEL_DIR / "optimized_classification_features.joblib"
)

CLASSIFICATION_THRESHOLD_FILE = (
    MODEL_DIR / "optimized_classification_threshold.txt"
)

REGRESSION_MODEL_FILE = (
    MODEL_DIR / "best_optimized_regression_model.joblib"
)

REGRESSION_FEATURES_FILE = (
    MODEL_DIR / "optimized_regression_features.joblib"
)


# ============================================================
# LOCATION
# ============================================================

LATITUDE = 10.0
LONGITUDE = 76.125

TIMEZONE = "Asia/Kolkata"

# Explicit ECMWF model
FORECAST_MODEL = "ecmwf_ifs025"

# Rain classification threshold
RAINFALL_THRESHOLD = 1.0

# Number of recent days requested from Open-Meteo
RECENT_DAYS = 16


# ============================================================
# API
# ============================================================

FORECAST_API = "https://api.open-meteo.com/v1/forecast"


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("KERALA RAINFALL PREDICTION SYSTEM")
print("55-FEATURE PRODUCTION FORECAST VERSION")
print("=" * 80)


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

print("\n" + "=" * 80)
print("CHECKING REQUIRED FILES")
print("=" * 80)


required_files = {
    "Classification model": CLASSIFICATION_MODEL_FILE,
    "Classification features": CLASSIFICATION_FEATURES_FILE,
    "Classification threshold": CLASSIFICATION_THRESHOLD_FILE,
    "Regression model": REGRESSION_MODEL_FILE,
    "Regression features": REGRESSION_FEATURES_FILE,
    "Historical data": HISTORICAL_FILE,
}


for name, path in required_files.items():

    if path.exists():
        print(f"✓ {name}: {path.name}")

    else:
        print(f"✗ MISSING {name}: {path}")
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


# ============================================================
# LOAD MODELS
# ============================================================

print("\n" + "=" * 80)
print("LOADING OPTIMIZED MODELS")
print("=" * 80)


classification_model = joblib.load(
    CLASSIFICATION_MODEL_FILE
)

classification_features = joblib.load(
    CLASSIFICATION_FEATURES_FILE
)

regression_model = joblib.load(
    REGRESSION_MODEL_FILE
)

regression_features = joblib.load(
    REGRESSION_FEATURES_FILE
)


with open(
    CLASSIFICATION_THRESHOLD_FILE,
    "r",
    encoding="utf-8"
) as f:

    classification_threshold = float(
        f.read().strip()
    )


print("Classification model:")
print("  Optimized Logistic Regression")

print("\nRegression model:")
print("  Optimized Extra Trees")

print(
    f"\nClassification threshold: "
    f"{classification_threshold}"
)

print(
    f"Classification features: "
    f"{len(classification_features)}"
)

print(
    f"Regression features: "
    f"{len(regression_features)}"
)


# ============================================================
# LOAD HISTORICAL DATA
# ============================================================

print("\n" + "=" * 80)
print("LOADING HISTORICAL DATA")
print("=" * 80)


df = pd.read_csv(
    HISTORICAL_FILE
)

df["date"] = pd.to_datetime(
    df["date"]
)

df = (
    df.sort_values("date")
      .drop_duplicates("date")
      .reset_index(drop=True)
)


print(f"Rows: {len(df)}")
print(
    f"Period: "
    f"{df['date'].min().date()} "
    f"to "
    f"{df['date'].max().date()}"
)


# ============================================================
# CURRENT DATE
# ============================================================

current_date = pd.Timestamp.now(
    tz=TIMEZONE
).tz_localize(None).normalize()


target_date = (
    current_date + pd.Timedelta(days=1)
)


print(
    f"\nCurrent date: "
    f"{current_date.date()}"
)

print(
    f"Target forecast date: "
    f"{target_date.date()}"
)


# ============================================================
# CHECK HISTORICAL DATA FRESHNESS
# ============================================================

latest_historical_date = df["date"].max()

days_old = (
    current_date - latest_historical_date
).days


print(
    f"\nLatest historical observation: "
    f"{latest_historical_date.date()}"
)

print(
    f"Days since latest historical observation: "
    f"{days_old}"
)


# ============================================================
# DOWNLOAD RECENT WEATHER HISTORY
# ============================================================

print("\n" + "=" * 80)
print("UPDATING RECENT WEATHER HISTORY")
print("=" * 80)

print(
    "Source: Open-Meteo recent weather-history data"
)

print(
    "Important: these recent values are model-based "
    "weather data, not station observations."
)


recent_start = (
    current_date - pd.Timedelta(days=RECENT_DAYS)
)

recent_end = current_date


params_recent = {

    "latitude": LATITUDE,

    "longitude": LONGITUDE,

    "hourly": (
        "temperature_2m,"
        "dew_point_2m,"
        "surface_pressure,"
        "wind_speed_10m,"
        "wind_direction_10m,"
        "precipitation"
    ),

    "past_days": RECENT_DAYS,

    "forecast_days": 1,

    "timezone": TIMEZONE,

    "temperature_unit": "celsius",

    "wind_speed_unit": "ms",

    "precipitation_unit": "mm",

    # Explicit ECMWF model
    "models": FORECAST_MODEL,
}


try:

    response_recent = requests.get(
        FORECAST_API,
        params=params_recent,
        timeout=60
    )

    response_recent.raise_for_status()

    recent_json = response_recent.json()

except Exception as e:

    raise RuntimeError(
        "Failed to retrieve recent weather history "
        f"from Open-Meteo:\n{e}"
    )


recent_hourly = recent_json.get(
    "hourly"
)

if recent_hourly is None:

    raise RuntimeError(
        "Open-Meteo did not return hourly recent data."
    )


recent_df = pd.DataFrame(
    recent_hourly
)


recent_df["time"] = pd.to_datetime(
    recent_df["time"]
)


# ============================================================
# CREATE DAILY RECENT WEATHER
# ============================================================

recent_df["date"] = (
    recent_df["time"]
    .dt.normalize()
)


recent_df["wind_direction_rad"] = np.deg2rad(
    recent_df["wind_direction_10m"]
)


recent_df["u10"] = (
    -recent_df["wind_speed_10m"]
    * np.sin(recent_df["wind_direction_rad"])
)


recent_df["v10"] = (
    -recent_df["wind_speed_10m"]
    * np.cos(recent_df["wind_direction_rad"])
)


recent_daily = (
    recent_df
    .groupby("date")
    .agg(
        temperature=(
            "temperature_2m",
            "mean"
        ),

        dew_point=(
            "dew_point_2m",
            "mean"
        ),

        pressure=(
            "surface_pressure",
            "mean"
        ),

        u10=(
            "u10",
            "mean"
        ),

        v10=(
            "v10",
            "mean"
        ),

        wind_speed=(
            "wind_speed_10m",
            "mean"
        ),

        rainfall=(
            "precipitation",
            "sum"
        ),
    )
    .reset_index()
)


recent_daily["date"] = pd.to_datetime(
    recent_daily["date"]
)


# ============================================================
# UPDATE ONLY THE MISSING RECENT DATES
# ============================================================

missing_recent = recent_daily[
    recent_daily["date"] > latest_historical_date
].copy()


if len(missing_recent) > 0:

    print(
        f"\nRecent dates available: "
        f"{missing_recent['date'].min().date()} "
        f"to "
        f"{missing_recent['date'].max().date()}"
    )

    print(
        f"New recent rows available: "
        f"{len(missing_recent)}"
    )

    # Append recent data
    df = pd.concat(
        [
            df,
            missing_recent
        ],
        ignore_index=True
    )

    df = (
        df.sort_values("date")
          .drop_duplicates(
              subset="date",
              keep="last"
          )
          .reset_index(drop=True)
    )

else:

    print(
        "\nNo newer recent dates were available."
    )


latest_after_update = df["date"].max()

days_old_after_update = (
    current_date - latest_after_update
).days


print(
    f"\nLatest available weather-history date: "
    f"{latest_after_update.date()}"
)

print(
    f"Days behind current date: "
    f"{days_old_after_update}"
)


if days_old_after_update > 1:

    print("\n" + "!" * 80)
    print("WARNING: RECENT WEATHER HISTORY STILL HAS A GAP")
    print("!" * 80)

    print(
        "The model will continue, but the rainfall history "
        "is not completely current."
    )

else:

    print(
        "\n✓ Recent weather history is current enough "
        "for next-day feature construction."
    )


# ============================================================
# REQUEST TARGET-DAY ECMWF FORECAST
# ============================================================

print("\n" + "=" * 80)
print("REQUESTING ECMWF FORECAST")
print("=" * 80)


print(
    f"Latitude : {LATITUDE}"
)

print(
    f"Longitude: {LONGITUDE}"
)

print(
    f"Timezone : {TIMEZONE}"
)

print(
    f"Model    : {FORECAST_MODEL}"
)

print(
    f"Target   : {target_date.date()}"
)


params_forecast = {

    "latitude": LATITUDE,

    "longitude": LONGITUDE,

    "hourly": (
        "temperature_2m,"
        "dew_point_2m,"
        "surface_pressure,"
        "wind_speed_10m,"
        "wind_direction_10m,"
        "precipitation"
    ),

    "forecast_days": 2,

    "timezone": TIMEZONE,

    "temperature_unit": "celsius",

    "wind_speed_unit": "ms",

    "precipitation_unit": "mm",

    # Explicit ECMWF model
    "models": FORECAST_MODEL,
}


try:

    response_forecast = requests.get(
        FORECAST_API,
        params=params_forecast,
        timeout=60
    )

    response_forecast.raise_for_status()

    forecast_json = response_forecast.json()

except Exception as e:

    raise RuntimeError(
        "Failed to retrieve ECMWF forecast:\n"
        f"{e}"
    )


forecast_hourly = forecast_json.get(
    "hourly"
)

if forecast_hourly is None:

    raise RuntimeError(
        "Open-Meteo did not return hourly forecast data."
    )


forecast_df = pd.DataFrame(
    forecast_hourly
)


forecast_df["time"] = pd.to_datetime(
    forecast_df["time"]
)


forecast_df["date"] = (
    forecast_df["time"]
    .dt.normalize()
)


# ============================================================
# TARGET DAY
# ============================================================

target_hours = forecast_df[
    forecast_df["date"] == target_date
].copy()


if len(target_hours) == 0:

    raise RuntimeError(
        "No target-day ECMWF forecast data was returned."
    )


print(
    f"\nTarget-day forecast hours: "
    f"{len(target_hours)}"
)


# ============================================================
# DAILY TARGET WEATHER
# ============================================================

target_hours["wind_direction_rad"] = np.deg2rad(
    target_hours["wind_direction_10m"]
)


target_hours["u10"] = (
    -target_hours["wind_speed_10m"]
    * np.sin(target_hours["wind_direction_rad"])
)


target_hours["v10"] = (
    -target_hours["wind_speed_10m"]
    * np.cos(target_hours["wind_direction_rad"])
)


target_weather = {

    "temperature": target_hours[
        "temperature_2m"
    ].mean(),

    "dew_point": target_hours[
        "dew_point_2m"
    ].mean(),

    "pressure": target_hours[
        "surface_pressure"
    ].mean(),

    "u10": target_hours[
        "u10"
    ].mean(),

    "v10": target_hours[
        "v10"
    ].mean(),

    "wind_speed": target_hours[
        "wind_speed_10m"
    ].mean(),

    "forecast_rainfall": target_hours[
        "precipitation"
    ].sum(),
}


print("\n" + "=" * 80)
print("ECMWF DAILY FORECAST WEATHER")
print("=" * 80)

print(
    f"Temperature: "
    f"{target_weather['temperature']:.2f} °C"
)

print(
    f"Dew point: "
    f"{target_weather['dew_point']:.2f} °C"
)

print(
    f"Pressure: "
    f"{target_weather['pressure']:.2f} hPa"
)

print(
    f"Wind speed: "
    f"{target_weather['wind_speed']:.2f} m/s"
)

print(
    f"ECMWF forecast precipitation: "
    f"{target_weather['forecast_rainfall']:.2f} mm"
)


# ============================================================
# COMBINE HISTORICAL + RECENT DATA
# ============================================================

# Target date itself must NOT be included in historical
# rainfall features.

history = df[
    df["date"] < target_date
].copy()

history = (
    history.sort_values("date")
           .drop_duplicates("date")
           .reset_index(drop=True)
)


if len(history) < 20:

    raise RuntimeError(
        "Not enough historical rows to generate "
        "the 55 engineered features."
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_mean(series, window):

    return (
        series
        .rolling(window)
        .mean()
        .iloc[-1]
    )


def safe_std(series, window):

    return (
        series
        .rolling(window)
        .std()
        .iloc[-1]
    )


def safe_change(series, periods):

    if len(series) <= periods:
        return np.nan

    return (
        series.iloc[-1]
        - series.iloc[-1 - periods]
    )


# ============================================================
# HISTORICAL SERIES
# ============================================================

rain = history["rainfall"]

temperature = history["temperature"]

dew_point = history["dew_point"]

pressure = history["pressure"]

wind_speed = history["wind_speed"]


# ============================================================
# RAINFALL LAGS
# ============================================================

rain_lag_1 = rain.iloc[-1]

rain_lag_2 = rain.iloc[-2]

rain_lag_3 = rain.iloc[-3]

rain_lag_4 = rain.iloc[-4]

rain_lag_5 = rain.iloc[-5]

rain_lag_6 = rain.iloc[-6]

rain_lag_7 = rain.iloc[-7]

rain_lag_14 = rain.iloc[-14]


# ============================================================
# RAINFALL ROLLING
# ============================================================

rain_3day = rain.tail(3).sum()

rain_7day = rain.tail(7).sum()

rain_14day = rain.tail(14).sum()

rain_mean_3day = safe_mean(
    rain,
    3
)

rain_mean_7day = safe_mean(
    rain,
    7
)

rain_mean_14day = safe_mean(
    rain,
    14
)

rain_std_7day = safe_std(
    rain,
    7
)

rain_std_14day = safe_std(
    rain,
    14
)


# ============================================================
# RAINFALL CHANGES
# ============================================================

rain_change_1 = safe_change(
    rain,
    1
)

rain_change_3 = safe_change(
    rain,
    3
)

rain_change_7 = safe_change(
    rain,
    7
)


# ============================================================
# TEMPERATURE CHANGES
# ============================================================

temperature_change_1 = safe_change(
    temperature,
    1
)

temperature_change_3 = safe_change(
    temperature,
    3
)

temperature_change_7 = safe_change(
    temperature,
    7
)


# ============================================================
# DEWPOINT CHANGES
# ============================================================

dewpoint_change_1 = safe_change(
    dew_point,
    1
)

dewpoint_change_3 = safe_change(
    dew_point,
    3
)

dewpoint_change_7 = safe_change(
    dew_point,
    7
)


# ============================================================
# PRESSURE CHANGES
# ============================================================

pressure_change_1 = safe_change(
    pressure,
    1
)

pressure_change_3 = safe_change(
    pressure,
    3
)

pressure_change_7 = safe_change(
    pressure,
    7
)


# ============================================================
# PRESSURE ROLLING
# ============================================================

pressure_mean_3day = safe_mean(
    pressure,
    3
)

pressure_mean_7day = safe_mean(
    pressure,
    7
)


# ============================================================
# WIND CHANGES
# ============================================================

wind_change_1 = safe_change(
    wind_speed,
    1
)

wind_change_3 = safe_change(
    wind_speed,
    3
)

wind_change_7 = safe_change(
    wind_speed,
    7
)


# ============================================================
# WIND ROLLING
# ============================================================

wind_mean_3day = safe_mean(
    wind_speed,
    3
)

wind_mean_7day = safe_mean(
    wind_speed,
    7
)


# ============================================================
# WEATHER DERIVED FEATURES
# ============================================================

dewpoint_depression = (
    target_weather["temperature"]
    - target_weather["dew_point"]
)


wind_speed_calculated = np.sqrt(
    target_weather["u10"] ** 2
    +
    target_weather["v10"] ** 2
)


wind_direction = np.arctan2(
    target_weather["u10"],
    target_weather["v10"]
)


wind_direction_sin = np.sin(
    wind_direction
)

wind_direction_cos = np.cos(
    wind_direction
)


if abs(target_weather["dew_point"]) > 1e-8:

    temp_dew_ratio = (
        target_weather["temperature"]
        /
        target_weather["dew_point"]
    )

else:

    temp_dew_ratio = 0.0


# ============================================================
# CALENDAR FEATURES
# ============================================================

month = target_date.month

day_of_year = target_date.dayofyear

day_of_week = target_date.dayofweek


month_sin = np.sin(
    2 * np.pi * month / 12
)

month_cos = np.cos(
    2 * np.pi * month / 12
)


day_year_sin = np.sin(
    2 * np.pi * day_of_year / 365.25
)

day_year_cos = np.cos(
    2 * np.pi * day_of_year / 365.25
)


day_week_sin = np.sin(
    2 * np.pi * day_of_week / 7
)

day_week_cos = np.cos(
    2 * np.pi * day_of_week / 7
)


# ============================================================
# CLASSIFICATION FEATURES
# EXACT SAVED FEATURE NAMES
# ============================================================

classification_values = {

    "temperature":
        target_weather["temperature"],

    "dew_point":
        target_weather["dew_point"],

    "pressure":
        target_weather["pressure"],

    "u10":
        target_weather["u10"],

    "v10":
        target_weather["v10"],

    "wind_speed":
        target_weather["wind_speed"],

    "rain_lag_1":
        rain_lag_1,

    "rain_lag_2":
        rain_lag_2,

    "rain_lag_3":
        rain_lag_3,

    "rain_lag_4":
        rain_lag_4,

    "rain_lag_5":
        rain_lag_5,

    "rain_lag_6":
        rain_lag_6,

    "rain_lag_7":
        rain_lag_7,

    "rain_lag_14":
        rain_lag_14,

    "rain_3day":
        rain_3day,

    "rain_7day":
        rain_7day,

    "rain_14day":
        rain_14day,

    "rain_mean_3day":
        rain_mean_3day,

    "rain_mean_7day":
        rain_mean_7day,

    "rain_mean_14day":
        rain_mean_14day,

    "rain_std_7day":
        rain_std_7day,

    "rain_std_14day":
        rain_std_14day,

    "rain_change_1":
        rain_change_1,

    "rain_change_3":
        rain_change_3,

    "rain_change_7":
        rain_change_7,

    "temperature_change_1":
        temperature_change_1,

    "temperature_change_3":
        temperature_change_3,

    "temperature_change_7":
        temperature_change_7,

    "dewpoint_change_1":
        dewpoint_change_1,

    "dewpoint_change_3":
        dewpoint_change_3,

    "dewpoint_change_7":
        dewpoint_change_7,

    "dewpoint_depression":
        dewpoint_depression,

    "pressure_change_1":
        pressure_change_1,

    "pressure_change_3":
        pressure_change_3,

    "pressure_change_7":
        pressure_change_7,

    "pressure_mean_3day":
        pressure_mean_3day,

    "pressure_mean_7day":
        pressure_mean_7day,

    "wind_change_1":
        wind_change_1,

    "wind_change_3":
        wind_change_3,

    "wind_change_7":
        wind_change_7,

    "wind_mean_3day":
        wind_mean_3day,

    "wind_mean_7day":
        wind_mean_7day,

    "calculated_wind_speed":
        wind_speed_calculated,

    "wind_direction_sin":
        wind_direction_sin,

    "wind_direction_cos":
        wind_direction_cos,

    "temperature_dewpoint_ratio":
        temp_dew_ratio,

    "month":
        month,

    "day_of_year":
        day_of_year,

    "day_of_week":
        day_of_week,

    "month_sin":
        month_sin,

    "month_cos":
        month_cos,

    "day_of_year_sin":
        day_year_sin,

    "day_of_year_cos":
        day_year_cos,

    "day_of_week_sin":
        day_week_sin,

    "day_of_week_cos":
        day_week_cos,
}


# ============================================================
# REGRESSION FEATURES
# EXACT SAVED FEATURE NAMES
# ============================================================

regression_values = {

    "temperature":
        target_weather["temperature"],

    "dew_point":
        target_weather["dew_point"],

    "pressure":
        target_weather["pressure"],

    "u10":
        target_weather["u10"],

    "v10":
        target_weather["v10"],

    "wind_speed":
        target_weather["wind_speed"],

    "rain_lag_1":
        rain_lag_1,

    "rain_lag_2":
        rain_lag_2,

    "rain_lag_3":
        rain_lag_3,

    "rain_lag_4":
        rain_lag_4,

    "rain_lag_5":
        rain_lag_5,

    "rain_lag_6":
        rain_lag_6,

    "rain_lag_7":
        rain_lag_7,

    "rain_lag_14":
        rain_lag_14,

    "rain_3day":
        rain_3day,

    "rain_7day":
        rain_7day,

    "rain_14day":
        rain_14day,

    "rain_3day_mean":
        rain_mean_3day,

    "rain_7day_mean":
        rain_mean_7day,

    "rain_14day_mean":
        rain_mean_14day,

    "rain_7day_std":
        rain_std_7day,

    "rain_14day_std":
        rain_std_14day,

    "rain_change_1day":
        rain_change_1,

    "rain_change_3day":
        rain_change_3,

    "rain_change_7day":
        rain_change_7,

    "temperature_change_1":
        temperature_change_1,

    "temperature_change_3":
        temperature_change_3,

    "temperature_change_7":
        temperature_change_7,

    "dewpoint_change_1":
        dewpoint_change_1,

    "dewpoint_change_3":
        dewpoint_change_3,

    "dewpoint_change_7":
        dewpoint_change_7,

    "dewpoint_depression":
        dewpoint_depression,

    "pressure_change_1":
        pressure_change_1,

    "pressure_change_3":
        pressure_change_3,

    "pressure_change_7":
        pressure_change_7,

    "pressure_3day_mean":
        pressure_mean_3day,

    "pressure_7day_mean":
        pressure_mean_7day,

    "wind_change_1":
        wind_change_1,

    "wind_change_3":
        wind_change_3,

    "wind_change_7":
        wind_change_7,

    "wind_3day_mean":
        wind_mean_3day,

    "wind_7day_mean":
        wind_mean_7day,

    "wind_speed_calculated":
        wind_speed_calculated,

    "wind_direction_sin":
        wind_direction_sin,

    "wind_direction_cos":
        wind_direction_cos,

    "temp_dew_ratio":
        temp_dew_ratio,

    "month":
        month,

    "day_of_year":
        day_of_year,

    "day_of_week":
        day_of_week,

    "month_sin":
        month_sin,

    "month_cos":
        month_cos,

    "day_year_sin":
        day_year_sin,

    "day_year_cos":
        day_year_cos,

    "day_week_sin":
        day_week_sin,

    "day_week_cos":
        day_week_cos,
}


# ============================================================
# FEATURE DATAFRAMES
# ============================================================

classification_input = pd.DataFrame(
    [classification_values]
)

regression_input = pd.DataFrame(
    [regression_values]
)


# ============================================================
# FEATURE VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("FEATURE VALIDATION")
print("=" * 80)

print(
    f"Classification input: "
    f"{classification_input.shape}"
)

print(
    f"Regression input: "
    f"{regression_input.shape}"
)


# Classification feature check

missing_classification = [
    feature
    for feature in classification_features
    if feature not in classification_input.columns
]


extra_classification = [
    feature
    for feature in classification_input.columns
    if feature not in classification_features
]


if missing_classification:

    raise ValueError(
        "Missing classification features:\n"
        + "\n".join(missing_classification)
    )


if extra_classification:

    print(
        "Warning: extra classification features:"
    )

    print(extra_classification)


# Regression feature check

missing_regression = [
    feature
    for feature in regression_features
    if feature not in regression_input.columns
]


extra_regression = [
    feature
    for feature in regression_input.columns
    if feature not in regression_features
]


if missing_regression:

    raise ValueError(
        "Missing regression features:\n"
        + "\n".join(missing_regression)
    )


if extra_regression:

    print(
        "Warning: extra regression features:"
    )

    print(extra_regression)


# Exact order

classification_input = (
    classification_input[
        classification_features
    ]
)

regression_input = (
    regression_input[
        regression_features
    ]
)


# Missing values

if classification_input.isnull().any().any():

    bad = classification_input.columns[
        classification_input.isnull().any()
    ].tolist()

    raise ValueError(
        "Classification contains NaN values:\n"
        + str(bad)
    )


if regression_input.isnull().any().any():

    bad = regression_input.columns[
        regression_input.isnull().any()
    ].tolist()

    raise ValueError(
        "Regression contains NaN values:\n"
        + str(bad)
    )


# Infinite values

if not np.isfinite(
    classification_input.values
).all():

    raise ValueError(
        "Classification contains invalid numeric values."
    )


if not np.isfinite(
    regression_input.values
).all():

    raise ValueError(
        "Regression contains invalid numeric values."
    )


print(
    f"✓ {len(classification_features)} "
    "classification features generated"
)

print(
    f"✓ {len(regression_features)} "
    "regression features generated"
)

print(
    "✓ Classification feature order verified"
)

print(
    "✓ Regression feature order verified"
)

print(
    "✓ No missing values"
)

print(
    "✓ No invalid numeric values"
)


# ============================================================
# RAIN CLASSIFICATION
# ============================================================

print("\n" + "=" * 80)
print("RAIN CLASSIFICATION")
print("=" * 80)


rain_probability = (
    classification_model
    .predict_proba(
        classification_input
    )[0][1]
)


rain_prediction = (
    rain_probability
    >= classification_threshold
)


print(
    f"Rain probability: "
    f"{rain_probability * 100:.2f}%"
)

print(
    f"Optimized threshold: "
    f"{classification_threshold:.2f}"
)

print(
    "Rain prediction: "
    + (
        "YES RAIN"
        if rain_prediction
        else
        "NO RAIN"
    )
)


# ============================================================
# RAINFALL REGRESSION
# ============================================================

print("\n" + "=" * 80)
print("RAINFALL REGRESSION")
print("=" * 80)


predicted_rainfall = float(
    regression_model.predict(
        regression_input
    )[0]
)


# Prevent physically impossible negative rainfall

predicted_rainfall = max(
    0.0,
    predicted_rainfall
)


print(
    f"Predicted rainfall: "
    f"{predicted_rainfall:.2f} mm"
)


# ============================================================
# PLANNING RISK
# ============================================================

if (
    rain_probability >= 0.80
    or predicted_rainfall >= 5.0
):

    planning_risk = "HIGH"

    recommendation = (
        "High chance of rain. "
        "Outdoor activities should be planned carefully."
    )

elif (
    rain_probability >= 0.50
    or predicted_rainfall >= 1.0
):

    planning_risk = "MEDIUM"

    recommendation = (
        "Moderate chance of rain. "
        "Keep a backup plan for outdoor activities."
    )

else:

    planning_risk = "LOW"

    recommendation = (
        "Weather conditions are generally favorable "
        "for outdoor activities."
    )


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 80)
print("FINAL RAINFALL PREDICTION")
print("=" * 80)

print(
    f"\nDate: {target_date.date()}"
)

print(
    f"Rain Probability: "
    f"{rain_probability * 100:.2f}%"
)

print(
    "Rain Prediction: "
    + (
        "YES RAIN"
        if rain_prediction
        else
        "NO RAIN"
    )
)

print(
    f"Predicted Rainfall: "
    f"{predicted_rainfall:.2f} mm"
)

print(
    f"Planning Risk: "
    f"{planning_risk}"
)

print(
    f"Recommendation: "
    f"{recommendation}"
)

