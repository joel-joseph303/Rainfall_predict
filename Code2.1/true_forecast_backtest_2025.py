import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = BASE_DIR / "processed"
MODEL_DIR = PROCESSED_DIR / "models"

DATA_FILE = PROCESSED_DIR / "daily_weather.csv"

CLASSIFICATION_MODEL_FILE = (
    MODEL_DIR / "best_optimized_logistic_regression.joblib"
)

REGRESSION_MODEL_FILE = (
    MODEL_DIR / "best_optimized_regression_model.joblib"
)

CLASSIFICATION_FEATURE_FILE = (
    MODEL_DIR / "optimized_classification_features.joblib"
)

REGRESSION_FEATURE_FILE = (
    MODEL_DIR / "optimized_regression_features.joblib"
)

THRESHOLD_FILE = (
    MODEL_DIR / "optimized_classification_threshold.txt"
)

RESULT_FILE = (
    PROCESSED_DIR / "true_forecast_backtest_2025.csv"
)


# ============================================================
# LOCATION
# ============================================================

LATITUDE = 10.0
LONGITUDE = 76.125

TIMEZONE = "Asia/Kolkata"

START_DATE = "2025-01-01"
END_DATE = "2025-12-30"

RAIN_THRESHOLD = 1.0


# ============================================================
# OPEN-METEO SETTINGS
# ============================================================

API_URL = (
    "https://previous-runs-api.open-meteo.com/v1/forecast"
)

# ECMWF IFS
OPEN_METEO_MODEL = "ecmwf_ifs025"


# ============================================================
# LOAD MODELS
# ============================================================

print("=" * 70)
print("TRUE 24-HOUR FORECAST BACKTEST")
print("=" * 70)

print("\nLoading models...")

classification_model = joblib.load(
    CLASSIFICATION_MODEL_FILE
)

regression_model = joblib.load(
    REGRESSION_MODEL_FILE
)

classification_features = joblib.load(
    CLASSIFICATION_FEATURE_FILE
)

regression_features = joblib.load(
    REGRESSION_FEATURE_FILE
)

with open(THRESHOLD_FILE, "r") as f:
    CLASSIFICATION_THRESHOLD = float(
        f.read().strip()
    )


print("Classification model loaded.")
print("Regression model loaded.")

print(
    f"Classification threshold: "
    f"{CLASSIFICATION_THRESHOLD:.2f}"
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
# VERIFY EXACT FEATURE COUNT
# ============================================================

if len(classification_features) != 55:
    raise ValueError(
        "Classification feature file does not contain 55 features."
    )

if len(regression_features) != 55:
    raise ValueError(
        "Regression feature file does not contain 55 features."
    )


# ============================================================
# LOAD ERA5 HISTORICAL DATA
# ============================================================

print("\nLoading historical ERA5 data...")

historical = pd.read_csv(DATA_FILE)

historical["date"] = pd.to_datetime(
    historical["date"]
)

historical = historical.sort_values(
    "date"
).reset_index(drop=True)


print(
    f"Historical rows: {len(historical)}"
)

print(
    f"Date range: "
    f"{historical['date'].min().date()} "
    f"to "
    f"{historical['date'].max().date()}"
)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "date",
    "temperature",
    "dew_point",
    "pressure",
    "u10",
    "v10",
    "wind_speed",
    "rainfall",
]

missing = [
    col
    for col in required_columns
    if col not in historical.columns
]

if missing:

    raise ValueError(
        "Missing columns from daily_weather.csv:\n"
        + "\n".join(missing)
    )


# ============================================================
# GET OPEN-METEO 24-HOUR-OLD FORECAST
# ============================================================

def get_previous_day1_forecast():

    print("\nRequesting Open-Meteo archived forecasts...")

    params = {

        "latitude": LATITUDE,

        "longitude": LONGITUDE,

        "start_date": START_DATE,

        "end_date": END_DATE,

        "hourly": ",".join([
            "temperature_2m_previous_day1",
            "dew_point_2m_previous_day1",
            "surface_pressure_previous_day1",
            "wind_speed_10m_previous_day1",
            "wind_direction_10m_previous_day1",
        ]),

        "timezone": TIMEZONE,

        "wind_speed_unit": "ms",

        "models": OPEN_METEO_MODEL,
    }

    print(
        f"Location: "
        f"{LATITUDE}, {LONGITUDE}"
    )

    print(
        f"Period: "
        f"{START_DATE} to {END_DATE}"
    )

    print(
        "Forecast lead: 24 hours"
    )

    response = requests.get(
        API_URL,
        params=params,
        timeout=120
    )

    response.raise_for_status()

    result = response.json()

    if "hourly" not in result:

        raise ValueError(
            "Open-Meteo response does not contain hourly data."
        )

    hourly = pd.DataFrame(
        result["hourly"]
    )

    hourly["time"] = pd.to_datetime(
        hourly["time"]
    )

    return hourly


forecast_hourly = get_previous_day1_forecast()

print(
    f"Hourly forecast rows: "
    f"{len(forecast_hourly)}"
)


# ============================================================
# RENAME FORECAST VARIABLES
# ============================================================

forecast_hourly = forecast_hourly.rename(
    columns={
        "temperature_2m_previous_day1":
            "temperature",

        "dew_point_2m_previous_day1":
            "dew_point",

        "surface_pressure_previous_day1":
            "pressure",

        "wind_speed_10m_previous_day1":
            "wind_speed",

        "wind_direction_10m_previous_day1":
            "wind_direction",
    }
)


# ============================================================
# CREATE U10 / V10 FROM FORECAST WIND
# ============================================================

# Open-Meteo wind direction is meteorological direction:
# direction FROM which the wind is blowing.
#
# Therefore:
#
# u = -speed * sin(direction)
# v = -speed * cos(direction)

direction_rad = np.deg2rad(
    forecast_hourly["wind_direction"]
)

forecast_hourly["u10"] = (
    -forecast_hourly["wind_speed"]
    *
    np.sin(direction_rad)
)

forecast_hourly["v10"] = (
    -forecast_hourly["wind_speed"]
    *
    np.cos(direction_rad)
)


# ============================================================
# CONVERT HOURLY FORECAST TO DAILY FORECAST
# ============================================================

forecast_hourly["date"] = (
    forecast_hourly["time"]
    .dt.normalize()
)


daily_forecast = (
    forecast_hourly
    .groupby("date")
    .agg({

        "temperature": "mean",

        "dew_point": "mean",

        "pressure": "mean",

        "u10": "mean",

        "v10": "mean",

        "wind_speed": "mean",

    })
    .reset_index()
)


print(
    f"Daily forecast rows: "
    f"{len(daily_forecast)}"
)

print(
    "Daily forecast variables created."
)


# ============================================================
# PREPARE HISTORICAL DATA FOR FEATURE ENGINEERING
# ============================================================

df = historical.copy()

df["date"] = pd.to_datetime(
    df["date"]
)

df = df.sort_values(
    "date"
).reset_index(drop=True)


# ============================================================
# CREATE TRUE FORECAST FEATURE DATASET
# ============================================================

print("\nBuilding forecast-time feature dataset...")


# ------------------------------------------------------------
# IMPORTANT:
#
# For each target date:
#
#   Weather variables:
#       Open-Meteo 24-hour-old forecast
#
#   Rainfall history:
#       Actual rainfall available BEFORE target date
#
#   Target rainfall:
#       Actual ERA5 rainfall
#
# ------------------------------------------------------------


# Historical rainfall remains the source of rainfall history.

rain_history = df[
    ["date", "rainfall"]
].copy()


# Forecast weather replaces the target day's
# observed weather variables.

forecast_weather = daily_forecast.copy()


# Actual target rainfall is retained separately.

actual_targets = df[
    ["date", "rainfall"]
].copy()

actual_targets = actual_targets.rename(
    columns={
        "rainfall": "actual_rainfall"
    }
)


# Merge forecast weather with actual target.

backtest = pd.merge(
    forecast_weather,
    actual_targets,
    on="date",
    how="inner"
)


# ============================================================
# CREATE FEATURES ONE TARGET DATE AT A TIME
# ============================================================

def build_features_for_date(
    target_date,
    historical_data,
    forecast_row
):

    hist = historical_data[
        historical_data["date"] < target_date
    ].copy()

    if len(hist) < 15:
        return None


    # --------------------------------------------------------
    # Rainfall history
    # --------------------------------------------------------

    rain = hist["rainfall"]


    row = {}


    # --------------------------------------------------------
    # Forecast weather variables
    # --------------------------------------------------------

    row["temperature"] = (
        forecast_row["temperature"]
    )

    row["dew_point"] = (
        forecast_row["dew_point"]
    )

    row["pressure"] = (
        forecast_row["pressure"]
    )

    row["u10"] = (
        forecast_row["u10"]
    )

    row["v10"] = (
        forecast_row["v10"]
    )

    row["wind_speed"] = (
        forecast_row["wind_speed"]
    )


    # --------------------------------------------------------
    # Rainfall lags
    # --------------------------------------------------------

    for lag in [
        1, 2, 3, 4,
        5, 6, 7, 14
    ]:

        row[f"rain_lag_{lag}"] = (
            rain.iloc[-lag]
        )


    # --------------------------------------------------------
    # Rainfall rolling sums
    # --------------------------------------------------------

    row["rain_3day"] = (
        rain.iloc[-3:].sum()
    )

    row["rain_7day"] = (
        rain.iloc[-7:].sum()
    )

    row["rain_14day"] = (
        rain.iloc[-14:].sum()
    )


    # --------------------------------------------------------
    # Rainfall rolling means
    # --------------------------------------------------------

    row["rain_mean_3day"] = (
        rain.iloc[-3:].mean()
    )

    row["rain_mean_7day"] = (
        rain.iloc[-7:].mean()
    )

    row["rain_mean_14day"] = (
        rain.iloc[-14:].mean()
    )


    # --------------------------------------------------------
    # Rainfall rolling std
    # --------------------------------------------------------

    row["rain_std_7day"] = (
        rain.iloc[-7:].std()
    )

    row["rain_std_14day"] = (
        rain.iloc[-14:].std()
    )


    # --------------------------------------------------------
    # Rainfall changes
    # --------------------------------------------------------

    row["rain_change_1"] = (
        rain.iloc[-1]
        -
        rain.iloc[-2]
    )

    row["rain_change_3"] = (
        rain.iloc[-1]
        -
        rain.iloc[-4]
    )

    row["rain_change_7"] = (
        rain.iloc[-1]
        -
        rain.iloc[-8]
    )

    # Regression aliases
    row["rain_change_1day"] = (
        row["rain_change_1"]
    )

    row["rain_change_3day"] = (
        row["rain_change_3"]
    )

    row["rain_change_7day"] = (
        row["rain_change_7"]
    )


    # Regression rolling aliases
    row["rain_3day_mean"] = (
        row["rain_mean_3day"]
    )

    row["rain_7day_mean"] = (
        row["rain_mean_7day"]
    )

    row["rain_14day_mean"] = (
        row["rain_mean_14day"]
    )

    row["rain_7day_std"] = (
        row["rain_std_7day"]
    )

    row["rain_14day_std"] = (
        row["rain_std_14day"]
    )


    # --------------------------------------------------------
    # Historical weather
    # --------------------------------------------------------

    temperature = hist["temperature"]

    dewpoint = hist["dew_point"]

    pressure = hist["pressure"]

    wind_speed = hist["wind_speed"]


    # --------------------------------------------------------
    # Temperature changes
    # --------------------------------------------------------

    for lag in [1, 3, 7]:

        row[f"temperature_change_{lag}"] = (
            temperature.iloc[-1]
            -
            temperature.iloc[-(lag + 1)]
        )


    # --------------------------------------------------------
    # Dewpoint changes
    # --------------------------------------------------------

    for lag in [1, 3, 7]:

        row[f"dewpoint_change_{lag}"] = (
            dewpoint.iloc[-1]
            -
            dewpoint.iloc[-(lag + 1)]
        )


    # --------------------------------------------------------
    # Dewpoint depression
    # --------------------------------------------------------

    row["dewpoint_depression"] = (
        row["temperature"]
        -
        row["dew_point"]
    )


    # --------------------------------------------------------
    # Pressure changes
    # --------------------------------------------------------

    for lag in [1, 3, 7]:

        row[f"pressure_change_{lag}"] = (
            pressure.iloc[-1]
            -
            pressure.iloc[-(lag + 1)]
        )


    # --------------------------------------------------------
    # Pressure rolling means
    # --------------------------------------------------------

    row["pressure_mean_3day"] = (
        pressure.iloc[-3:].mean()
    )

    row["pressure_mean_7day"] = (
        pressure.iloc[-7:].mean()
    )

    row["pressure_3day_mean"] = (
        row["pressure_mean_3day"]
    )

    row["pressure_7day_mean"] = (
        row["pressure_mean_7day"]
    )


    # --------------------------------------------------------
    # Wind changes
    # --------------------------------------------------------

    for lag in [1, 3, 7]:

        row[f"wind_change_{lag}"] = (
            wind_speed.iloc[-1]
            -
            wind_speed.iloc[-(lag + 1)]
        )


    # --------------------------------------------------------
    # Wind rolling means
    # --------------------------------------------------------

    row["wind_mean_3day"] = (
        wind_speed.iloc[-3:].mean()
    )

    row["wind_mean_7day"] = (
        wind_speed.iloc[-7:].mean()
    )

    row["wind_3day_mean"] = (
        row["wind_mean_3day"]
    )

    row["wind_7day_mean"] = (
        row["wind_mean_7day"]
    )


    # --------------------------------------------------------
    # Calculated forecast wind speed
    # --------------------------------------------------------

    calculated_wind = np.sqrt(
        row["u10"] ** 2
        +
        row["v10"] ** 2
    )

    row["calculated_wind_speed"] = (
        calculated_wind
    )

    row["wind_speed_calculated"] = (
        calculated_wind
    )


    # --------------------------------------------------------
    # Wind direction
    # --------------------------------------------------------

    wind_direction = np.arctan2(
        row["v10"],
        row["u10"]
    )

    row["wind_direction_sin"] = (
        np.sin(wind_direction)
    )

    row["wind_direction_cos"] = (
        np.cos(wind_direction)
    )


    # --------------------------------------------------------
    # Temperature / dewpoint ratio
    # --------------------------------------------------------

    ratio = (
        row["temperature"]
        /
        (
            abs(row["dew_point"])
            + 1e-6
        )
    )

    row["temperature_dewpoint_ratio"] = ratio
    row["temp_dew_ratio"] = ratio


    # --------------------------------------------------------
    # Calendar
    # --------------------------------------------------------

    row["month"] = target_date.month

    row["day_of_year"] = (
        target_date.dayofyear
    )

    row["day_of_week"] = (
        target_date.dayofweek
    )


    # --------------------------------------------------------
    # Cyclical calendar
    # --------------------------------------------------------

    row["month_sin"] = np.sin(
        2 * np.pi
        * row["month"]
        / 12
    )

    row["month_cos"] = np.cos(
        2 * np.pi
        * row["month"]
        / 12
    )

    row["day_of_year_sin"] = np.sin(
        2 * np.pi
        * row["day_of_year"]
        / 365.25
    )

    row["day_of_year_cos"] = np.cos(
        2 * np.pi
        * row["day_of_year"]
        / 365.25
    )

    row["day_of_week_sin"] = np.sin(
        2 * np.pi
        * row["day_of_week"]
        / 7
    )

    row["day_of_week_cos"] = np.cos(
        2 * np.pi
        * row["day_of_week"]
        / 7
    )


    # Regression aliases

    row["day_year_sin"] = (
        row["day_of_year_sin"]
    )

    row["day_year_cos"] = (
        row["day_of_year_cos"]
    )

    row["day_week_sin"] = (
        row["day_of_week_sin"]
    )

    row["day_week_cos"] = (
        row["day_of_week_cos"]
    )


    return row


# ============================================================
# BUILD ALL BACKTEST ROWS
# ============================================================

print(
    "\nConstructing forecast-time features..."
)

feature_rows = []

forecast_lookup = (
    daily_forecast
    .set_index("date")
)


for index, target_date in enumerate(
    forecast_lookup.index
):

    if index % 50 == 0:

        print(
            f"Processing "
            f"{index + 1}/"
            f"{len(forecast_lookup)}"
        )

    if target_date < pd.Timestamp(
        START_DATE
    ):

        continue

    if target_date > pd.Timestamp(
        END_DATE
    ):

        continue

    forecast_row = (
        forecast_lookup.loc[target_date]
    )

    features = build_features_for_date(
        target_date,
        df,
        forecast_row
    )

    if features is None:
        continue

    features["date"] = target_date

    feature_rows.append(features)


feature_df = pd.DataFrame(
    feature_rows
)


# ============================================================
# MERGE ACTUAL RAINFALL
# ============================================================

feature_df = pd.merge(
    feature_df,
    actual_targets,
    on="date",
    how="inner"
)


print(
    f"\nBacktest samples: "
    f"{len(feature_df)}"
)


# ============================================================
# VERIFY FEATURES
# ============================================================

missing_classification = [
    feature
    for feature in classification_features
    if feature not in feature_df.columns
]

missing_regression = [
    feature
    for feature in regression_features
    if feature not in feature_df.columns
]


if missing_classification:

    raise ValueError(
        "Missing classification features:\n"
        +
        "\n".join(
            missing_classification
        )
    )


if missing_regression:

    raise ValueError(
        "Missing regression features:\n"
        +
        "\n".join(
            missing_regression
        )
    )


# ============================================================
# REMOVE INVALID ROWS
# ============================================================

feature_df = feature_df.dropna(
    subset=list(
        set(
            classification_features
            +
            regression_features
        )
    )
).copy()


print(
    f"Usable samples: "
    f"{len(feature_df)}"
)


# ============================================================
# CLASSIFICATION
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION")
print("=" * 70)


X_classification = feature_df[
    classification_features
]

actual_rainfall = (
    feature_df["actual_rainfall"]
    .values
)

actual_class = (
    actual_rainfall
    >= RAIN_THRESHOLD
).astype(int)


rain_probability = (
    classification_model
    .predict_proba(
        X_classification
    )[:, 1]
)


predicted_class = (
    rain_probability
    >= CLASSIFICATION_THRESHOLD
).astype(int)


accuracy = accuracy_score(
    actual_class,
    predicted_class
)

precision = precision_score(
    actual_class,
    predicted_class,
    zero_division=0
)

recall = recall_score(
    actual_class,
    predicted_class,
    zero_division=0
)

f1 = f1_score(
    actual_class,
    predicted_class,
    zero_division=0
)

roc_auc = roc_auc_score(
    actual_class,
    rain_probability
)

cm = confusion_matrix(
    actual_class,
    predicted_class
)


print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)

print(
    f"ROC-AUC   : {roc_auc:.4f}"
)

print("\nConfusion Matrix:")

print(cm)


# ============================================================
# REGRESSION
# ============================================================

print("\n" + "=" * 70)
print("REGRESSION")
print("=" * 70)


X_regression = feature_df[
    regression_features
]


predicted_rainfall = (
    regression_model
    .predict(
        X_regression
    )
)


# Rainfall cannot be negative.

predicted_rainfall = np.maximum(
    predicted_rainfall,
    0
)


mae = mean_absolute_error(
    actual_rainfall,
    predicted_rainfall
)

mse = mean_squared_error(
    actual_rainfall,
    predicted_rainfall
)

rmse = np.sqrt(mse)

r2 = r2_score(
    actual_rainfall,
    predicted_rainfall
)


print(
    f"MAE  : {mae:.4f} mm"
)

print(
    f"MSE  : {mse:.4f} mm²"
)

print(
    f"RMSE : {rmse:.4f} mm"
)

print(
    f"R²   : {r2:.4f}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

results = feature_df[
    ["date", "actual_rainfall"]
].copy()


results["actual_class"] = (
    actual_class
)

results["rain_probability"] = (
    rain_probability
)

results["predicted_class"] = (
    predicted_class
)

results["predicted_rainfall"] = (
    predicted_rainfall
)


results.to_csv(
    RESULT_FILE,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("TRUE 24-HOUR BACKTEST COMPLETE")
print("=" * 70)

print(
    f"Samples               : "
    f"{len(results)}"
)

print(
    f"Forecast lead         : 24 hours"
)

print(
    f"Rain threshold        : "
    f"{RAIN_THRESHOLD:.2f} mm"
)

print(
    f"Classification cutoff : "
    f"{CLASSIFICATION_THRESHOLD:.2f}"
)


print("\nCLASSIFICATION")

print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1        : {f1:.4f}"
)

print(
    f"ROC-AUC   : {roc_auc:.4f}"
)


print("\nREGRESSION")

print(
    f"MAE       : {mae:.4f} mm"
)

print(
    f"MSE       : {mse:.4f} mm²"
)

print(
    f"RMSE      : {rmse:.4f} mm"
)

print(
    f"R²        : {r2:.4f}"
)


print(
    "\nResults saved to:"
)

print(
    RESULT_FILE
)

print(
    "\nExisting model files were NOT modified."
)