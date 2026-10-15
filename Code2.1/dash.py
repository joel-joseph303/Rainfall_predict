import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Kerala Rainfall Prediction",
    page_icon="🌧️",
    layout="wide",
)


# ============================================================
# LOGGING
# ============================================================

LOG_FILE = Path("rainfall_dashboard.log")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = BASE_DIR / "processed"
MODEL_DIR = PROCESSED_DIR / "models"

HISTORICAL_FILE = (
    PROCESSED_DIR / "daily_weather.csv"
)

PREDICTION_HISTORY_FILE = (
    PROCESSED_DIR / "prediction_history.csv"
)


# ============================================================
# MODEL FILES
# ============================================================

CLASSIFICATION_MODEL_FILE = (
    MODEL_DIR /
    "best_optimized_logistic_regression.joblib"
)

CLASSIFICATION_FEATURES_FILE = (
    MODEL_DIR /
    "optimized_classification_features.joblib"
)

CLASSIFICATION_THRESHOLD_FILE = (
    MODEL_DIR /
    "optimized_classification_threshold.txt"
)

REGRESSION_MODEL_FILE = (
    MODEL_DIR /
    "best_optimized_regression_model.joblib"
)

REGRESSION_FEATURES_FILE = (
    MODEL_DIR /
    "optimized_regression_features.joblib"
)


# ============================================================
# API CONFIGURATION
# ============================================================

FORECAST_API = (
    "https://api.open-meteo.com/v1/forecast"
)

TIMEZONE = "Asia/Kolkata"

FORECAST_MODEL = "ecmwf_ifs025"


# ============================================================
# LOCATION
# ============================================================

LATITUDE = 10.0
LONGITUDE = 76.125


# ============================================================
# RAINFALL SETTINGS
# ============================================================

RAIN_THRESHOLD = 1.0


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    required = {
        "Classification model":
            CLASSIFICATION_MODEL_FILE,

        "Classification features":
            CLASSIFICATION_FEATURES_FILE,

        "Classification threshold":
            CLASSIFICATION_THRESHOLD_FILE,

        "Regression model":
            REGRESSION_MODEL_FILE,

        "Regression features":
            REGRESSION_FEATURES_FILE,
    }

    for name, path in required.items():

        if not path.exists():

            raise FileNotFoundError(
                f"{name} not found:\n{path}"
            )

    classification_model = joblib.load(
        CLASSIFICATION_MODEL_FILE
    )

    classification_features = list(
        joblib.load(
            CLASSIFICATION_FEATURES_FILE
        )
    )

    regression_model = joblib.load(
        REGRESSION_MODEL_FILE
    )

    regression_features = list(
        joblib.load(
            REGRESSION_FEATURES_FILE
        )
    )

    with open(
        CLASSIFICATION_THRESHOLD_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        threshold = float(
            file.read().strip()
        )

    if len(classification_features) != 55:
        raise ValueError(
            f"Expected 55 classification features, "
            f"found {len(classification_features)}."
        )

    if len(regression_features) != 55:
        raise ValueError(
            f"Expected 55 regression features, "
            f"found {len(regression_features)}."
        )

    if not 0 <= threshold <= 1:
        raise ValueError(
            f"Invalid classification threshold: {threshold}"
        )

    return (
        classification_model,
        classification_features,
        regression_model,
        regression_features,
        threshold,
    )


# ============================================================
# LOAD HISTORICAL DATA
# ============================================================

@st.cache_data
def load_historical():

    if not HISTORICAL_FILE.exists():

        raise FileNotFoundError(
            f"Historical data not found:\n"
            f"{HISTORICAL_FILE}"
        )

    df = pd.read_csv(
        HISTORICAL_FILE
    )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["date"]
    )

    required_columns = [
        "temperature",
        "dew_point",
        "pressure",
        "u10",
        "v10",
        "wind_speed",
        "rainfall",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Historical dataset is missing: "
            + ", ".join(missing)
        )

    df = (
        df
        .sort_values("date")
        .drop_duplicates(
            subset=["date"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return df


# ============================================================
# API REQUEST
# ============================================================

def request_api(
    url,
    params,
):

    response = requests.get(
        url,
        params=params,
        timeout=120,
    )

    response.raise_for_status()

    return response.json()


# ============================================================
# WIND CONVERSION
# ============================================================

def convert_wind_to_uv(
    wind_speed,
    wind_direction,
):

    direction_rad = np.deg2rad(
        wind_direction
    )

    u10 = (
        -wind_speed
        * np.sin(direction_rad)
    )

    v10 = (
        -wind_speed
        * np.cos(direction_rad)
    )

    return u10, v10


# ============================================================
# HOURLY → DAILY
# ============================================================

def hourly_to_daily(data):

    if "hourly" not in data:

        raise RuntimeError(
            "Open-Meteo response does not contain hourly data."
        )

    hourly = data["hourly"]

    required = [
        "time",
        "temperature_2m",
        "dew_point_2m",
        "surface_pressure",
        "wind_speed_10m",
        "wind_direction_10m",
        "precipitation",
    ]

    missing = [
        column
        for column in required
        if column not in hourly
    ]

    if missing:

        raise RuntimeError(
            "Missing API fields: "
            + ", ".join(missing)
        )

    wind_speed = np.asarray(
        hourly["wind_speed_10m"],
        dtype=float,
    )

    wind_direction = np.asarray(
        hourly["wind_direction_10m"],
        dtype=float,
    )

    u10, v10 = convert_wind_to_uv(
        wind_speed,
        wind_direction,
    )

    frame = pd.DataFrame(
        {
            "time": pd.to_datetime(
                hourly["time"]
            ),

            "temperature": np.asarray(
                hourly["temperature_2m"],
                dtype=float,
            ),

            "dew_point": np.asarray(
                hourly["dew_point_2m"],
                dtype=float,
            ),

            "pressure": np.asarray(
                hourly["surface_pressure"],
                dtype=float,
            ),

            "u10": u10,

            "v10": v10,

            "wind_speed": wind_speed,

            "rainfall": np.asarray(
                hourly["precipitation"],
                dtype=float,
            ),
        }
    )

    frame["date"] = (
        frame["time"]
        .dt.normalize()
    )

    daily = (
        frame
        .groupby("date")
        .agg(
            {
                "temperature": "mean",
                "dew_point": "mean",
                "pressure": "mean",
                "u10": "mean",
                "v10": "mean",
                "wind_speed": "mean",
                "rainfall": "sum",
            }
        )
        .reset_index()
    )

    return daily


# ============================================================
# GET RECENT WEATHER HISTORY
# ============================================================

def get_recent_weather(
    current_date,
):

    params = {

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

        "past_days": 16,

        "forecast_days": 1,

        "timezone": TIMEZONE,

        "temperature_unit": "celsius",

        "wind_speed_unit": "ms",

        "precipitation_unit": "mm",

        "models": FORECAST_MODEL,
    }

    data = request_api(
        FORECAST_API,
        params,
    )

    daily = hourly_to_daily(
        data
    )

    return daily


# ============================================================
# GET ECMWF FORECAST
# ============================================================

def get_future_forecast(
    prediction_date,
):

    params = {

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

        "start_date":
            prediction_date.strftime(
                "%Y-%m-%d"
            ),

        "end_date":
            prediction_date.strftime(
                "%Y-%m-%d"
            ),

        "timezone": TIMEZONE,

        "temperature_unit": "celsius",

        "wind_speed_unit": "ms",

        "precipitation_unit": "mm",

        "models": FORECAST_MODEL,
    }

    data = request_api(
        FORECAST_API,
        params,
    )

    return hourly_to_daily(
        data
    )


# ============================================================
# CREATE 55 FEATURES
# ============================================================

def create_prediction_features(
    prediction_date,
    historical,
    recent_daily,
    forecast_daily,
):

    prediction_date = pd.Timestamp(
        prediction_date
    )

    # --------------------------------------------------------
    # Combine original historical data and recent data
    # --------------------------------------------------------

    combined = pd.concat(
        [
            historical,
            recent_daily,
        ],
        ignore_index=True,
    )

    combined = (
        combined
        .sort_values("date")
        .drop_duplicates(
            subset=["date"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    # Only dates before target are history.
    history = combined[
        combined["date"] < prediction_date
    ].copy()

    history = (
        history
        .sort_values("date")
        .reset_index(drop=True)
    )

    if len(history) < 20:

        raise RuntimeError(
            "Not enough historical data to create "
            "55 features."
        )

    # --------------------------------------------------------
    # Target forecast day
    # --------------------------------------------------------

    target_rows = forecast_daily[
        forecast_daily["date"] ==
        prediction_date
    ]

    if target_rows.empty:

        raise RuntimeError(
            "Forecast data for target date is unavailable."
        )

    target = target_rows.iloc[0]

    # --------------------------------------------------------
    # Series
    # --------------------------------------------------------

    rain = history["rainfall"]

    temperature_history = (
        history["temperature"]
    )

    dewpoint_history = (
        history["dew_point"]
    )

    pressure_history = (
        history["pressure"]
    )

    wind_history = (
        history["wind_speed"]
    )

    # --------------------------------------------------------
    # Rain lags
    # --------------------------------------------------------

    rain_lag_1 = rain.iloc[-1]
    rain_lag_2 = rain.iloc[-2]
    rain_lag_3 = rain.iloc[-3]
    rain_lag_4 = rain.iloc[-4]
    rain_lag_5 = rain.iloc[-5]
    rain_lag_6 = rain.iloc[-6]
    rain_lag_7 = rain.iloc[-7]
    rain_lag_14 = rain.iloc[-14]

    # --------------------------------------------------------
    # Rainfall rolling features
    # --------------------------------------------------------

    rain_3day = rain.tail(3).sum()
    rain_7day = rain.tail(7).sum()
    rain_14day = rain.tail(14).sum()

    rain_mean_3day = (
        rain.tail(3).mean()
    )

    rain_mean_7day = (
        rain.tail(7).mean()
    )

    rain_mean_14day = (
        rain.tail(14).mean()
    )

    rain_std_7day = (
        rain.tail(7).std()
    )

    rain_std_14day = (
        rain.tail(14).std()
    )

    # --------------------------------------------------------
    # Rain changes
    # --------------------------------------------------------

    rain_change_1 = (
        rain.iloc[-1]
        - rain.iloc[-2]
    )

    rain_change_3 = (
        rain.iloc[-1]
        - rain.iloc[-4]
    )

    rain_change_7 = (
        rain.iloc[-1]
        - rain.iloc[-8]
    )

    # --------------------------------------------------------
    # Forecast weather
    # --------------------------------------------------------

    temperature = float(
        target["temperature"]
    )

    dew_point = float(
        target["dew_point"]
    )

    pressure = float(
        target["pressure"]
    )

    u10 = float(
        target["u10"]
    )

    v10 = float(
        target["v10"]
    )

    wind_speed = float(
        target["wind_speed"]
    )

    # --------------------------------------------------------
    # Weather changes
    # --------------------------------------------------------

    temperature_change_1 = (
        temperature
        - temperature_history.iloc[-1]
    )

    temperature_change_3 = (
        temperature
        - temperature_history.iloc[-3]
    )

    temperature_change_7 = (
        temperature
        - temperature_history.iloc[-7]
    )

    dewpoint_change_1 = (
        dew_point
        - dewpoint_history.iloc[-1]
    )

    dewpoint_change_3 = (
        dew_point
        - dewpoint_history.iloc[-3]
    )

    dewpoint_change_7 = (
        dew_point
        - dewpoint_history.iloc[-7]
    )

    # --------------------------------------------------------
    # Dew point depression
    # --------------------------------------------------------

    dewpoint_depression = (
        temperature
        - dew_point
    )

    # --------------------------------------------------------
    # Pressure changes
    # --------------------------------------------------------

    pressure_change_1 = (
        pressure
        - pressure_history.iloc[-1]
    )

    pressure_change_3 = (
        pressure
        - pressure_history.iloc[-3]
    )

    pressure_change_7 = (
        pressure
        - pressure_history.iloc[-7]
    )

    pressure_mean_3day = (
        pressure_history.tail(3).mean()
    )

    pressure_mean_7day = (
        pressure_history.tail(7).mean()
    )

    # --------------------------------------------------------
    # Wind changes
    # --------------------------------------------------------

    wind_change_1 = (
        wind_speed
        - wind_history.iloc[-1]
    )

    wind_change_3 = (
        wind_speed
        - wind_history.iloc[-3]
    )

    wind_change_7 = (
        wind_speed
        - wind_history.iloc[-7]
    )

    wind_mean_3day = (
        wind_history.tail(3).mean()
    )

    wind_mean_7day = (
        wind_history.tail(7).mean()
    )

    # --------------------------------------------------------
    # Calculated wind
    # --------------------------------------------------------

    wind_speed_calculated = np.sqrt(
        u10 ** 2
        +
        v10 ** 2
    )

    wind_direction = np.arctan2(
        u10,
        v10
    )

    wind_direction_sin = np.sin(
        wind_direction
    )

    wind_direction_cos = np.cos(
        wind_direction
    )

    # --------------------------------------------------------
    # Temperature / dew point ratio
    # --------------------------------------------------------

    if abs(dew_point) > 1e-8:

        temp_dew_ratio = (
            temperature
            /
            dew_point
        )

    else:

        temp_dew_ratio = 0.0

    # --------------------------------------------------------
    # Calendar
    # --------------------------------------------------------

    month = prediction_date.month

    day_of_year = (
        prediction_date.dayofyear
    )

    day_of_week = (
        prediction_date.dayofweek
    )

    month_sin = np.sin(
        2 * np.pi * month / 12
    )

    month_cos = np.cos(
        2 * np.pi * month / 12
    )

    day_year_sin = np.sin(
        2 * np.pi *
        day_of_year /
        365.25
    )

    day_year_cos = np.cos(
        2 * np.pi *
        day_of_year /
        365.25
    )

    day_week_sin = np.sin(
        2 * np.pi *
        day_of_week /
        7
    )

    day_week_cos = np.cos(
        2 * np.pi *
        day_of_week /
        7
    )

    # ========================================================
    # CLASSIFICATION FEATURES
    # ========================================================

    classification_values = {

        "temperature": temperature,
        "dew_point": dew_point,
        "pressure": pressure,
        "u10": u10,
        "v10": v10,
        "wind_speed": wind_speed,

        "rain_lag_1": rain_lag_1,
        "rain_lag_2": rain_lag_2,
        "rain_lag_3": rain_lag_3,
        "rain_lag_4": rain_lag_4,
        "rain_lag_5": rain_lag_5,
        "rain_lag_6": rain_lag_6,
        "rain_lag_7": rain_lag_7,
        "rain_lag_14": rain_lag_14,

        "rain_3day": rain_3day,
        "rain_7day": rain_7day,
        "rain_14day": rain_14day,

        "rain_mean_3day": rain_mean_3day,
        "rain_mean_7day": rain_mean_7day,
        "rain_mean_14day": rain_mean_14day,

        "rain_std_7day": rain_std_7day,
        "rain_std_14day": rain_std_14day,

        "rain_change_1": rain_change_1,
        "rain_change_3": rain_change_3,
        "rain_change_7": rain_change_7,

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

        "month": month,
        "day_of_year": day_of_year,
        "day_of_week": day_of_week,

        "month_sin": month_sin,
        "month_cos": month_cos,

        "day_of_year_sin":
            day_year_sin,

        "day_of_year_cos":
            day_year_cos,

        "day_of_week_sin":
            day_week_sin,

        "day_of_week_cos":
            day_week_cos,
    }

    # ========================================================
    # REGRESSION FEATURES
    # ========================================================

    regression_values = {

        "temperature": temperature,
        "dew_point": dew_point,
        "pressure": pressure,
        "u10": u10,
        "v10": v10,
        "wind_speed": wind_speed,

        "rain_lag_1": rain_lag_1,
        "rain_lag_2": rain_lag_2,
        "rain_lag_3": rain_lag_3,
        "rain_lag_4": rain_lag_4,
        "rain_lag_5": rain_lag_5,
        "rain_lag_6": rain_lag_6,
        "rain_lag_7": rain_lag_7,
        "rain_lag_14": rain_lag_14,

        "rain_3day": rain_3day,
        "rain_7day": rain_7day,
        "rain_14day": rain_14day,

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

        "month": month,
        "day_of_year": day_of_year,
        "day_of_week": day_of_week,

        "month_sin": month_sin,
        "month_cos": month_cos,

        "day_year_sin":
            day_year_sin,

        "day_year_cos":
            day_year_cos,

        "day_week_sin":
            day_week_sin,

        "day_week_cos":
            day_week_cos,
    }

    return (
        pd.DataFrame(
            [classification_values]
        ),
        pd.DataFrame(
            [regression_values]
        ),
    )


# ============================================================
# GENERATE PREDICTION
# ============================================================

def generate_prediction(
    prediction_date,
    historical,
    classification_model,
    classification_features,
    regression_model,
    regression_features,
    threshold,
):

    recent_daily = get_recent_weather(
        prediction_date
    )

    forecast_daily = get_future_forecast(
        prediction_date
    )

    (
        classification_input,
        regression_input,
    ) = create_prediction_features(
        prediction_date,
        historical,
        recent_daily,
        forecast_daily,
    )

    # --------------------------------------------------------
    # Validate classification features
    # --------------------------------------------------------

    missing_classification = [
        feature
        for feature in classification_features
        if feature not in classification_input.columns
    ]

    if missing_classification:

        raise RuntimeError(
            "Missing classification features: "
            + ", ".join(
                missing_classification
            )
        )

    classification_input = (
        classification_input[
            classification_features
        ]
    )

    # --------------------------------------------------------
    # Validate regression features
    # --------------------------------------------------------

    missing_regression = [
        feature
        for feature in regression_features
        if feature not in regression_input.columns
    ]

    if missing_regression:

        raise RuntimeError(
            "Missing regression features: "
            + ", ".join(
                missing_regression
            )
        )

    regression_input = (
        regression_input[
            regression_features
        ]
    )

    # --------------------------------------------------------
    # Validate values
    # --------------------------------------------------------

    if (
        classification_input
        .isnull()
        .any()
        .any()
    ):

        raise RuntimeError(
            "Classification features contain NaN."
        )

    if (
        regression_input
        .isnull()
        .any()
        .any()
    ):

        raise RuntimeError(
            "Regression features contain NaN."
        )

    if not np.isfinite(classification_input.to_numpy()).all():
        raise RuntimeError(
            "Classification features contain invalid numeric values."
        )

    if not np.isfinite(regression_input.to_numpy()).all():
        raise RuntimeError(
            "Regression features contain invalid numeric values."
        )

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    rain_probability = float(
        classification_model
        .predict_proba(
            classification_input
        )[0][1]
    )

    rain_prediction = (
        rain_probability >= threshold
    )

    # --------------------------------------------------------
    # Regression
    # --------------------------------------------------------

    predicted_rainfall = float(
        regression_model.predict(
            regression_input
        )[0]
    )

    predicted_rainfall = max(
        0.0,
        predicted_rainfall,
    )

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    probability_percent = (
        rain_probability * 100
    )

    if probability_percent >= 80:

        confidence = "VERY LIKELY"

    elif probability_percent >= 60:

        confidence = "LIKELY"

    elif probability_percent >= 40:

        confidence = "UNCERTAIN"

    else:

        confidence = "LOW"

    # --------------------------------------------------------
    # Rainfall category
    # --------------------------------------------------------

    if predicted_rainfall < 0.5:

        category = "VERY LIGHT / NEGLIGIBLE"

    elif predicted_rainfall < 2:

        category = "LIGHT"

    elif predicted_rainfall < 5:

        category = "MODERATE"

    else:

        category = "HEAVY"

    # --------------------------------------------------------
    # Planning risk
    # --------------------------------------------------------

    if (
        probability_percent >= 80
        or predicted_rainfall >= 5
    ):

        risk = "HIGH"

        recommendation = (
            "High chance of rain. Outdoor activities "
            "should be planned carefully."
        )

    elif (
        probability_percent >= 50
        or predicted_rainfall >= 2
    ):

        risk = "MODERATE"

        recommendation = (
            "Moderate chance of rain. Consider having "
            "an alternative plan for outdoor activities."
        )

    else:

        risk = "LOW"

        recommendation = (
            "Weather conditions are generally favorable "
            "for outdoor activities."
        )

    # --------------------------------------------------------
    # Forecast weather
    # --------------------------------------------------------

    forecast_day = (
        forecast_daily[
            forecast_daily["date"] ==
            pd.Timestamp(prediction_date)
        ]
        .iloc[0]
    )

    return {

        "probability":
            probability_percent,

        "prediction":
            "RAIN"
            if rain_prediction
            else "NO RAIN",

        "rainfall":
            predicted_rainfall,

        "confidence":
            confidence,

        "category":
            category,

        "risk":
            risk,

        "recommendation":
            recommendation,

        "temperature":
            float(
                forecast_day["temperature"]
            ),

        "dew_point":
            float(
                forecast_day["dew_point"]
            ),

        "pressure":
            float(
                forecast_day["pressure"]
            ),

        "wind_speed":
            float(
                forecast_day["wind_speed"]
            ),

        "forecast_rainfall":
            float(
                forecast_day["rainfall"]
            ),

        "rain_lag_1":
            float(
                classification_input.iloc[0][
                    "rain_lag_1"
                ]
            ),

        "rain_lag_2":
            float(
                classification_input.iloc[0][
                    "rain_lag_2"
                ]
            ),

        "rain_lag_3":
            float(
                classification_input.iloc[0][
                    "rain_lag_3"
                ]
            ),

        "rain_lag_7":
            float(
                classification_input.iloc[0][
                    "rain_lag_7"
                ]
            ),

        "rain_3day":
            float(
                classification_input.iloc[0][
                    "rain_3day"
                ]
            ),

        "rain_7day":
            float(
                classification_input.iloc[0][
                    "rain_7day"
                ]
            ),

        "rain_14day":
            float(
                classification_input.iloc[0][
                    "rain_14day"
                ]
            ),

        "pressure_change":
            float(
                classification_input.iloc[0][
                    "pressure_change_1"
                ]
            ),

        "temperature_change":
            float(
                classification_input.iloc[0][
                    "temperature_change_1"
                ]
            ),

        "dewpoint_change":
            float(
                classification_input.iloc[0][
                    "dewpoint_change_1"
                ]
            ),
    }


# ============================================================
# SAVE HISTORY
# ============================================================

def save_prediction_history(
    prediction_date,
    result,
):

    new_row = pd.DataFrame(
        [
            {
                "forecast_date":
                    prediction_date.strftime(
                        "%Y-%m-%d"
                    ),

                "rain_probability":
                    round(
                        result["probability"],
                        2,
                    ),

                "rain_prediction":
                    result["prediction"],

                "predicted_rainfall_mm":
                    round(
                        result["rainfall"],
                        2,
                    ),

                "confidence":
                    result["confidence"],

                "risk":
                    result["risk"],
            }
        ]
    )

    try:

        if PREDICTION_HISTORY_FILE.exists():

            history = pd.read_csv(
                PREDICTION_HISTORY_FILE
            )

            if "forecast_date" in history.columns:

                history = history[
                    history[
                        "forecast_date"
                    ].astype(str)
                    != prediction_date.strftime(
                        "%Y-%m-%d"
                    )
                ]

            history = pd.concat(
                [
                    history,
                    new_row,
                ],
                ignore_index=True,
            )

        else:

            history = new_row

        history.to_csv(
            PREDICTION_HISTORY_FILE,
            index=False,
        )

    except Exception as error:

        logger.exception(
            "Could not save prediction history."
        )

        st.warning(
            f"Prediction generated, but history "
            f"could not be saved: {error}"
        )


# ============================================================
# LOAD APPLICATION
# ============================================================

try:

    (
        classification_model,
        classification_features,
        regression_model,
        regression_features,
        probability_threshold,
    ) = load_models()

    historical = load_historical()

except Exception as error:

    logger.exception(
        "Application initialization failed."
    )

    st.error(
        "Unable to load the rainfall prediction system."
    )

    with st.expander(
        "Technical error"
    ):

        st.exception(error)

    st.stop()


# ============================================================
# DATA INFORMATION
# ============================================================

historical_start = (
    historical["date"].min()
)

historical_end = (
    historical["date"].max()
)

historical_rows = len(
    historical
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title(
        "🌧️ Rainfall AI"
    )

    st.markdown(
        """
        ### Kerala Rainfall Prediction

        Next-day rainfall prediction using:

        - Historical weather data
        - Recent rainfall patterns
        - 55 engineered features
        - ECMWF forecast
        - Optimized Logistic Regression
        - Optimized Extra Trees
        """
    )

    st.divider()

    st.info(
        f"""
        **FINAL OPTIMIZED MODELS**

        Classification:
        Optimized Logistic Regression

        Regression:
        Optimized Extra Trees

        Classification threshold:
        {probability_threshold:.0%}

        Rainfall threshold:
        {RAIN_THRESHOLD:.1f} mm

        Features:
        55
        """
    )

    st.divider()

    st.caption(
        f"Location: "
        f"{LATITUDE}°N, {LONGITUDE}°E"
    )

    st.caption(
        f"Historical rows: "
        f"{historical_rows:,}"
    )

    st.caption(
        f"Historical dataset: "
        f"{historical_start:%Y-%m-%d} → "
        f"{historical_end:%Y-%m-%d}"
    )

    st.divider()

    st.success(
        "✓ 55-feature models loaded"
    )

    st.success(
        "✓ ECMWF forecast enabled"
    )


# ============================================================
# HEADER
# ============================================================

st.title(
    "🌧️ Kerala Rainfall Prediction System"
)

st.markdown(
    """
    ### AI-based 7-day rainfall prediction

    The system combines recent weather history with
    ECMWF forecast data and 55 engineered features
    to estimate rain probability, rainfall amount
    and planning risk.
    """
)


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🔮 Prediction",
        "📊 Model Performance",
        "📜 Prediction History",
        "ℹ️ About",
    ]
)


# ============================================================
# PREDICTION TAB
# ============================================================

with tab1:

    st.subheader(
        "Select Forecast Date (Next 7 Days)"
    )

    today = (
        pd.Timestamp.now(
            tz=TIMEZONE
        )
        .normalize()
        .tz_localize(None)
    )

    tomorrow = (
        today +
        pd.Timedelta(days=1)
    )

    prediction_date = st.date_input(
        "Prediction date",

        value=tomorrow.date(),

        min_value=tomorrow.date(),

        max_value=(tomorrow + pd.Timedelta(days=6)).date(),
    )

    prediction_date = pd.Timestamp(
        prediction_date
    )

    st.caption(
        f"Current system date: "
        f"{today:%Y-%m-%d}"
    )

    if st.button(
        "🔮 Predict Rainfall",
        type="primary",
        use_container_width=True,
    ):

        with st.spinner(
            "Fetching recent weather and ECMWF "
            "forecast data..."
        ):

            try:

                result = generate_prediction(
                    prediction_date,
                    historical,
                    classification_model,
                    classification_features,
                    regression_model,
                    regression_features,
                    probability_threshold,
                )

                st.session_state[
                    "prediction_result"
                ] = result

                st.session_state[
                    "prediction_date"
                ] = prediction_date

                save_prediction_history(
                    prediction_date,
                    result,
                )

                st.success(
                    "Prediction completed successfully."
                )

            except Exception as error:

                logger.exception(
                    "Prediction failed."
                )

                st.error(
                    f"Prediction failed: {error}"
                )

                with st.expander(
                    "Technical details"
                ):

                    st.exception(error)

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    if (
        "prediction_result"
        in st.session_state
    ):

        result = st.session_state[
            "prediction_result"
        ]

        result_date = st.session_state[
            "prediction_date"
        ]

        st.divider()

        st.subheader(
            f"🌧️ Prediction for "
            f"{result_date:%d %B %Y}"
        )

        # ----------------------------------------------------
        # MAIN METRICS
        # ----------------------------------------------------

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.metric(
                "Rain Probability",
                f"{result['probability']:.2f}%",
            )

        with c2:

            st.metric(
                "Predicted Rainfall",
                f"{result['rainfall']:.2f} mm",
            )

        with c3:

            st.metric(
                "Rain Prediction",
                result["prediction"],
            )

        with c4:

            st.metric(
                "Planning Risk",
                result["risk"],
            )

        st.divider()

        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        if result["prediction"] == "RAIN":

            st.success(
                f"🌧️ RAIN EXPECTED — "
                f"{result['probability']:.2f}% probability"
            )

        else:

            st.info(
                f"☀️ NO SIGNIFICANT RAIN EXPECTED — "
                f"{result['probability']:.2f}% probability"
            )

        st.write(
            f"**Confidence:** "
            f"{result['confidence']}"
        )

        st.write(
            f"**Rainfall category:** "
            f"{result['category']}"
        )

        st.write(
            f"**Classification threshold:** "
            f"{probability_threshold:.0%}"
        )

        st.write(
            f"**Rainfall threshold:** "
            f"{RAIN_THRESHOLD:.2f} mm"
        )

        # ----------------------------------------------------
        # WEATHER
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "🌡️ ECMWF Forecast Weather"
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.metric(
                "Temperature",
                f"{result['temperature']:.2f} °C",
            )

        with c2:

            st.metric(
                "Dew Point",
                f"{result['dew_point']:.2f} °C",
            )

        with c3:

            st.metric(
                "Pressure",
                f"{result['pressure']:.2f} hPa",
            )

        with c4:

            st.metric(
                "Wind Speed",
                f"{result['wind_speed']:.2f} m/s",
            )

        st.caption(
            f"ECMWF forecast precipitation: "
            f"{result['forecast_rainfall']:.2f} mm"
        )

        # ----------------------------------------------------
        # RECENT RAINFALL
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "🌧️ Recent Rainfall"
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Previous Day",
                f"{result['rain_lag_1']:.2f} mm",
            )

        with c2:

            st.metric(
                "Previous 3 Days",
                f"{result['rain_3day']:.2f} mm",
            )

        with c3:

            st.metric(
                "Previous 7 Days",
                f"{result['rain_7day']:.2f} mm",
            )

        # ----------------------------------------------------
        # WEATHER CHANGES
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "📈 Weather Changes"
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Pressure Change",
                f"{result['pressure_change']:+.2f} hPa",
            )

        with c2:

            st.metric(
                "Temperature Change",
                f"{result['temperature_change']:+.2f} °C",
            )

        with c3:

            st.metric(
                "Dew Point Change",
                f"{result['dewpoint_change']:+.2f} °C",
            )

        # ----------------------------------------------------
        # PLANNING
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "⚠️ Planning Assessment"
        )

        if result["risk"] == "HIGH":

            st.error(
                f"**Risk Level: HIGH**\n\n"
                f"{result['recommendation']}"
            )

        elif result["risk"] == "MODERATE":

            st.warning(
                f"**Risk Level: MODERATE**\n\n"
                f"{result['recommendation']}"
            )

        else:

            st.success(
                f"**Risk Level: LOW**\n\n"
                f"{result['recommendation']}"
            )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

with tab2:

    st.header(
        "📊 Model Performance"
    )

    st.subheader(
        "Optimized Classification Model"
    )

    st.markdown(
        """
        **Model:** Logistic Regression  
        **Selected configuration:** C = 0.01, class_weight = balanced  
        **Validation:** 2025 holdout  
        """
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:

        st.metric(
            "Accuracy",
            "93.96%",
        )

    with c2:

        st.metric(
            "Precision",
            "92.22%",
        )

    with c3:

        st.metric(
            "Recall",
            "84.69%",
        )

    with c4:

        st.metric(
            "F1 Score",
            "88.30%",
        )

    with c5:

        st.metric(
            "ROC-AUC",
            "97.51%",
        )

    st.caption(
        "2025 validation — 364 samples"
    )

    st.divider()

    st.subheader(
        "Optimized Regression Model"
    )

    st.markdown(
        """
        **Model:** Extra Trees Regressor  
        **n_estimators:** 700  
        **min_samples_split:** 5  
        **min_samples_leaf:** 2  
        **max_features:** 1.0  
        **max_depth:** None
        """
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "MAE",
            "0.3406 mm",
        )

    with c2:

        st.metric(
            "MSE",
            "0.3721",
        )

    with c3:

        st.metric(
            "RMSE",
            "0.6100 mm",
        )

    with c4:

        st.metric(
            "R²",
            "0.8029",
        )

    st.caption(
        "2025 validation — 364 samples"
    )

    st.divider()

    st.subheader(
        "Optimized Regression Comparison"
    )

    comparison = pd.DataFrame(
        {
            "Model": [
                "Extra Trees",
                "Gradient Boosting",
                "Random Forest",
                "Hist Gradient Boosting",
            ],

            "CV RMSE": [
                0.846654,
                0.834074,
                0.859756,
                0.868359,
            ],

            "MAE (mm)": [
                0.340592,
                0.338996,
                0.350140,
                0.362237,
            ],

            "RMSE (mm)": [
                0.609980,
                0.628989,
                0.638486,
                0.660479,
            ],

            "R²": [
                0.802854,
                0.790375,
                0.783997,
                0.768860,
            ],
        }
    )

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True,
    )

    st.success(
        "Final optimized regression model: "
        "Extra Trees Regressor"
    )

    st.divider()

    st.subheader(
        "Classification Confusion Matrix"
    )

    confusion = pd.DataFrame(
        {
            "": [
                "Actual NO RAIN",
                "Actual RAIN",
            ],

            "Predicted NO RAIN": [
                259,
                15,
            ],

            "Predicted RAIN": [
                7,
                83,
            ],
        }
    )

    st.dataframe(
        confusion,
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        f"Operational probability threshold: "
        f"{probability_threshold:.0%}"
    )


# ============================================================
# PREDICTION HISTORY
# ============================================================

with tab3:

    st.header(
        "📜 Prediction History"
    )

    if PREDICTION_HISTORY_FILE.exists():

        try:

            history = pd.read_csv(
                PREDICTION_HISTORY_FILE
            )

            if len(history) > 0:

                history = (
                    history
                    .sort_values(
                        "forecast_date",
                        ascending=False,
                    )
                )

                st.dataframe(
                    history,
                    use_container_width=True,
                    hide_index=True,
                )

            else:

                st.info(
                    "No prediction history available."
                )

        except Exception as error:

            st.error(
                f"Unable to read prediction history: "
                f"{error}"
            )

    else:

        st.info(
            "No prediction history available yet."
        )


# ============================================================
# ABOUT
# ============================================================

with tab4:

    st.header(
        "ℹ️ About the Project"
    )

    st.markdown(
        f"""
        ## Kerala Rainfall Prediction using Machine Learning

        This system predicts next-day rainfall using
        historical weather information, recent weather
        history and ECMWF forecast data.

        ### Current System

        - Historical dataset:
          **{historical_start:%Y-%m-%d} → {historical_end:%Y-%m-%d}**
        - Historical rows:
          **{historical_rows:,}**
        - Forecast location:
          **{LATITUDE}°N, {LONGITUDE}°E**
        - Forecast model:
          **ECMWF IFS 0.25°**
        - Engineered features:
          **55**

        ### Classification

        **Optimized Logistic Regression**

        - C = 0.01
        - class_weight = balanced
        - Probability threshold = 67%
        - Rain threshold = 1.0 mm

        ### Regression

        **Optimized Extra Trees Regressor**

        - 700 estimators
        - min_samples_split = 5
        - min_samples_leaf = 2
        - max_features = 1.0
        - max_depth = None

        ### Pipeline

        Historical weather
        → Recent weather history
        → ECMWF forecast
        → 55 feature engineering
        → Logistic Regression
        → Rain probability
        → Extra Trees
        → Rainfall amount
        → Planning risk

        ### Important

        The dashboard loads the trained models only.

        It does not retrain or overwrite the ML models.
        """
    )

    st.success(
        "✓ 55-feature rainfall prediction system ready"
    )