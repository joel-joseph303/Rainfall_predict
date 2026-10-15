# ============================================================
# FINAL 55-FEATURE MODEL EVALUATION
# REGRESSION + CLASSIFICATION
# ============================================================

import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
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

OUTPUT_FILE = (
    PROCESSED_DIR / "final_55_feature_evaluation.csv"
)


# ============================================================
# VALIDATION SETTINGS
# ============================================================

VALIDATION_START = "2025-01-01"
VALIDATION_END = "2025-12-30"

RAIN_THRESHOLD = 1.0


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("FINAL 55-FEATURE MODEL EVALUATION")
print("=" * 80)

print()
print("This evaluation does NOT modify any model files.")

print()
print("Validation period:")
print(f"{VALIDATION_START} to {VALIDATION_END}")


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

print()
print("=" * 80)
print("CHECKING REQUIRED FILES")
print("=" * 80)

required_files = [
    DATA_FILE,
    CLASSIFICATION_MODEL_FILE,
    CLASSIFICATION_FEATURES_FILE,
    CLASSIFICATION_THRESHOLD_FILE,
    REGRESSION_MODEL_FILE,
    REGRESSION_FEATURES_FILE,
]

for file in required_files:

    if not file.exists():

        raise FileNotFoundError(
            f"\nRequired file not found:\n{file}"
        )

    print(f"✓ {file.name}")


# ============================================================
# LOAD HISTORICAL DATA
# ============================================================

print()
print("=" * 80)
print("LOADING HISTORICAL DATA")
print("=" * 80)

df = pd.read_csv(DATA_FILE)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    "date"
).reset_index(drop=True)

print(f"Rows : {len(df)}")

print(
    f"Period: "
    f"{df['date'].min().date()} "
    f"to "
    f"{df['date'].max().date()}"
)


# ============================================================
# CREATE 55 FEATURES
# ============================================================

print()
print("=" * 80)
print("CREATING 55 FEATURES")
print("=" * 80)


# ------------------------------------------------------------
# BASIC CALCULATED FEATURES
# ------------------------------------------------------------

df["calculated_wind_speed"] = np.sqrt(
    df["u10"] ** 2 +
    df["v10"] ** 2
)

df["dewpoint_depression"] = (
    df["temperature"] -
    df["dew_point"]
)

df["temperature_dewpoint_ratio"] = (
    df["temperature"] /
    df["dew_point"].replace(0, np.nan)
)


# ------------------------------------------------------------
# RAINFALL LAGS
# ------------------------------------------------------------

for lag in [1, 2, 3, 4, 5, 6, 7, 14]:

    df[f"rain_lag_{lag}"] = (
        df["rainfall"].shift(lag)
    )


# ------------------------------------------------------------
# RAINFALL ROLLING SUMS
# ------------------------------------------------------------

for window in [3, 7, 14]:

    df[f"rain_{window}day"] = (
        df["rainfall"]
        .shift(1)
        .rolling(window)
        .sum()
    )


# ------------------------------------------------------------
# RAINFALL ROLLING MEANS
# REGRESSION NAMING
# ------------------------------------------------------------

df["rain_3day_mean"] = (
    df["rainfall"]
    .shift(1)
    .rolling(3)
    .mean()
)

df["rain_7day_mean"] = (
    df["rainfall"]
    .shift(1)
    .rolling(7)
    .mean()
)

df["rain_14day_mean"] = (
    df["rainfall"]
    .shift(1)
    .rolling(14)
    .mean()
)


# ------------------------------------------------------------
# RAINFALL STANDARD DEVIATION
# ------------------------------------------------------------

df["rain_7day_std"] = (
    df["rainfall"]
    .shift(1)
    .rolling(7)
    .std()
)

df["rain_14day_std"] = (
    df["rainfall"]
    .shift(1)
    .rolling(14)
    .std()
)


# ------------------------------------------------------------
# RAINFALL CHANGES
# ------------------------------------------------------------

df["rain_change_1day"] = (
    df["rainfall"].shift(1)
    -
    df["rainfall"].shift(2)
)

df["rain_change_3day"] = (
    df["rainfall"].shift(1)
    -
    df["rainfall"].shift(4)
)

df["rain_change_7day"] = (
    df["rainfall"].shift(1)
    -
    df["rainfall"].shift(8)
)


# ------------------------------------------------------------
# TEMPERATURE CHANGES
# ------------------------------------------------------------

for lag in [1, 3, 7]:

    df[f"temperature_change_{lag}"] = (
        df["temperature"]
        -
        df["temperature"].shift(lag)
    )


# ------------------------------------------------------------
# DEW POINT CHANGES
# ------------------------------------------------------------

for lag in [1, 3, 7]:

    df[f"dewpoint_change_{lag}"] = (
        df["dew_point"]
        -
        df["dew_point"].shift(lag)
    )


# ------------------------------------------------------------
# PRESSURE CHANGES
# ------------------------------------------------------------

for lag in [1, 3, 7]:

    df[f"pressure_change_{lag}"] = (
        df["pressure"]
        -
        df["pressure"].shift(lag)
    )


# ------------------------------------------------------------
# PRESSURE ROLLING MEANS
# ------------------------------------------------------------

df["pressure_3day_mean"] = (
    df["pressure"]
    .shift(1)
    .rolling(3)
    .mean()
)

df["pressure_7day_mean"] = (
    df["pressure"]
    .shift(1)
    .rolling(7)
    .mean()
)


# ------------------------------------------------------------
# WIND CHANGES
# ------------------------------------------------------------

for lag in [1, 3, 7]:

    df[f"wind_change_{lag}"] = (
        df["wind_speed"]
        -
        df["wind_speed"].shift(lag)
    )


# ------------------------------------------------------------
# WIND ROLLING MEANS
# ------------------------------------------------------------

df["wind_3day_mean"] = (
    df["wind_speed"]
    .shift(1)
    .rolling(3)
    .mean()
)

df["wind_7day_mean"] = (
    df["wind_speed"]
    .shift(1)
    .rolling(7)
    .mean()
)


# ------------------------------------------------------------
# WIND DIRECTION
# ------------------------------------------------------------

wind_direction = np.arctan2(
    df["u10"],
    df["v10"]
)

df["wind_direction_sin"] = np.sin(
    wind_direction
)

df["wind_direction_cos"] = np.cos(
    wind_direction
)


# ------------------------------------------------------------
# CALENDAR FEATURES
# ------------------------------------------------------------

df["month"] = (
    df["date"].dt.month
)

df["day_of_year"] = (
    df["date"].dt.dayofyear
)

df["day_of_week"] = (
    df["date"].dt.dayofweek
)


# ------------------------------------------------------------
# CYCLIC CALENDAR FEATURES
# ------------------------------------------------------------

df["day_year_sin"] = np.sin(
    2 * np.pi *
    df["day_of_year"] / 365
)

df["day_year_cos"] = np.cos(
    2 * np.pi *
    df["day_of_year"] / 365
)

df["day_week_sin"] = np.sin(
    2 * np.pi *
    df["day_of_week"] / 7
)

df["day_week_cos"] = np.cos(
    2 * np.pi *
    df["day_of_week"] / 7
)

df["month_sin"] = np.sin(
    2 * np.pi *
    df["month"] / 12
)

df["month_cos"] = np.cos(
    2 * np.pi *
    df["month"] / 12
)


# ============================================================
# CLASSIFICATION FEATURE ALIASES
# ============================================================

# The classification model was saved with slightly
# different names for some mathematically identical features.


df["rain_mean_3day"] = (
    df["rain_3day_mean"]
)

df["rain_mean_7day"] = (
    df["rain_7day_mean"]
)

df["rain_mean_14day"] = (
    df["rain_14day_mean"]
)


df["rain_std_7day"] = (
    df["rain_7day_std"]
)

df["rain_std_14day"] = (
    df["rain_14day_std"]
)


df["rain_change_1"] = (
    df["rain_change_1day"]
)

df["rain_change_3"] = (
    df["rain_change_3day"]
)

df["rain_change_7"] = (
    df["rain_change_7day"]
)


df["pressure_mean_3day"] = (
    df["pressure_3day_mean"]
)

df["pressure_mean_7day"] = (
    df["pressure_7day_mean"]
)


df["wind_mean_3day"] = (
    df["wind_3day_mean"]
)

df["wind_mean_7day"] = (
    df["wind_7day_mean"]
)


df["wind_speed_calculated"] = (
    df["calculated_wind_speed"]
)


df["temp_dew_ratio"] = (
    df["temperature_dewpoint_ratio"]
)


df["day_of_year_sin"] = (
    df["day_year_sin"]
)

df["day_of_year_cos"] = (
    df["day_year_cos"]
)


df["day_of_week_sin"] = (
    df["day_week_sin"]
)

df["day_of_week_cos"] = (
    df["day_week_cos"]
)


# ============================================================
# LOAD OPTIMIZED MODELS
# ============================================================

print()
print("=" * 80)
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


# ------------------------------------------------------------
# LOAD CLASSIFICATION THRESHOLD
# ------------------------------------------------------------

with open(
    CLASSIFICATION_THRESHOLD_FILE,
    "r"
) as f:

    classification_threshold = float(
        f.read().strip()
    )


print(
    "Classification model:",
    type(classification_model).__name__
)

print(
    "Regression model:",
    type(regression_model).__name__
)

print(
    "Classification features:",
    len(classification_features)
)

print(
    "Regression features:",
    len(regression_features)
)

print(
    "Classification threshold:",
    classification_threshold
)


# ============================================================
# VERIFY FEATURE COUNTS
# ============================================================

if len(classification_features) != 55:

    raise ValueError(
        "Classification model does not contain "
        "exactly 55 features."
    )


if len(regression_features) != 55:

    raise ValueError(
        "Regression model does not contain "
        "exactly 55 features."
    )


# ============================================================
# CREATE RAIN CLASS
# ============================================================

df["rain_class"] = (
    df["rainfall"] >= RAIN_THRESHOLD
).astype(int)


# ============================================================
# CREATE VALIDATION DATASET
# ============================================================

print()
print("=" * 80)
print("CREATING VALIDATION DATASET")
print("=" * 80)

validation = df[
    (df["date"] >= VALIDATION_START)
    &
    (df["date"] <= VALIDATION_END)
].copy()


print(
    f"Validation samples: {len(validation)}"
)


# ============================================================
# CHECK ALL REQUIRED COLUMNS
# ============================================================

required_columns = list(
    set(classification_features)
    |
    set(regression_features)
)

missing_columns = [
    column
    for column in required_columns
    if column not in validation.columns
]

if missing_columns:

    print()
    print("MISSING FEATURE COLUMNS:")

    for column in missing_columns:

        print(
            " -",
            column
        )

    raise ValueError(
        "\nSome required 55-feature columns "
        "were not created."
    )


# ============================================================
# REMOVE ROWS WITH MISSING FEATURES
# ============================================================

validation = validation.dropna(
    subset=required_columns
).copy()


print(
    f"Usable validation samples: "
    f"{len(validation)}"
)


# ============================================================
# CREATE FEATURE MATRICES
# ============================================================

X_classification = validation[
    classification_features
].copy()

X_regression = validation[
    regression_features
].copy()


y_classification = validation[
    "rain_class"
].values

y_regression = validation[
    "rainfall"
].values


print()
print(
    "Classification matrix:",
    X_classification.shape
)

print(
    "Regression matrix:",
    X_regression.shape
)


# ============================================================
# VERIFY FEATURE ORDER
# ============================================================

print()
print("=" * 80)
print("VERIFYING FEATURES")
print("=" * 80)


if list(X_classification.columns) != list(
    classification_features
):

    raise ValueError(
        "Classification feature order does not match "
        "the saved feature list."
    )


if list(X_regression.columns) != list(
    regression_features
):

    raise ValueError(
        "Regression feature order does not match "
        "the saved feature list."
    )


print(
    "✓ Classification feature order verified"
)

print(
    "✓ Regression feature order verified"
)


# ============================================================
# VERIFY NUMERIC VALUES
# ============================================================

if not np.isfinite(
    X_classification.to_numpy()
).all():

    raise ValueError(
        "Classification matrix contains "
        "invalid numeric values."
    )


if not np.isfinite(
    X_regression.to_numpy()
).all():

    raise ValueError(
        "Regression matrix contains "
        "invalid numeric values."
    )


print(
    "✓ No missing values"
)

print(
    "✓ No invalid numeric values"
)


# ============================================================
# CLASSIFICATION PREDICTION
# ============================================================

print()
print("=" * 80)
print("CLASSIFICATION EVALUATION")
print("=" * 80)


rain_probability = (
    classification_model
    .predict_proba(
        X_classification
    )[:, 1]
)


predicted_class = (
    rain_probability
    >= classification_threshold
).astype(int)


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

accuracy = accuracy_score(
    y_classification,
    predicted_class
)

precision = precision_score(
    y_classification,
    predicted_class,
    zero_division=0
)

recall = recall_score(
    y_classification,
    predicted_class,
    zero_division=0
)

f1 = f1_score(
    y_classification,
    predicted_class,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_classification,
    rain_probability
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_classification,
    predicted_class
)

tn, fp, fn, tp = cm.ravel()


print()
print("CLASSIFICATION METRICS")
print("----------------------")

print(
    f"Accuracy  : "
    f"{accuracy:.4f} "
    f"({accuracy * 100:.2f}%)"
)

print(
    f"Precision : "
    f"{precision:.4f} "
    f"({precision * 100:.2f}%)"
)

print(
    f"Recall    : "
    f"{recall:.4f} "
    f"({recall * 100:.2f}%)"
)

print(
    f"F1 Score  : "
    f"{f1:.4f} "
    f"({f1 * 100:.2f}%)"
)

print(
    f"ROC-AUC   : "
    f"{roc_auc:.4f}"
)


print()
print("CONFUSION MATRIX")
print("----------------")

print(
    f"TN : {tn}"
)

print(
    f"FP : {fp}"
)

print(
    f"FN : {fn}"
)

print(
    f"TP : {tp}"
)


# ============================================================
# REGRESSION PREDICTION
# ============================================================

print()
print("=" * 80)
print("REGRESSION EVALUATION")
print("=" * 80)


predicted_rainfall = (
    regression_model
    .predict(
        X_regression
    )
)


# ============================================================
# REGRESSION METRICS
# ============================================================

mae = mean_absolute_error(
    y_regression,
    predicted_rainfall
)

mse = mean_squared_error(
    y_regression,
    predicted_rainfall
)

rmse = np.sqrt(
    mse
)

r2 = r2_score(
    y_regression,
    predicted_rainfall
)


print()
print("REGRESSION METRICS")
print("------------------")

print(
    f"MAE  : "
    f"{mae:.4f} mm"
)

print(
    f"MSE  : "
    f"{mse:.4f}"
)

print(
    f"RMSE : "
    f"{rmse:.4f} mm"
)

print(
    f"R²   : "
    f"{r2:.4f}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

results = pd.DataFrame({

    "model_type": [
        "Classification",
        "Regression"
    ],

    "model": [
        type(classification_model).__name__,
        type(regression_model).__name__
    ],

    "features": [
        len(classification_features),
        len(regression_features)
    ],

    "validation_start": [
        VALIDATION_START,
        VALIDATION_START
    ],

    "validation_end": [
        VALIDATION_END,
        VALIDATION_END
    ],

    "samples": [
        len(validation),
        len(validation)
    ],

    "accuracy": [
        accuracy,
        np.nan
    ],

    "precision": [
        precision,
        np.nan
    ],

    "recall": [
        recall,
        np.nan
    ],

    "f1": [
        f1,
        np.nan
    ],

    "roc_auc": [
        roc_auc,
        np.nan
    ],

    "MAE": [
        np.nan,
        mae
    ],

    "MSE": [
        np.nan,
        mse
    ],

    "RMSE": [
        np.nan,
        rmse
    ],

    "R2": [
        np.nan,
        r2
    ],

    "threshold": [
        classification_threshold,
        np.nan
    ],

    "TN": [
        tn,
        np.nan
    ],

    "FP": [
        fp,
        np.nan
    ],

    "FN": [
        fn,
        np.nan
    ],

    "TP": [
        tp,
        np.nan
    ]
})


results.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 80)
print("FINAL 55-FEATURE EVALUATION SUMMARY")
print("=" * 80)


print()
print("CLASSIFICATION")
print("----------------")

print(
    f"Accuracy  : "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Precision : "
    f"{precision * 100:.2f}%"
)

print(
    f"Recall    : "
    f"{recall * 100:.2f}%"
)

print(
    f"F1 Score  : "
    f"{f1 * 100:.2f}%"
)

print(
    f"ROC-AUC   : "
    f"{roc_auc:.4f}"
)


print()
print("REGRESSION")
print("----------------")

print(
    f"MAE  : "
    f"{mae:.4f} mm"
)

print(
    f"MSE  : "
    f"{mse:.4f}"
)

print(
    f"RMSE : "
    f"{rmse:.4f} mm"
)

print(
    f"R²   : "
    f"{r2:.4f}"
)


print()
print("CONFUSION MATRIX")
print("----------------")

print(
    f"TN : {tn}"
)

print(
    f"FP : {fp}"
)

print(
    f"FN : {fn}"
)

print(
    f"TP : {tp}"
)


print()
print("Results saved to:")
print(OUTPUT_FILE)


print()
print("=" * 80)
print("EVALUATION COMPLETE")
print("=" * 80)