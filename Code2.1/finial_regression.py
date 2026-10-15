import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR /
    "processed" /
    "daily_weather.csv"
)

MODEL_DIR = (
    BASE_DIR /
    "processed" /
    "models"
)

FEATURE_FILE = (
    MODEL_DIR /
    "features.joblib"
)

RANDOM_FOREST_FILE = (
    MODEL_DIR /
    "random_forest_final.joblib"
)

OUTPUT_FILE = (
    BASE_DIR /
    "processed" /
    "final_regression_verification.csv"
)


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

TRAIN_END = "2024-10-19"
VALIDATION_START = "2025-01-01"
VALIDATION_END = "2025-12-30"


# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("FINAL CONTROLLED REGRESSION VERIFICATION")
print("=" * 75)

print("\nIMPORTANT:")
print("Existing Random Forest model will NOT be modified.")
print("No model files will be overwritten.")

print("\nTraining cutoff:", TRAIN_END)
print(
    "Validation period:",
    VALIDATION_START,
    "to",
    VALIDATION_END
)


# ============================================================
# LOAD RANDOM FOREST
# ============================================================

print("\n" + "=" * 75)
print("LOADING CURRENT RANDOM FOREST")
print("=" * 75)

random_forest = joblib.load(
    RANDOM_FOREST_FILE
)

features = joblib.load(
    FEATURE_FILE
)

print("Random Forest : Loaded")
print("Features      :", len(features))


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 75)
print("LOADING HISTORICAL DATA")
print("=" * 75)

df = pd.read_csv(
    DATA_FILE
)

df["date"] = pd.to_datetime(
    df["date"]
)

df = (
    df
    .sort_values("date")
    .reset_index(drop=True)
)

print(
    "\nData period:",
    df["date"].min().date(),
    "to",
    df["date"].max().date()
)

print(
    "Total rows:",
    len(df)
)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 75)
print("CREATING FEATURES")
print("=" * 75)


df["rain_lag_1"] = (
    df["rainfall"].shift(1)
)

df["rain_lag_2"] = (
    df["rainfall"].shift(2)
)

df["rain_lag_3"] = (
    df["rainfall"].shift(3)
)

df["rain_lag_7"] = (
    df["rainfall"].shift(7)
)


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


df["pressure_change"] = (
    df["pressure"]
    -
    df["pressure"].shift(1)
)

df["temperature_change"] = (
    df["temperature"]
    -
    df["temperature"].shift(1)
)

df["dewpoint_change"] = (
    df["dew_point"]
    -
    df["dew_point"].shift(1)
)


df["month"] = (
    df["date"].dt.month
)

df["day_of_year"] = (
    df["date"].dt.dayofyear
)

df["day_of_week"] = (
    df["date"].dt.dayofweek
)


# ============================================================
# TARGET
# ============================================================

df["target_rainfall"] = (
    df["rainfall"].shift(-1)
)


# ============================================================
# REMOVE INVALID ROWS
# ============================================================

df = (
    df
    .dropna()
    .reset_index(drop=True)
)


# ============================================================
# EXACT TRAINING / VALIDATION SPLIT
# ============================================================

train_end_date = pd.Timestamp(
    TRAIN_END
)

validation_start_date = pd.Timestamp(
    VALIDATION_START
)

validation_end_date = pd.Timestamp(
    VALIDATION_END
)


train_df = df[
    df["date"] <= train_end_date
].copy()


validation_df = df[
    (
        df["date"] >= validation_start_date
    )
    &
    (
        df["date"] <= validation_end_date
    )
].copy()


print("\n" + "=" * 75)
print("DATA SPLIT")
print("=" * 75)

print(
    "\nTraining:",
    train_df["date"].min().date(),
    "to",
    train_df["date"].max().date()
)

print(
    "Training samples:",
    len(train_df)
)

print(
    "\nValidation:",
    validation_df["date"].min().date(),
    "to",
    validation_df["date"].max().date()
)

print(
    "Validation samples:",
    len(validation_df)
)


# ============================================================
# INPUT / TARGET
# ============================================================

X_train = train_df[
    features
]

y_train = train_df[
    "target_rainfall"
]

X_validation = validation_df[
    features
]

y_validation = validation_df[
    "target_rainfall"
]


print("\nFeature matrices:")
print(
    "X_train      :",
    X_train.shape
)

print(
    "X_validation :",
    X_validation.shape
)


# ============================================================
# TRAIN HIST GRADIENT BOOSTING
# ============================================================

print("\n" + "=" * 75)
print("TRAINING HIST GRADIENT BOOSTING")
print("=" * 75)

hist_gradient_boosting = HistGradientBoostingRegressor(
    learning_rate=0.05,
    max_iter=300,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    l2_regularization=1.0,
    random_state=42
)

hist_gradient_boosting.fit(
    X_train,
    y_train
)

print("Training complete.")


# ============================================================
# TRAIN GRADIENT BOOSTING
# ============================================================

print("\n" + "=" * 75)
print("TRAINING GRADIENT BOOSTING")
print("=" * 75)

gradient_boosting = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    min_samples_split=10,
    min_samples_leaf=5,
    subsample=0.9,
    random_state=42
)

gradient_boosting.fit(
    X_train,
    y_train
)

print("Training complete.")


# ============================================================
# PREDICTIONS
# ============================================================

print("\n" + "=" * 75)
print("GENERATING VALIDATION PREDICTIONS")
print("=" * 75)


rf_prediction = random_forest.predict(
    X_validation
)

hist_prediction = hist_gradient_boosting.predict(
    X_validation
)

gb_prediction = gradient_boosting.predict(
    X_validation
)


# Prevent negative rainfall predictions

rf_prediction = np.maximum(
    rf_prediction,
    0
)

hist_prediction = np.maximum(
    hist_prediction,
    0
)

gb_prediction = np.maximum(
    gb_prediction,
    0
)


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    actual,
    predicted
):

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    r2 = r2_score(
        actual,
        predicted
    )

    return mae, rmse, r2


# ============================================================
# CALCULATE METRICS
# ============================================================

rf_mae, rf_rmse, rf_r2 = calculate_metrics(
    y_validation,
    rf_prediction
)

hist_mae, hist_rmse, hist_r2 = calculate_metrics(
    y_validation,
    hist_prediction
)

gb_mae, gb_rmse, gb_r2 = calculate_metrics(
    y_validation,
    gb_prediction
)


# ============================================================
# CREATE COMPARISON
# ============================================================

comparison = pd.DataFrame({

    "Model": [
        "Current Random Forest",
        "Hist Gradient Boosting",
        "Gradient Boosting"
    ],

    "MAE": [
        rf_mae,
        hist_mae,
        gb_mae
    ],

    "RMSE": [
        rf_rmse,
        hist_rmse,
        gb_rmse
    ],

    "R2": [
        rf_r2,
        hist_r2,
        gb_r2
    ]

})


# ============================================================
# RANKING
# ============================================================

comparison["MAE_rank"] = (
    comparison["MAE"]
    .rank(
        method="min",
        ascending=True
    )
)

comparison["RMSE_rank"] = (
    comparison["RMSE"]
    .rank(
        method="min",
        ascending=True
    )
)

comparison["R2_rank"] = (
    comparison["R2"]
    .rank(
        method="min",
        ascending=False
    )
)

comparison["Overall_rank_score"] = (
    comparison["MAE_rank"]
    +
    comparison["RMSE_rank"]
    +
    comparison["R2_rank"]
)

comparison = (
    comparison
    .sort_values(
        "Overall_rank_score"
    )
    .reset_index(drop=True)
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n\n" + "=" * 75)
print("FINAL CONTROLLED 2025 COMPARISON")
print("=" * 75)

print(
    "\nTraining period:",
    train_df["date"].min().date(),
    "to",
    train_df["date"].max().date()
)

print(
    "Validation period:",
    validation_df["date"].min().date(),
    "to",
    validation_df["date"].max().date()
)

print(
    "Validation samples:",
    len(validation_df)
)

print("\n" + "-" * 75)

print(
    comparison[
        [
            "Model",
            "MAE",
            "RMSE",
            "R2"
        ]
    ].to_string(
        index=False,
        formatters={
            "MAE": "{:.4f}".format,
            "RMSE": "{:.4f}".format,
            "R2": "{:.4f}".format
        }
    )
)


# ============================================================
# WINNER
# ============================================================

winner = comparison.iloc[0]

print("\n" + "=" * 75)
print("FINAL REGRESSION CANDIDATE")
print("=" * 75)

print(
    "\nModel:",
    winner["Model"]
)

print(
    "MAE  :",
    f"{winner['MAE']:.4f} mm"
)

print(
    "RMSE :",
    f"{winner['RMSE']:.4f} mm"
)

print(
    "R²   :",
    f"{winner['R2']:.4f}"
)


# ============================================================
# SAVE COMPARISON
# ============================================================

comparison.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SAFETY CHECK
# ============================================================

print("\n" + "=" * 75)
print("MODEL SAFETY CHECK")
print("=" * 75)

print(
    "\nrandom_forest_final.joblib:"
)

print(
    RANDOM_FOREST_FILE
)

print(
    "\nSTATUS: NOT MODIFIED"
)

print(
    "\nResults saved to:"
)

print(
    OUTPUT_FILE
)


print("\n" + "=" * 75)
print("FINAL CONTROLLED VERIFICATION COMPLETE")
print("=" * 75)