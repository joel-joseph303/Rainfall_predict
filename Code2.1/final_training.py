import pandas as pd
import joblib

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR /
    "processed" /
    "rainfall_features.csv"
)

MODEL_DIR = (
    BASE_DIR /
    "processed" /
    "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("FINAL MODEL TRAINING")
print("=" * 70)

df = pd.read_csv(DATA_FILE)

df["date"] = pd.to_datetime(df["date"])

df = (
    df
    .sort_values("date")
    .reset_index(drop=True)
)


# ============================================================
# FEATURES
# ============================================================

features = [
    "temperature",
    "dew_point",
    "pressure",
    "u10",
    "v10",
    "wind_speed",
    "rain_lag_1",
    "rain_lag_2",
    "rain_lag_3",
    "rain_lag_7",
    "rain_3day",
    "rain_7day",
    "pressure_change",
    "temperature_change",
    "dewpoint_change",
    "month",
    "day_of_year",
    "day_of_week"
]


X = df[features]


# ============================================================
# TARGETS
# ============================================================

# Classification target
df["rain_class"] = (
    df["target_rainfall"] >= 1.0
).astype(int)

y_class = df["rain_class"]


# Regression target
y_reg = df["target_rainfall"]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_class_train = y_class.iloc[:split_index]
y_class_test = y_class.iloc[split_index:]

y_reg_train = y_reg.iloc[:split_index]
y_reg_test = y_reg.iloc[split_index:]


print("\nDATA SPLIT")
print("=" * 70)

print("Training:", len(X_train))
print("Testing :", len(X_test))

print(
    "Training period:",
    df["date"].iloc[0],
    "to",
    df["date"].iloc[split_index - 1]
)

print(
    "Testing period:",
    df["date"].iloc[split_index],
    "to",
    df["date"].iloc[-1]
)


# ============================================================
# FINAL CLASSIFICATION MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 70)

classification_model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)

classification_model.fit(
    X_train,
    y_class_train
)


# ============================================================
# FINAL REGRESSION MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING RANDOM FOREST")
print("=" * 70)

regression_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    max_features=0.7,
    min_samples_leaf=2,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)

regression_model.fit(
    X_train,
    y_reg_train
)


# ============================================================
# SAVE MODELS
# ============================================================

classification_path = (
    MODEL_DIR /
    "logistic_regression_final.joblib"
)

regression_path = (
    MODEL_DIR /
    "random_forest_final.joblib"
)

joblib.dump(
    classification_model,
    classification_path
)

joblib.dump(
    regression_model,
    regression_path
)


# ============================================================
# SAVE FEATURE LIST
# ============================================================

feature_path = (
    MODEL_DIR /
    "features.joblib"
)

joblib.dump(
    features,
    feature_path
)


# ============================================================
# SAVE MODEL INFORMATION
# ============================================================

model_info = {
    "classification_model": "Logistic Regression",
    "classification_threshold": 0.50,

    "regression_model": "Random Forest",

    "n_estimators": 200,
    "max_depth": 10,
    "max_features": 0.7,
    "min_samples_leaf": 2,
    "min_samples_split": 2,

    "rainfall_threshold_mm": 1.0,

    "regression_mae": 0.5786,
    "regression_rmse": 0.8945,
    "regression_r2": 0.5365,

    "classification_accuracy": 0.8055,
    "classification_precision": 0.5917,
    "classification_recall": 0.8621,
    "classification_f1": 0.7018,
    "classification_roc_auc": 0.8876
}

info_path = (
    MODEL_DIR /
    "model_info.joblib"
)

joblib.dump(
    model_info,
    info_path
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("FINAL MODELS SAVED")
print("=" * 70)

print("\nClassification model:")
print(classification_path)

print("\nRegression model:")
print(regression_path)

print("\nFeature list:")
print(feature_path)

print("\nModel information:")
print(info_path)

print("\n" + "=" * 70)
print("FINAL MODEL TRAINING COMPLETE")
print("=" * 70)