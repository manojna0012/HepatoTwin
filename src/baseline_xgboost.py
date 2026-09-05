import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = Path(
    "data/processed/nhanes_preprocessed.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("NHANES XGBOOST BASELINE MODEL")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print("\nDataset shape:", df.shape)


# ============================================================
# TARGET
# ============================================================

TARGET = "LUXCAPM"


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "RIDAGEYR",
    "BMXBMI",
    "BMXWAIST",
    "LBXSATSI",
    "LBXSASSI",
    "LBXGH",
    "LBXTC",
    "LBDHDD",
    "DR1TKCAL",
    "DR1TPROT",
    "DR1TCARB",
    "DR1TTFAT",

    "RIAGENDR_2.0",

    "RIDRETH3_2.0",
    "RIDRETH3_3.0",
    "RIDRETH3_4.0",
    "RIDRETH3_6.0",
    "RIDRETH3_7.0",
]


# ============================================================
# CHECK COLUMNS
# ============================================================

missing_features = [
    feature
    for feature in FEATURES
    if feature not in df.columns
]

if missing_features:
    raise ValueError(
        f"Missing feature columns: {missing_features}"
    )

if TARGET not in df.columns:
    raise ValueError(
        f"Target column {TARGET} not found"
    )


# ============================================================
# CREATE X AND Y
# ============================================================

X = df[FEATURES]
y = df[TARGET]

print("\nX shape:", X.shape)
print("y shape:", y.shape)

print("\nFeatures:")
for feature in FEATURES:
    print(" ", feature)

print("\nTarget:", TARGET)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTrain samples:", len(X_train))
print("Test samples:", len(X_test))


# ============================================================
# XGBOOST REGRESSOR
# ============================================================

model = XGBRegressor(
    objective="reg:squarederror",

    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,

    subsample=0.8,
    colsample_bytree=0.8,

    random_state=42,
    n_jobs=-1
)


# ============================================================
# TRAIN
# ============================================================

print("\nTraining XGBoost...")

model.fit(
    X_train,
    y_train
)

print("Training complete.")


# ============================================================
# PREDICTION
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# METRICS
# ============================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("BASELINE MODEL RESULTS")
print("=" * 60)

print(f"MAE  : {mae:.2f} dB/m")
print(f"RMSE : {rmse:.2f} dB/m")
print(f"R²   : {r2:.3f}")


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance.to_string(index=False))


# ============================================================
# SAVE FEATURE IMPORTANCE
# ============================================================

OUTPUT_DIR = Path("data/results")
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

importance.to_csv(
    OUTPUT_DIR / "xgboost_feature_importance.csv",
    index=False
)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

predictions = pd.DataFrame({
    "Actual_CAP": y_test.values,
    "Predicted_CAP": y_pred
})

predictions.to_csv(
    OUTPUT_DIR / "xgboost_test_predictions.csv",
    index=False
)

print("\nSaved:")
print("data/results/xgboost_feature_importance.csv")
print("data/results/xgboost_test_predictions.csv")

print("\nBaseline XGBoost completed.")