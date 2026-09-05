import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor
import joblib


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = Path(
    "data/processed/nhanes_preprocessed.csv"
)

OUTPUT_DIR = Path("data/results")
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


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
# LOAD DATA
# ============================================================

print("=" * 60)
print("HEPATOTWIN - MODEL COMPARISON")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print("\nDataset shape:", df.shape)


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
# DEFINE MODELS
# ============================================================

models = {

    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
}


# ============================================================
# TRAIN AND EVALUATE
# ============================================================

results = []

trained_models = {}

for name, model in models.items():

    print("\n" + "-" * 60)
    print(f"Training {name}...")
    print("-" * 60)

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(X_test)

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

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    trained_models[name] = model

    print(f"MAE  : {mae:.2f} dB/m")
    print(f"RMSE : {rmse:.2f} dB/m")
    print(f"R²   : {r2:.3f}")


# ============================================================
# MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="RMSE",
    ascending=True
)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    results_df.to_string(
        index=False,
        formatters={
            "MAE": "{:.2f}".format,
            "RMSE": "{:.2f}".format,
            "R2": "{:.3f}".format
        }
    )
)


# ============================================================
# SELECT BEST MODEL
# ============================================================

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[
    best_model_name
]

print("\n" + "=" * 60)
print("SELECTED MODEL")
print("=" * 60)

print("Selected model:", best_model_name)

print(
    f"Selection criterion: lowest RMSE"
)


# ============================================================
# SAVE COMPARISON RESULTS
# ============================================================

comparison_file = (
    OUTPUT_DIR /
    "model_comparison.csv"
)

results_df.to_csv(
    comparison_file,
    index=False
)


# ============================================================
# SAVE SELECTED MODEL
# ============================================================

selected_model_file = (
    OUTPUT_DIR /
    "selected_patient_state_model.joblib"
)

joblib.dump(
    best_model,
    selected_model_file
)


# ============================================================
# SAVE MODEL NAME
# ============================================================

selection_file = (
    OUTPUT_DIR /
    "selected_model.txt"
)

with open(
    selection_file,
    "w"
) as f:

    f.write(
        f"Selected model: {best_model_name}\n"
    )

    f.write(
        "Selection criterion: lowest RMSE\n"
    )


# ============================================================
# COMPLETE
# ============================================================

print("\nSaved:")
print(comparison_file)
print(selected_model_file)
print(selection_file)

print("\nModel comparison completed.")