"""MLP neural-network experiment for CAP prediction."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from src.cap_features import CAPFeatures, RAW_FEATURES, RAW_NUMERIC, ENGINEERED
from src.train_final_model import ROOT, TARGET, load_cohort

SOURCE = ROOT / "data/processed/nhanes_merged.csv"
SPLIT_DIR = ROOT / "data/results/second_pass"
OUT = ROOT / "data/results/experiments/mlp"


def metrics(y, pred):
    return {
        "MAE": float(mean_absolute_error(y, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "R2": float(r2_score(y, pred)),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(SOURCE).set_index("SEQN")
    cohort = load_cohort(SOURCE)
    data = data.loc[cohort.SEQN]

    train_ids = pd.read_csv(SPLIT_DIR / "oof_predictions.csv")["SEQN"]
    test_ids = pd.read_csv(SPLIT_DIR / "test_predictions.csv")["SEQN"]
    train = data.loc[train_ids]
    test = data.loc[test_ids]

    if set(train.index) & set(test.index):
        raise ValueError("Train/test participant IDs overlap.")
    if len(train) != 7216 or len(test) != 1805:
        raise ValueError("Unexpected train/test split sizes.")

    X, y = train[RAW_FEATURES], train[TARGET]
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(
            strategy="median", keep_empty_features=True,
            add_indicator=True
        )),
        ("scale", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(
            strategy="most_frequent", keep_empty_features=True
        )),
        ("encode", OneHotEncoder(
            categories=[[1., 2.], [1., 2., 3., 4., 6., 7.]],
            drop="first", handle_unknown="error", sparse_output=False
        )),
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, RAW_NUMERIC + ENGINEERED),
        ("categorical", categorical_pipeline, ["RIAGENDR", "RIDRETH3"]),
    ])

    model = Pipeline([
        ("features", CAPFeatures()),
        ("preprocess", preprocessing),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=0.01,
            learning_rate_init=0.001,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42,
        )),
    ])

    print("Running five-fold CV: MLP neural network", flush=True)
    with threadpool_limits(limits=2):
        pred = cross_val_predict(model, X, y, cv=cv, n_jobs=1)

    result = {"Model": "MLP (128, 64)", **metrics(y, pred)}
    pd.DataFrame([result]).to_csv(OUT / "cv_metrics.csv", index=False)
    pd.DataFrame({
        "SEQN": train.index,
        "Actual_CAP": y.to_numpy(),
        "MLP_prediction": pred,
    }).to_csv(OUT / "oof_predictions.csv", index=False)

    (OUT / "metadata.json").write_text(json.dumps({
        "train_n": len(train),
        "holdout_n": len(test),
        "target": TARGET,
        "units": "dB/m",
        "features": RAW_FEATURES,
        "engineered_features": ENGINEERED,
        "cv": "KFold(n_splits=5, shuffle=True, random_state=42)",
        "holdout_used_for_selection": False,
        "note": "MLP is a neural-network regression baseline; no holdout evaluation."
    }, indent=2))

    print("\nMLP CV results:")
    print(pd.DataFrame([result]).to_string(index=False))
    print(f"Saved results to: {OUT}", flush=True)


if __name__ == "__main__":
    main()
