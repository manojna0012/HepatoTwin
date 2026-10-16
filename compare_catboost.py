from pathlib import Path
import json
import importlib.util

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)
from xgboost import XGBRegressor

from src.train_final_model import ROOT, TARGET, make_pipeline
from src.cap_features import CAPFeatures, RAW_FEATURES, RAW_NUMERIC, ENGINEERED


# ============================================================
# Configuration
# ============================================================

DATA_PATH = ROOT / "data" / "processed" / "nhanes_merged.csv"
OUTPUT_DIR = ROOT / "data" / "results" / "experiments" / "catboost_comparison"

OUTER_FOLDS = 5
SEED = 42
BASE_NUMERIC = list(dict.fromkeys(RAW_NUMERIC + ENGINEERED))


def make_xgb_pipeline():
    """Current XGBoost baseline, reproduced for identical CV folds."""
    estimator = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=600,
        learning_rate=0.02,
        max_depth=4,
        min_child_weight=8,
        subsample=0.8,
        colsample_bytree=0.6,
        reg_lambda=3,
        reg_alpha=0.5,
        random_state=SEED,
        n_jobs=-1,
    )

    preprocessing = make_pipeline(
        numeric=BASE_NUMERIC,
        estimator=estimator,
        add_indicator=True,
    )

    return Pipeline([
        ("features", CAPFeatures()),
        ("model", preprocessing),
    ])


def make_catboost_pipeline():
    """CatBoost model using the same engineered and preprocessed inputs."""
    if importlib.util.find_spec("catboost") is None:
        raise ImportError(
            "CatBoost is not installed. Install it inside your active venv "
            "with: python -m pip install catboost"
        )

    from catboost import CatBoostRegressor

    estimator = CatBoostRegressor(
        iterations=1200,
        learning_rate=0.03,
        depth=6,
        loss_function="RMSE",
        l2_leaf_reg=5,
        random_seed=SEED,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )

    preprocessing = make_pipeline(
        numeric=BASE_NUMERIC,
        estimator=estimator,
        add_indicator=True,
    )

    return Pipeline([
        ("features", CAPFeatures()),
        ("model", preprocessing),
    ])


def metrics(y_true, y_pred):
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mean_prediction_bias": float(np.mean(y_pred - y_true)),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    required = ["SEQN", "LUAXSTAT", TARGET] + list(RAW_FEATURES)
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    cohort = df.loc[
        (df["LUAXSTAT"] == 1) & df[TARGET].notna()
    ].copy()

    if cohort["SEQN"].duplicated().any():
        raise ValueError("Duplicate SEQN values found in eligible cohort.")

    cohort = cohort.reset_index(drop=True)
    X = cohort[RAW_FEATURES].copy()
    y = cohort[TARGET].astype(float).to_numpy()

    print(f"Eligible participants: {len(cohort)}")
    print(f"Target: {TARGET}")
    print(f"Outer folds: {OUTER_FOLDS}")
    print("Comparing XGBoost and CatBoost on identical folds.")

    splitter = KFold(
        n_splits=OUTER_FOLDS,
        shuffle=True,
        random_state=SEED,
    )

    model_names = ["xgboost_baseline", "catboost"]
    oof_predictions = {
        name: np.full(len(cohort), np.nan)
        for name in model_names
    }
    fold_rows = []

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X), start=1):
        print(f"\nFold {fold}/{OUTER_FOLDS}")

        X_train = X.iloc[train_idx].copy()
        X_valid = X.iloc[valid_idx].copy()
        y_train = y[train_idx]
        y_valid = y[valid_idx]

        models = {
            "xgboost_baseline": make_xgb_pipeline(),
            "catboost": make_catboost_pipeline(),
        }

        for name, model in models.items():
            print(f"  Training {name}...")
            model.fit(X_train, y_train)
            pred = model.predict(X_valid)
            oof_predictions[name][valid_idx] = pred

            fold_metric = metrics(y_valid, pred)
            fold_rows.append({
                "fold": fold,
                "model": name,
                **fold_metric,
            })
            print(
                f"    R2={fold_metric['r2']:.4f}, "
                f"RMSE={fold_metric['rmse']:.3f}, "
                f"MAE={fold_metric['mae']:.3f}"
            )

    summary_rows = []
    for name in model_names:
        summary_rows.append({
            "model": name,
            "participants": len(y),
            **metrics(y, oof_predictions[name]),
        })

    summary = pd.DataFrame(summary_rows)
    fold_results = pd.DataFrame(fold_rows)

    summary.to_csv(OUTPUT_DIR / "summary_metrics.csv", index=False)
    fold_results.to_csv(OUTPUT_DIR / "fold_metrics.csv", index=False)

    predictions_df = pd.DataFrame({
        "SEQN": cohort["SEQN"].to_numpy(),
        "actual_CAP": y,
        **oof_predictions,
    })
    predictions_df.to_csv(
        OUTPUT_DIR / "oof_predictions.csv", index=False
    )

    metadata = {
        "target": TARGET,
        "eligible_participants": int(len(cohort)),
        "outer_folds": OUTER_FOLDS,
        "random_seed": SEED,
        "models": model_names,
        "features": "Same raw inputs and CAPFeatures engineering",
        "evaluation": "Pooled out-of-fold predictions from identical folds",
        "note": (
            "Internal cross-validation, not external validation. "
            "CatBoost and XGBoost use the same outer folds and feature set."
        ),
    }
    with open(OUTPUT_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n========== POOLED OOF RESULTS ==========")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    xgb_r2 = summary.loc[
        summary["model"] == "xgboost_baseline", "r2"
    ].iloc[0]
    cat_r2 = summary.loc[
        summary["model"] == "catboost", "r2"
    ].iloc[0]

    print(f"\nCatBoost R2 change vs XGBoost: {cat_r2 - xgb_r2:+.4f}")
    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
