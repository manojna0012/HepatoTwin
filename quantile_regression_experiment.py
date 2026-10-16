
"""Compare baseline XGBoost with quantile XGBoost for CAP prediction.

Run from the HepatoTwin project root:
    python quantile_regression_experiment.py

Outputs are saved to a new experiment directory.
Existing models and experiment results are not overwritten.
"""

import json

import numpy as np
import pandas as pd
import xgboost

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor

from src.cap_features import (
    CAPFeatures,
    RAW_NUMERIC,
    ENGINEERED,
    RAW_FEATURES,
)
from src.train_final_model import ROOT, TARGET, make_pipeline


SEED = 42
N_SPLITS = 5

PARAMS = dict(
    n_estimators=600,
    learning_rate=0.02,
    max_depth=4,
    min_child_weight=8,
    subsample=0.8,
    colsample_bytree=0.6,
    reg_lambda=3.0,
    reg_alpha=0.5,
)

OUTPUT_DIR = ROOT / "data/results/experiments/quantile_regression"


def build_baseline():
    """Reference squared-error XGBoost pipeline."""
    inner = make_pipeline(
        params=PARAMS,
        numeric=RAW_NUMERIC + ENGINEERED,
        add_indicator=True,
    )
    return Pipeline([
        ("features", CAPFeatures()),
        ("regressor", inner),
    ])


def build_quantile_model(alpha):
    """XGBoost model for one conditional quantile."""
    estimator = XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=alpha,
        random_state=SEED,
        n_jobs=2,
        **PARAMS,
    )

    inner = make_pipeline(
        numeric=RAW_NUMERIC + ENGINEERED,
        add_indicator=True,
        estimator=estimator,
    )

    return Pipeline([
        ("features", CAPFeatures()),
        ("regressor", inner),
    ])


def calculate_metrics(y, prediction):
    return {
        "R2": float(r2_score(y, prediction)),
        "RMSE": float(np.sqrt(mean_squared_error(y, prediction))),
        "MAE": float(mean_absolute_error(y, prediction)),
        "Bias_pred_minus_actual": float(np.mean(prediction - y)),
    }


def pinball_loss(y, prediction, alpha):
    """Mean quantile (pinball) loss."""
    error = y - prediction
    return float(np.mean(np.maximum(alpha * error, (alpha - 1) * error)))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    source = ROOT / "data/processed/nhanes_merged.csv"
    previous_dir = ROOT / "data/results/second_pass"
    oof_path = previous_dir / "oof_predictions.csv"

    # IMPORTANT FIX:
    # Read the full dataset so the extra lab columns needed by CAPFeatures
    # are retained. Apply the same cohort eligibility filters as before.
    cohort = pd.read_csv(source)

    required = ["SEQN", "LUAXSTAT", TARGET] + RAW_FEATURES
    missing = sorted(set(required) - set(cohort.columns))
    if missing:
        raise ValueError(
            "The source CSV is missing required columns: "
            + ", ".join(missing)
        )

    cohort = cohort.loc[
        cohort["LUAXSTAT"].eq(1) & cohort[TARGET].notna()
    ].copy()

    if cohort.empty:
        raise ValueError("No eligible rows remain after cohort filtering.")

    if cohort["SEQN"].isna().any() or cohort["SEQN"].duplicated().any():
        raise ValueError("Cohort must have unique, nonmissing SEQN values.")

    cohort = cohort.set_index("SEQN")

    old_oof = pd.read_csv(oof_path)

    if not {"SEQN", "Actual_CAP"}.issubset(old_oof.columns):
        raise ValueError(
            "Existing OOF file must contain SEQN and Actual_CAP columns."
        )

    if old_oof["SEQN"].duplicated().any():
        raise ValueError("Duplicate SEQN values found in existing OOF file.")

    missing_ids = set(old_oof["SEQN"]) - set(cohort.index)
    if missing_ids:
        raise ValueError(
            f"{len(missing_ids)} OOF participant IDs are absent from the cohort."
        )

    # Preserve exactly the existing OOF participant order and folds.
    train = cohort.loc[old_oof["SEQN"].to_numpy()].copy()
    X = train[RAW_FEATURES]
    y = train[TARGET].to_numpy(dtype=float)

    np.testing.assert_allclose(
        y,
        old_oof["Actual_CAP"].to_numpy(dtype=float),
        rtol=0,
        atol=1e-8,
        err_msg="CAP targets do not match the existing OOF file.",
    )

    cv = list(
        KFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=SEED,
        ).split(X)
    )

    model_names = [
        "baseline_xgb",
        "quantile_q10",
        "quantile_q50",
        "quantile_q90",
    ]

    predictions = {
        name: np.full(len(X), np.nan)
        for name in model_names
    }

    fold_rows = []

    with threadpool_limits(limits=2):
        for fold, (train_idx, valid_idx) in enumerate(cv, start=1):
            X_train = X.iloc[train_idx]
            X_valid = X.iloc[valid_idx]
            y_train = y[train_idx]
            y_valid = y[valid_idx]

            print(f"\nFold {fold}/{N_SPLITS}")

            # Baseline squared-error model.
            baseline = build_baseline()
            baseline.fit(X_train, y_train)
            predictions["baseline_xgb"][valid_idx] = baseline.predict(X_valid)

            # Lower, median, and upper conditional quantiles.
            for name, alpha in [
                ("quantile_q10", 0.10),
                ("quantile_q50", 0.50),
                ("quantile_q90", 0.90),
            ]:
                model = build_quantile_model(alpha)
                model.fit(X_train, y_train)
                predictions[name][valid_idx] = model.predict(X_valid)

            for name in model_names:
                fold_pred = predictions[name][valid_idx]
                fold_rows.append({
                    "fold": fold,
                    "model": name,
                    **calculate_metrics(y_valid, fold_pred),
                })

            print("  Baseline and all quantile models fitted.")

    # Save pooled out-of-fold predictions.
    oof = pd.DataFrame({
        "SEQN": old_oof["SEQN"],
        "Actual_CAP": y,
        **predictions,
    })

    lower = oof["quantile_q10"].to_numpy()
    median = oof["quantile_q50"].to_numpy()
    upper = oof["quantile_q90"].to_numpy()

    # Do not reorder the quantiles silently; report crossing explicitly.
    oof["q10_q50_crossing"] = lower > median
    oof["q50_q90_crossing"] = median > upper

    # Interval coverage and width use the raw q10/q90 predictions.
    interval_coverage = np.mean((y >= lower) & (y <= upper))
    interval_width = np.mean(upper - lower)

    summary_rows = []
    for name in model_names:
        pred = predictions[name]
        summary_rows.append({
            "model": name,
            **calculate_metrics(y, pred),
        })

    summary = pd.DataFrame(summary_rows)
    fold_metrics = pd.DataFrame(fold_rows)

    interval_summary = {
        "n_rows": int(len(y)),
        "nominal_interval_coverage": 0.80,
        "empirical_q10_q90_coverage": float(interval_coverage),
        "mean_q10_q90_interval_width": float(interval_width),
        "q10_pinball_loss": pinball_loss(y, lower, 0.10),
        "q50_pinball_loss": pinball_loss(y, median, 0.50),
        "q90_pinball_loss": pinball_loss(y, upper, 0.90),
        "q10_q50_crossing_count": int((lower > median).sum()),
        "q50_q90_crossing_count": int((median > upper).sum()),
        "xgboost_version": xgboost.__version__,
        "seed": SEED,
        "folds": N_SPLITS,
        "params": PARAMS,
    }

    oof.to_csv(OUTPUT_DIR / "oof_predictions.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "summary_metrics.csv", index=False)
    fold_metrics.to_csv(OUTPUT_DIR / "fold_metrics.csv", index=False)

    with open(
        OUTPUT_DIR / "interval_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(interval_summary, file, indent=2)

    print("\n=== Pooled out-of-fold metrics ===")
    print(summary.to_string(
        index=False,
        float_format=lambda value: f"{value:.5f}",
    ))

    print("\n=== Quantile interval diagnostics ===")
    print(json.dumps(interval_summary, indent=2))

    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
