from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)

from src.train_final_model import ROOT, TARGET


# ============================================================
# Configuration
# ============================================================

DATA_PATH = ROOT / "data" / "processed" / "nhanes_merged.csv"
BASELINE_PREDICTIONS = (
    ROOT / "data" / "results" / "experiments"
    / "catboost_comparison" / "oof_predictions.csv"
)
OUTPUT_DIR = (
    ROOT / "data" / "results" / "experiments"
    / "prediction_calibration"
)

N_SPLITS = 5
SEED = 42


def calculate_metrics(y_true, y_pred):
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mean_prediction_bias": float(np.mean(y_pred - y_true)),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load predictions generated out-of-fold by the baseline model.
    predictions = pd.read_csv(BASELINE_PREDICTIONS)

    required_pred = ["SEQN", "actual_CAP", "xgboost_baseline"]
    missing = [c for c in required_pred if c not in predictions.columns]
    if missing:
        raise ValueError(
            f"Prediction file is missing columns: {missing}. "
            "Expected the output from compare_catboost.py."
        )

    if predictions["SEQN"].duplicated().any():
        raise ValueError("Duplicate SEQN values found in prediction file.")

    source = pd.read_csv(DATA_PATH, usecols=["SEQN", "LUAXSTAT", TARGET])
    eligible = source.loc[
        (source["LUAXSTAT"] == 1) & source[TARGET].notna()
    ].copy()

    if eligible["SEQN"].duplicated().any():
        raise ValueError("Duplicate SEQN values found in source cohort.")

    merged = eligible.merge(
        predictions[["SEQN", "actual_CAP", "xgboost_baseline"]],
        on="SEQN",
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(eligible):
        print(
            f"Warning: matched {len(merged)} of {len(eligible)} eligible "
            "participants. Continuing with matched rows."
        )

    if merged.empty:
        raise ValueError("No participants matched between source and predictions.")

    if not np.allclose(
        merged[TARGET].to_numpy(float),
        merged["actual_CAP"].to_numpy(float),
        equal_nan=True,
    ):
        raise ValueError("Saved actual_CAP values do not match source data.")

    merged = merged.dropna(
        subset=["actual_CAP", "xgboost_baseline"]
    ).reset_index(drop=True)

    y = merged["actual_CAP"].to_numpy(float)
    pred = merged["xgboost_baseline"].to_numpy(float)

    # Baseline pooled OOF metrics.
    baseline_metrics = calculate_metrics(y, pred)

    # Cross-fit the calibration layer:
    # fit a and b on other folds' OOF predictions, evaluate on the held-out fold.
    # This avoids fitting calibration on the same rows being evaluated.
    splitter = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SEED,
    )

    calibrated_oof = np.full(len(merged), np.nan)
    calibration_rows = []

    X_pred = pred.reshape(-1, 1)

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X_pred), start=1):
        calibrator = LinearRegression()
        calibrator.fit(X_pred[train_idx], y[train_idx])

        fold_calibrated = calibrator.predict(X_pred[valid_idx])
        calibrated_oof[valid_idx] = fold_calibrated

        fold_metrics = calculate_metrics(y[valid_idx], fold_calibrated)
        calibration_rows.append({
            "fold": fold,
            "intercept": float(calibrator.intercept_),
            "slope": float(calibrator.coef_[0]),
            **fold_metrics,
        })

        print(
            f"Calibration fold {fold}/{N_SPLITS}: "
            f"intercept={calibrator.intercept_:.4f}, "
            f"slope={calibrator.coef_[0]:.4f}, "
            f"R2={fold_metrics['r2']:.4f}"
        )

    calibrated_metrics = calculate_metrics(y, calibrated_oof)

    # Compare on exactly the same pooled participants.
    summary = pd.DataFrame([
        {"model": "xgboost_uncalibrated", **baseline_metrics},
        {"model": "xgboost_cross_fitted_calibration", **calibrated_metrics},
    ])

    summary.to_csv(OUTPUT_DIR / "summary_metrics.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(
        OUTPUT_DIR / "fold_metrics.csv", index=False
    )

    output_predictions = merged[["SEQN", "actual_CAP"]].copy()
    output_predictions["xgboost_oof"] = pred
    output_predictions["calibrated_oof"] = calibrated_oof
    output_predictions["uncalibrated_residual"] = pred - y
    output_predictions["calibrated_residual"] = calibrated_oof - y
    output_predictions.to_csv(
        OUTPUT_DIR / "oof_predictions.csv", index=False
    )

    metadata = {
        "target": TARGET,
        "participants": int(len(merged)),
        "calibration": "Linear regression fit on other folds' OOF predictions",
        "folds": N_SPLITS,
        "random_seed": SEED,
        "source_predictions": str(BASELINE_PREDICTIONS),
        "note": (
            "This evaluates a cross-fitted calibration layer on existing "
            "OOF predictions. It does not retrain the underlying XGBoost "
            "model in each calibration fold. Treat this as a calibration "
            "diagnostic; final validation should repeat the full procedure "
            "inside outer training folds."
        ),
    }
    with open(OUTPUT_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n========== CALIBRATION COMPARISON ==========")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\nCalibration slope/intercept by fold:")
    print(pd.DataFrame(calibration_rows)[
        ["fold", "intercept", "slope"]
    ].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
