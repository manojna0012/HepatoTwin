
"""Compare the existing CAP regression baseline with a probe-aware model.

Outputs are written to a new experiment directory. Existing model artifacts
and baseline files are not modified.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.train_final_model import ROOT, TARGET, make_pipeline
from src.cap_features import (
    CAPFeatures,
    RAW_FEATURES,
    RAW_NUMERIC,
    ENGINEERED,
)


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

DATA_PATH = ROOT / "data" / "processed" / "nhanes_merged.csv"
REFERENCE_OOF = ROOT / "data" / "results" / "second_pass" / "oof_predictions.csv"
OUTPUT_DIR = ROOT / "data" / "results" / "experiments" / "probe_aware_model"

N_SPLITS = 5
SEED = 42

BASE_NUMERIC = list(dict.fromkeys(RAW_NUMERIC + ENGINEERED))

XGB_PARAMS = {
    "n_estimators": 600,
    "learning_rate": 0.02,
    "max_depth": 4,
    "min_child_weight": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "reg_lambda": 3,
    "reg_alpha": 0.5,
}


# ------------------------------------------------------------
# Feature transformer for the probe-aware model
# ------------------------------------------------------------

def normalize_probe(value):
    """Normalize common CSV representations of M and XL."""
    if pd.isna(value):
        return np.nan

    value = str(value).strip().upper()

    # Handles values stored as strings such as "b'M'" or "b'XL'".
    value = value.replace("B'", "").replace('"', "").replace("'", "")
    value = value.strip()

    if value == "M":
        return 0.0
    if value == "XL":
        return 1.0

    return np.nan


class ProbeAwareFeatures(TransformerMixin, BaseEstimator):
    """Apply existing CAP features and append a binary probe indicator."""

    def fit(self, X, y=None):
        self.cap_features_ = CAPFeatures()
        self.cap_features_.fit(X[RAW_FEATURES], y)
        return self

    def transform(self, X):
        transformed = self.cap_features_.transform(X[RAW_FEATURES]).copy()

        # M = 0, XL = 1; missing/unknown values are imputed downstream.
        transformed["probe_is_XL"] = (
            X["LUAPNME"].map(normalize_probe).astype(float)
        )

        return transformed


# ------------------------------------------------------------
# Models
# ------------------------------------------------------------

def make_regressor(extra_numeric=None):
    """Build the same XGBoost model and preprocessing as the baseline."""
    numeric_columns = list(BASE_NUMERIC)

    if extra_numeric:
        numeric_columns.extend(extra_numeric)

    estimator = XGBRegressor(
        objective="reg:squarederror",
        random_state=SEED,
        n_jobs=2,
        **XGB_PARAMS,
    )

    return make_pipeline(
        numeric=numeric_columns,
        estimator=estimator,
        add_indicator=True,
    )


def make_baseline():
    """Existing engineered baseline: no probe feature."""
    return Pipeline([
        ("features", CAPFeatures()),
        ("model", make_regressor()),
    ])


def make_probe_aware():
    """Same regression model, with probe type added as an input."""
    return Pipeline([
        ("features", ProbeAwareFeatures()),
        ("model", make_regressor(extra_numeric=["probe_is_XL"])),
    ])


# ------------------------------------------------------------
# Metrics and residual analysis
# ------------------------------------------------------------

def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    return {
        "R2": float(r2_score(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "MAE": float(mean_absolute_error(actual, predicted)),
        "mean_prediction_bias": float(np.mean(predicted - actual)),
    }


def load_cohort():
    data = pd.read_csv(DATA_PATH)

    required = [
        "SEQN",
        "LUAXSTAT",
        TARGET,
        "LUAPNME",
        *RAW_FEATURES,
    ]
    missing = sorted(set(required) - set(data.columns))

    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    data = data.loc[
        data["LUAXSTAT"].eq(1) & data[TARGET].notna()
    ].copy()

    if data["SEQN"].isna().any() or data["SEQN"].duplicated().any():
        raise ValueError("Eligible cohort must have unique, nonmissing SEQN.")

    # Match the participant order used by the earlier experiment when possible.
    # This ensures a fair comparison against the existing 7,216-person OOF run.
    if REFERENCE_OOF.exists():
        previous = pd.read_csv(REFERENCE_OOF)

        if "SEQN" in previous.columns:
            eligible_ids = set(data["SEQN"])
            requested_ids = previous["SEQN"].tolist()

            if (
                previous["SEQN"].is_unique
                and set(requested_ids).issubset(eligible_ids)
            ):
                indexed = data.set_index("SEQN", drop=False)
                data = indexed.loc[requested_ids].reset_index(drop=True)

                if "Actual_CAP" in previous.columns:
                    np.testing.assert_allclose(
                        data[TARGET].to_numpy(dtype=float),
                        previous["Actual_CAP"].to_numpy(dtype=float),
                        rtol=0,
                        atol=1e-8,
                    )
                elif "actual_CAP" in previous.columns:
                    np.testing.assert_allclose(
                        data[TARGET].to_numpy(dtype=float),
                        previous["actual_CAP"].to_numpy(dtype=float),
                        rtol=0,
                        atol=1e-8,
                    )

                print(
                    "Using participant order from existing OOF predictions:",
                    len(data),
                )
            else:
                print(
                    "Warning: reference OOF IDs did not match the eligible "
                    "cohort. Using the full eligible cohort instead."
                )
    else:
        print(
            "No reference OOF file found; using the full eligible cohort."
        )

    return data.reset_index(drop=True)


def save_probe_residuals(predictions):
    """Summarize errors by probe type and actual CAP band."""
    data = predictions.copy()
    data["probe_type"] = data["LUAPNME"].map(
        lambda value: (
            "M" if normalize_probe(value) == 0.0
            else "XL" if normalize_probe(value) == 1.0
            else "Unknown"
        )
    )

    bins = [-np.inf, 199, 247, 279, 319, 359, 389, np.inf]
    labels = [
        "<=199",
        "200-247",
        "248-279",
        "280-319",
        "320-359",
        "360-389",
        ">=390",
    ]

    data["CAP_band"] = pd.cut(
        data["actual_CAP"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    rows = []

    for (probe, cap_band), group in data.groupby(
        ["probe_type", "CAP_band"],
        observed=True,
        dropna=False,
    ):
        row = {
            "probe_type": probe,
            "CAP_band": str(cap_band),
            "participants": len(group),
        }

        for model_name in ["baseline_prediction", "probe_prediction"]:
            metrics = calculate_metrics(
                group["actual_CAP"],
                group[model_name],
            )
            prefix = "baseline" if model_name == "baseline_prediction" else "probe"
            row[f"{prefix}_R2"] = metrics["R2"]
            row[f"{prefix}_RMSE"] = metrics["RMSE"]
            row[f"{prefix}_MAE"] = metrics["MAE"]
            row[f"{prefix}_bias"] = metrics["mean_prediction_bias"]

        rows.append(row)

    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / "probe_residuals.csv",
        index=False,
    )


# ------------------------------------------------------------
# Main experiment
# ------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cohort = load_cohort()

    X_baseline = cohort[RAW_FEATURES].copy()
    X_probe = cohort[RAW_FEATURES + ["LUAPNME"]].copy()
    y = cohort[TARGET].astype(float).to_numpy()

    models = {
        "baseline_xgb": make_baseline(),
        "probe_aware_xgb": make_probe_aware(),
    }

    cv = list(
        KFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=SEED,
        ).split(X_baseline)
    )

    predictions = {
        name: np.full(len(cohort), np.nan, dtype=float)
        for name in models
    }

    fold_rows = []

    print(f"\nParticipants: {len(cohort)}")
    print(f"Cross-validation: {N_SPLITS}-fold, seed={SEED}")
    print("Running baseline and probe-aware models...\n")

    for fold_number, (train_idx, valid_idx) in enumerate(cv, start=1):
        print(f"Fold {fold_number}/{N_SPLITS}", flush=True)

        for name, template in models.items():
            model = clone(template)

            if name == "baseline_xgb":
                X_train = X_baseline.iloc[train_idx]
                X_valid = X_baseline.iloc[valid_idx]
            else:
                X_train = X_probe.iloc[train_idx]
                X_valid = X_probe.iloc[valid_idx]

            model.fit(X_train, y[train_idx])
            pred = model.predict(X_valid)
            predictions[name][valid_idx] = pred

            fold_score = calculate_metrics(y[valid_idx], pred)

            fold_rows.append({
                "fold": fold_number,
                "model": name,
                "participants": len(valid_idx),
                **fold_score,
            })

            print(
                f"  {name}: R2={fold_score['R2']:.4f}, "
                f"RMSE={fold_score['RMSE']:.3f}, "
                f"MAE={fold_score['MAE']:.3f}",
                flush=True,
            )

    # Pooled out-of-fold metrics: every participant is evaluated when held out.
    summary_rows = []

    for name, pred in predictions.items():
        summary_rows.append({
            "model": name,
            "participants": len(y),
            **calculate_metrics(y, pred),
        })

    summary = pd.DataFrame(summary_rows)
    folds = pd.DataFrame(fold_rows)

    prediction_frame = pd.DataFrame({
        "SEQN": cohort["SEQN"].to_numpy(),
        "actual_CAP": y,
        "LUAPNME": cohort["LUAPNME"].to_numpy(),
        "baseline_prediction": predictions["baseline_xgb"],
        "probe_prediction": predictions["probe_aware_xgb"],
    })

    summary.to_csv(OUTPUT_DIR / "summary_metrics.csv", index=False)
    folds.to_csv(OUTPUT_DIR / "fold_metrics.csv", index=False)
    prediction_frame.to_csv(
        OUTPUT_DIR / "oof_predictions.csv",
        index=False,
    )

    save_probe_residuals(prediction_frame)

    print("\n=== Pooled OOF comparison ===")
    print(summary.to_string(index=False))

    baseline_r2 = summary.loc[
        summary["model"].eq("baseline_xgb"), "R2"
    ].iloc[0]
    probe_r2 = summary.loc[
        summary["model"].eq("probe_aware_xgb"), "R2"
    ].iloc[0]

    print(f"\nChange in R2: {probe_r2 - baseline_r2:+.5f}")
    print(f"\nResults saved to:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
