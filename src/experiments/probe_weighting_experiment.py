
"""Test whether weighting high-CAP training observations helps.

The existing probe-aware model is the unweighted reference.
All models use identical five-fold splits and XGBoost parameters.
No existing model artifact is modified.
"""

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import KFold

from src.train_final_model import ROOT, TARGET
from src.cap_features import RAW_FEATURES

# Import the experiment's established model builders.
from src.experiments.probe_aware_regression import (
    make_probe_aware,
    normalize_probe,
)


DATA_PATH = ROOT / "data" / "processed" / "nhanes_merged.csv"
REFERENCE_OOF = (
    ROOT / "data" / "results" / "second_pass" / "oof_predictions.csv"
)
OUTPUT_DIR = (
    ROOT / "data" / "results" / "experiments" / "probe_weighting"
)

N_SPLITS = 5
SEED = 42

# Exploratory settings: emphasize high-CAP observations in training only.
HIGH_CAP_THRESHOLD = 320
WEIGHT_STRENGTHS = [1.5, 2.0]


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "mean_prediction_bias": float(np.mean(y_pred - y_true)),
    }


def load_cohort():
    data = pd.read_csv(DATA_PATH)

    required = [
        "SEQN", "LUAXSTAT", TARGET, "LUAPNME", *RAW_FEATURES
    ]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = data.loc[
        data["LUAXSTAT"].eq(1) & data[TARGET].notna()
    ].copy()

    if data["SEQN"].isna().any() or data["SEQN"].duplicated().any():
        raise ValueError("SEQN must be unique and nonmissing.")

    # Match the established 7,216-person OOF order if available.
    if REFERENCE_OOF.exists():
        old = pd.read_csv(REFERENCE_OOF)

        if "SEQN" in old.columns:
            ids = old["SEQN"].tolist()

            if old["SEQN"].is_unique and set(ids).issubset(set(data["SEQN"])):
                data = data.set_index("SEQN", drop=False).loc[ids].reset_index(
                    drop=True
                )

                actual_col = (
                    "Actual_CAP"
                    if "Actual_CAP" in old.columns
                    else "actual_CAP"
                    if "actual_CAP" in old.columns
                    else None
                )

                if actual_col:
                    np.testing.assert_allclose(
                        data[TARGET].to_numpy(dtype=float),
                        old[actual_col].to_numpy(dtype=float),
                        rtol=0,
                        atol=1e-8,
                    )

                print(f"Using reference OOF cohort: {len(data)} participants")
            else:
                print("Reference OOF IDs did not match; using full eligible cohort.")
    else:
        print("Reference OOF file not found; using full eligible cohort.")

    return data.reset_index(drop=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_cohort()
    X = data[RAW_FEATURES + ["LUAPNME"]].copy()
    y = data[TARGET].astype(float).to_numpy()

    # Reference model plus two weighted candidates.
    model_names = ["probe_unweighted"] + [
        f"probe_weight_{strength:g}x" for strength in WEIGHT_STRENGTHS
    ]

    oof_predictions = {
        name: np.full(len(data), np.nan, dtype=float)
        for name in model_names
    }

    cv = list(
        KFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=SEED,
        ).split(X)
    )

    fold_rows = []

    print(f"\nParticipants: {len(data)}")
    print(f"High-CAP threshold: {HIGH_CAP_THRESHOLD}")
    print(f"Weight strengths: {WEIGHT_STRENGTHS}")
    print("Starting five-fold evaluation...\n")

    for fold, (train_idx, valid_idx) in enumerate(cv, start=1):
        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]
        y_train = y[train_idx]
        y_valid = y[valid_idx]

        print(f"Fold {fold}/{N_SPLITS}", flush=True)

        # Unweighted reference.
        reference = clone(make_probe_aware())
        reference.fit(X_train, y_train)
        pred = reference.predict(X_valid)
        oof_predictions["probe_unweighted"][valid_idx] = pred

        score = metrics(y_valid, pred)
        fold_rows.append({
            "fold": fold,
            "model": "probe_unweighted",
            "participants": len(valid_idx),
            **score,
        })
        print(
            f"  Unweighted: R2={score['R2']:.4f}, "
            f"RMSE={score['RMSE']:.3f}, MAE={score['MAE']:.3f}"
        )

        # Weighted candidates: weights are based only on training targets.
        high_mask = y_train >= HIGH_CAP_THRESHOLD

        for strength in WEIGHT_STRENGTHS:
            name = f"probe_weight_{strength:g}x"

            sample_weights = np.ones(len(y_train), dtype=float)
            sample_weights[high_mask] = strength

            model = clone(make_probe_aware())
            model.fit(
                X_train,
                y_train,
                model__model__sample_weight=sample_weights,
            )

            pred = model.predict(X_valid)
            oof_predictions[name][valid_idx] = pred

            score = metrics(y_valid, pred)
            fold_rows.append({
                "fold": fold,
                "model": name,
                "participants": len(valid_idx),
                **score,
            })

            print(
                f"  {name}: R2={score['R2']:.4f}, "
                f"RMSE={score['RMSE']:.3f}, MAE={score['MAE']:.3f}"
            )

    # Overall pooled out-of-fold metrics.
    summary_rows = []
    for name, pred in oof_predictions.items():
        summary_rows.append({
            "model": name,
            "participants": len(y),
            **metrics(y, pred),
        })

    summary = pd.DataFrame(summary_rows)
    folds = pd.DataFrame(fold_rows)

    summary.to_csv(OUTPUT_DIR / "summary_metrics.csv", index=False)
    folds.to_csv(OUTPUT_DIR / "fold_metrics.csv", index=False)

    # Store all predictions for subsequent subgroup analysis.
    pred_frame = pd.DataFrame({
        "SEQN": data["SEQN"].to_numpy(),
        "actual_CAP": y,
        "LUAPNME": data["LUAPNME"].to_numpy(),
    })

    for name, pred in oof_predictions.items():
        pred_frame[f"{name}_prediction"] = pred

    pred_frame.to_csv(OUTPUT_DIR / "oof_predictions.csv", index=False)

    # Subgroup errors by probe and CAP band.
    pred_frame["probe_type"] = data["LUAPNME"].map(
        lambda value: (
            "M" if normalize_probe(value) == 0.0
            else "XL" if normalize_probe(value) == 1.0
            else "Unknown"
        )
    )

    bins = [-np.inf, 199, 319, 359, 389, np.inf]
    labels = ["<=199", "200-319", "320-359", "360-389", ">=390"]

    pred_frame["CAP_band"] = pd.cut(
        pred_frame["actual_CAP"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    subgroup_rows = []

    for (probe, band), group in pred_frame.groupby(
        ["probe_type", "CAP_band"],
        observed=True,
        dropna=False,
    ):
        for name in model_names:
            pred_col = f"{name}_prediction"
            subgroup_rows.append({
                "probe_type": probe,
                "CAP_band": str(band),
                "model": name,
                "participants": len(group),
                **metrics(group["actual_CAP"], group[pred_col]),
            })

    pd.DataFrame(subgroup_rows).to_csv(
        OUTPUT_DIR / "subgroup_metrics.csv",
        index=False,
    )

    print("\n=== Pooled OOF comparison ===")
    print(summary.to_string(index=False))

    print("\n=== Results saved to ===")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
