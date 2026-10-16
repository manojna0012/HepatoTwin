
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

PREDICTIONS_PATH = (
    ROOT / "data" / "results" / "experiments"
    / "hybrid_regression" / "oof_predictions.csv"
)

OUTPUT_DIR = (
    ROOT / "data" / "results" / "experiments"
    / "residual_analysis"
)

CEILING = 400.0
NEAR_CEILING = 390.0


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1. Load source measurements and OOF predictions
    # --------------------------------------------------------

    df = pd.read_csv(DATA_PATH)

    required = ["SEQN", "LUAXSTAT", TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing source columns: {missing}")

    eligible = df.loc[
        (df["LUAXSTAT"] == 1) & df[TARGET].notna(),
        ["SEQN", TARGET],
    ].copy()

    if eligible["SEQN"].duplicated().any():
        raise ValueError("Duplicate SEQN values in eligible cohort.")

    pred = pd.read_csv(PREDICTIONS_PATH)

    required_pred = [
        "SEQN",
        "actual_CAP",
        "baseline_xgboost",
    ]
    missing = [c for c in required_pred if c not in pred.columns]
    if missing:
        raise ValueError(
            f"Prediction file is missing columns: {missing}"
        )

    if pred["SEQN"].duplicated().any():
        raise ValueError("Duplicate SEQN values in predictions.")

    # Match source target values to prediction rows by participant ID.
    merged = eligible.merge(
        pred[["SEQN", "actual_CAP", "baseline_xgboost"]],
        on="SEQN",
        how="inner",
        validate="one_to_one",
    )

    if merged.empty:
        raise ValueError(
            "No matching participants between source and prediction files."
        )

    # Check that saved predictions align with the source target.
    mismatch = ~np.isclose(
        merged[TARGET].to_numpy(dtype=float),
        merged["actual_CAP"].to_numpy(dtype=float),
        equal_nan=True,
    )
    if mismatch.any():
        raise ValueError(
            f"{int(mismatch.sum())} target values disagree between "
            "source data and prediction file."
        )

    merged = merged.dropna(
        subset=[TARGET, "baseline_xgboost"]
    ).copy()

    merged["residual"] = (
        merged["baseline_xgboost"] - merged[TARGET]
    )
    merged["absolute_error"] = merged["residual"].abs()
    merged["squared_error"] = merged["residual"] ** 2

    actual = merged[TARGET].to_numpy(dtype=float)
    predicted = merged["baseline_xgboost"].to_numpy(dtype=float)

    # --------------------------------------------------------
    # 2. Overall metrics
    # --------------------------------------------------------

    overall = {
        "participants": int(len(merged)),
        "r2": float(r2_score(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae": float(mean_absolute_error(actual, predicted)),
        "mean_prediction_bias": float(np.mean(predicted - actual)),
    }

    print("\n========== OVERALL OOF PERFORMANCE ==========")
    for key, value in overall.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    with open(
        OUTPUT_DIR / "overall_metrics.json", "w", encoding="utf-8"
    ) as f:
        json.dump(overall, f, indent=2)

    # --------------------------------------------------------
    # 3. Target distribution and ceiling concentration
    # --------------------------------------------------------

    cap = merged[TARGET]

    distribution = {
        "valid_measurements": int(cap.notna().sum()),
        "exactly_400": int((cap == CEILING).sum()),
        "above_390": int((cap > NEAR_CEILING).sum()),
        "above_395": int((cap > 395).sum()),
        "at_or_above_399": int((cap >= 399).sum()),
        "max_value": float(cap.max()),
        "percent_exactly_400": float((cap == CEILING).mean() * 100),
        "percent_above_390": float((cap > NEAR_CEILING).mean() * 100),
    }

    print("\n========== CEILING AUDIT ==========")
    for key, value in distribution.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")

    with open(
        OUTPUT_DIR / "ceiling_audit.json", "w", encoding="utf-8"
    ) as f:
        json.dump(distribution, f, indent=2)

    # Exact frequency table for high CAP values.
    high_frequency = (
        merged.loc[merged[TARGET] >= 380, TARGET]
        .value_counts()
        .sort_index()
        .rename_axis("CAP")
        .reset_index(name="count")
    )
    high_frequency["percentage_of_cohort"] = (
        high_frequency["count"] / len(merged) * 100
    )
    high_frequency.to_csv(
        OUTPUT_DIR / "high_cap_frequency.csv", index=False
    )

    # --------------------------------------------------------
    # 4. Error metrics across actual CAP ranges
    # --------------------------------------------------------

    bins = [-np.inf, 200, 248, 268, 280, 320, 360, 390, np.inf]
    labels = [
        "<200",
        "200–247",
        "248–267",
        "268–279",
        "280–319",
        "320–359",
        "360–390",
        ">390",
    ]

    merged["cap_range"] = pd.cut(
        merged[TARGET],
        bins=bins,
        labels=labels,
        right=False,
    )

    rows = []
    for label, group in merged.groupby(
        "cap_range", observed=False
    ):
        if len(group) == 0:
            continue

        rows.append({
            "cap_range": str(label),
            "n": int(len(group)),
            "mean_actual": float(group[TARGET].mean()),
            "mean_predicted": float(group["baseline_xgboost"].mean()),
            "mean_bias": float(group["residual"].mean()),
            "mae": float(group["absolute_error"].mean()),
            "rmse": float(np.sqrt(group["squared_error"].mean())),
        })

    range_metrics = pd.DataFrame(rows)
    range_metrics.to_csv(
        OUTPUT_DIR / "metrics_by_cap_range.csv", index=False
    )

    print("\n========== ERRORS BY ACTUAL CAP RANGE ==========")
    print(range_metrics.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}",
    ))

    # --------------------------------------------------------
    # 5. Prediction shrinkage at the extremes
    # --------------------------------------------------------

    slope, intercept = np.polyfit(actual, predicted, 1)

    shrinkage = {
        "prediction_vs_actual_slope": float(slope),
        "prediction_vs_actual_intercept": float(intercept),
        "interpretation": (
            "A slope below 1 is consistent with predictions varying "
            "less than actual values, but does not by itself establish "
            "the cause."
        ),
    }

    print("\n========== PREDICTION SPREAD ==========")
    print(f"Prediction-vs-actual slope: {slope:.4f}")
    print(f"Prediction-vs-actual intercept: {intercept:.4f}")

    with open(
        OUTPUT_DIR / "prediction_spread.json", "w", encoding="utf-8"
    ) as f:
        json.dump(shrinkage, f, indent=2)

    # --------------------------------------------------------
    # 6. Save participant-level diagnostic file
    # --------------------------------------------------------

    merged[[
        "SEQN",
        TARGET,
        "baseline_xgboost",
        "residual",
        "absolute_error",
        "cap_range",
    ]].to_csv(
        OUTPUT_DIR / "residuals_by_participant.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 7. Plots
    # --------------------------------------------------------

    plt.figure(figsize=(8, 6))
    plt.scatter(actual, predicted, alpha=0.25, s=12)
    plt.plot(
        [actual.min(), actual.max()],
        [actual.min(), actual.max()],
        linestyle="--",
    )
    plt.xlabel("Actual CAP (dB/m)")
    plt.ylabel("Out-of-fold predicted CAP (dB/m)")
    plt.title("Actual vs OOF-predicted CAP")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(actual, merged["residual"], alpha=0.25, s=12)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Actual CAP (dB/m)")
    plt.ylabel("Prediction minus actual (dB/m)")
    plt.title("OOF residuals vs actual CAP")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "residuals_vs_actual.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.hist(cap, bins=np.arange(100, 405, 5), edgecolor="black")
    plt.axvline(CEILING, linestyle="--")
    plt.xlabel("CAP (dB/m)")
    plt.ylabel("Participant count")
    plt.title("CAP distribution near the upper measurement limit")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cap_distribution.png", dpi=160)
    plt.close()

    print(f"\nAll results saved to:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
