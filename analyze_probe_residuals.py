
"""Evaluate CAP model residuals separately for M and XL probes.

Run from the HepatoTwin project root:
    python analyze_probe_residuals.py

Read-only analysis; does not retrain or modify any model.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.train_final_model import ROOT, TARGET


SOURCE = ROOT / "data/processed/nhanes_merged.csv"
OOF_FILE = (
    ROOT / "data/results/experiments/quantile_regression/oof_predictions.csv"
)


def normalize_probe(value):
    """Normalize strings or byte-like values such as b'XL'."""
    if pd.isna(value):
        return "Missing"

    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")

    value = str(value).strip()

    if value.startswith("b'") and value.endswith("'"):
        value = value[2:-1]
    elif value.startswith('b"') and value.endswith('"'):
        value = value[2:-1]

    return value


def summarize(group, label):
    y = group["Actual_CAP"].to_numpy(dtype=float)
    pred = group["baseline_xgb"].to_numpy(dtype=float)
    residual = pred - y

    result = {
        "group": label,
        "n": len(group),
        "actual_mean": y.mean(),
        "predicted_mean": pred.mean(),
        "bias_pred_minus_actual": residual.mean(),
        "MAE": mean_absolute_error(y, pred),
        "RMSE": np.sqrt(mean_squared_error(y, pred)),
        "R2": r2_score(y, pred) if len(group) >= 2 else np.nan,
        "actual_400_count": int(np.sum(y == 400)),
        "actual_ge390_count": int(np.sum(y >= 390)),
    }
    return result


def main():
    source = pd.read_csv(SOURCE)
    oof = pd.read_csv(OOF_FILE)

    needed_source = ["SEQN", "LUAXSTAT", TARGET, "LUAPNME"]
    missing = [c for c in needed_source if c not in source.columns]
    if missing:
        raise ValueError(f"Source CSV missing columns: {missing}")

    needed_oof = ["SEQN", "Actual_CAP", "baseline_xgb"]
    missing = [c for c in needed_oof if c not in oof.columns]
    if missing:
        raise ValueError(f"OOF file missing columns: {missing}")

    source = source.loc[
        source["LUAXSTAT"].eq(1) & source[TARGET].notna(),
        needed_source,
    ].copy()

    if source["SEQN"].duplicated().any():
        raise ValueError("Duplicate SEQN values in source data.")
    if oof["SEQN"].duplicated().any():
        raise ValueError("Duplicate SEQN values in OOF predictions.")

    # Join only participants with out-of-fold predictions.
    df = oof.merge(
        source,
        on="SEQN",
        how="left",
        validate="one_to_one",
        suffixes=("_oof", "_source"),
        indicator=True,
    )

    unmatched = int((df["_merge"] != "both").sum())
    if unmatched:
        raise ValueError(
            f"{unmatched} OOF participants could not be matched to source."
        )

    # Check that target values still match after the join.
    np.testing.assert_allclose(
        df["Actual_CAP"].to_numpy(dtype=float),
        df[TARGET].to_numpy(dtype=float),
        atol=1e-8,
        rtol=0,
    )

    df["probe"] = df["LUAPNME"].map(normalize_probe)
    df["residual_pred_minus_actual"] = (
        df["baseline_xgb"] - df["Actual_CAP"]
    )

    print("\n=== 1. Overall OOF performance ===")
    print(pd.DataFrame([
        summarize(df, "All OOF participants")
    ]).round(3).to_string(index=False))

    print("\n=== 2. Performance by probe ===")
    rows = []
    for probe, group in df.groupby("probe", dropna=False):
        rows.append(summarize(group, str(probe)))

    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\n=== 3. Error by probe and actual CAP range ===")
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
    df["CAP_band"] = pd.cut(
        df["Actual_CAP"],
        bins=bins,
        labels=labels,
        right=True,
    )

    rows = []
    for (probe, band), group in df.groupby(
        ["probe", "CAP_band"], observed=True
    ):
        if len(group) == 0:
            continue

        rows.append({
            "probe": probe,
            "CAP_band": str(band),
            "n": len(group),
            "mean_actual": group["Actual_CAP"].mean(),
            "mean_predicted": group["baseline_xgb"].mean(),
            "bias_pred_minus_actual": (
                group["baseline_xgb"] - group["Actual_CAP"]
            ).mean(),
            "MAE": mean_absolute_error(
                group["Actual_CAP"], group["baseline_xgb"]
            ),
        })

    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\n=== 4. Prediction errors for actual CAP = 400 ===")
    at_400 = df.loc[df["Actual_CAP"].eq(400)]

    if len(at_400):
        print(
            at_400.groupby("probe").agg(
                n=("Actual_CAP", "size"),
                mean_prediction=("baseline_xgb", "mean"),
                median_prediction=("baseline_xgb", "median"),
                mean_error=("residual_pred_minus_actual", "mean"),
                min_prediction=("baseline_xgb", "min"),
                max_prediction=("baseline_xgb", "max"),
            ).round(3).to_string()
        )
    else:
        print("No CAP=400 records occur in this OOF subset.")

    output = ROOT / "data/results/experiments/quantile_regression"
    output.mkdir(parents=True, exist_ok=True)
    df.to_csv(output / "probe_residuals.csv", index=False)

    print(f"\nDetailed joined records saved to: {output / 'probe_residuals.csv'}")


if __name__ == "__main__":
    main()
