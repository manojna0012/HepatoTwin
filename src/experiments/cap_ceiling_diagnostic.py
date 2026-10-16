
"""Diagnose CAP values near the upper measurement boundary.

Uses existing out-of-fold predictions. Does not retrain or modify any model.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.train_final_model import ROOT, TARGET


DATA_PATH = ROOT / "data" / "processed" / "nhanes_merged.csv"
OOF_PATH = (
    ROOT / "data" / "results" / "experiments"
    / "probe_aware_model" / "oof_predictions.csv"
)
OUTPUT_DIR = (
    ROOT / "data" / "results" / "experiments" / "cap_ceiling_diagnostic"
)

CAP_BANDS = [
    ("below_390", -np.inf, 390),
    ("390_to_399", 390, 400),
    ("exactly_400", 400, 400),
]


def normalize_probe(value):
    """Normalize probe values stored as M, XL, or byte-like strings."""
    if pd.isna(value):
        return "Unknown"

    value = str(value).strip().upper()
    value = value.replace("B'", "").replace("'", "").replace('"', "").strip()

    if value == "M":
        return "M"
    if value == "XL":
        return "XL"
    return "Unknown"


def metrics(group, prediction_column):
    actual = group["actual_CAP"].to_numpy(dtype=float)
    predicted = group[prediction_column].to_numpy(dtype=float)
    error = predicted - actual

    return {
        "n": len(group),
        "mean_actual": float(np.mean(actual)),
        "mean_predicted": float(np.mean(predicted)),
        "mean_prediction_bias": float(np.mean(error)),
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "median_absolute_error": float(np.median(np.abs(error))),
        "actual_400_count": int(np.sum(actual == 400)),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not OOF_PATH.exists():
        raise FileNotFoundError(
            f"Could not find OOF predictions: {OOF_PATH}\n"
            "Run the probe-aware experiment first."
        )

    source = pd.read_csv(DATA_PATH)
    oof = pd.read_csv(OOF_PATH)

    required_source = ["SEQN", "LUAPNME", "LUXCPIQR", TARGET, "LUAXSTAT"]
    missing = sorted(set(required_source) - set(source.columns))
    if missing:
        raise ValueError(
            f"Source data is missing columns: {missing}. "
            "Check the spelling of the CAP IQR field."
        )

    required_oof = [
        "SEQN", "actual_CAP",
        "baseline_prediction", "probe_prediction",
    ]
    missing = sorted(set(required_oof) - set(oof.columns))
    if missing:
        raise ValueError(f"OOF file is missing columns: {missing}")

    # Join source measurement information to existing OOF predictions.
    source = source[
        ["SEQN", "LUAPNME", "LUXCPIQR", TARGET, "LUAXSTAT"]
    ].copy()

    if source["SEQN"].duplicated().any():
        raise ValueError("Duplicate SEQN values found in source data.")

    merged = oof.merge(
        source,
        on="SEQN",
        how="left",
        validate="one_to_one",
        suffixes=("", "_source"),
    )

    if merged["LUAPNME"].isna().all():
        raise ValueError("Could not join probe type to OOF participants.")

    if merged["LUAXSTAT"].isna().any():
        raise ValueError("Some OOF participants were not found in source data.")

    # Ensure the target values agree between the OOF file and source.
    np.testing.assert_allclose(
        merged["actual_CAP"].to_numpy(dtype=float),
        merged[TARGET].to_numpy(dtype=float),
        rtol=0,
        atol=1e-8,
    )

    merged["probe_type"] = merged["LUAPNME"].map(normalize_probe)
    merged["LUXCPIQR"] = pd.to_numeric(
        merged["LUXCPIQR"], errors="coerce"
    )

    # Assign CAP bands: [below 390], [390, 400), [exactly 400].
    merged["CAP_band"] = np.select(
        [
            merged["actual_CAP"] < 390,
            (merged["actual_CAP"] >= 390) & (merged["actual_CAP"] < 400),
            merged["actual_CAP"] == 400,
        ],
        ["below_390", "390_to_399", "exactly_400"],
        default="outside_expected_range",
    )

    # 1. Overall CAP boundary audit.
    audit_rows = []
    for band, lower, upper in CAP_BANDS:
        if band == "below_390":
            subset = merged[merged["actual_CAP"] < 390]
        elif band == "390_to_399":
            subset = merged[
                (merged["actual_CAP"] >= 390)
                & (merged["actual_CAP"] < 400)
            ]
        else:
            subset = merged[merged["actual_CAP"] == 400]

        audit_rows.append({
            "CAP_band": band,
            **metrics(subset, "probe_prediction"),
            "mean_CAP_IQR": float(subset["LUXCPIQR"].mean()),
            "median_CAP_IQR": float(subset["LUXCPIQR"].median()),
            "CAP_IQR_nonmissing_n": int(subset["LUXCPIQR"].notna().sum()),
        })

    pd.DataFrame(audit_rows).to_csv(
        OUTPUT_DIR / "cap_band_summary.csv", index=False
    )

    # 2. Probe-specific CAP distributions and errors.
    probe_rows = []
    for probe, probe_group in merged.groupby("probe_type", dropna=False):
        for band in ["below_390", "390_to_399", "exactly_400"]:
            group = probe_group[probe_group["CAP_band"] == band]
            if group.empty:
                continue

            row = {
                "probe_type": probe,
                "CAP_band": band,
                **metrics(group, "probe_prediction"),
                "mean_CAP_IQR": float(group["LUXCPIQR"].mean()),
                "median_CAP_IQR": float(group["LUXCPIQR"].median()),
                "CAP_IQR_nonmissing_n": int(group["LUXCPIQR"].notna().sum()),
            }
            probe_rows.append(row)

    pd.DataFrame(probe_rows).to_csv(
        OUTPUT_DIR / "probe_cap_band_summary.csv", index=False
    )

    # 3. CAP IQR comparison: exactly 400 vs below 400.
    iqr_rows = []
    for label, subset in [
        ("exactly_400", merged[merged["actual_CAP"] == 400]),
        ("below_400", merged[merged["actual_CAP"] < 400]),
        ("at_least_390", merged[merged["actual_CAP"] >= 390]),
    ]:
        values = subset["LUXCPIQR"].dropna()
        iqr_rows.append({
            "group": label,
            "participants": len(subset),
            "IQR_nonmissing_n": len(values),
            "IQR_mean": float(values.mean()) if len(values) else np.nan,
            "IQR_std": float(values.std()) if len(values) > 1 else np.nan,
            "IQR_median": float(values.median()) if len(values) else np.nan,
            "IQR_q1": float(values.quantile(0.25)) if len(values) else np.nan,
            "IQR_q3": float(values.quantile(0.75)) if len(values) else np.nan,
            "IQR_min": float(values.min()) if len(values) else np.nan,
            "IQR_max": float(values.max()) if len(values) else np.nan,
        })

    pd.DataFrame(iqr_rows).to_csv(
        OUTPUT_DIR / "cap_iqr_comparison.csv", index=False
    )

    # 4. Probe type counts and upper-boundary rates in the OOF cohort.
    count_rows = []
    for probe, group in merged.groupby("probe_type", dropna=False):
        count_rows.append({
            "probe_type": probe,
            "participants": len(group),
            "mean_CAP": float(group["actual_CAP"].mean()),
            "exactly_400_n": int((group["actual_CAP"] == 400).sum()),
            "exactly_400_pct": float((group["actual_CAP"] == 400).mean() * 100),
            "at_least_390_n": int((group["actual_CAP"] >= 390).sum()),
            "at_least_390_pct": float((group["actual_CAP"] >= 390).mean() * 100),
            "mean_prediction_bias": float(
                (group["probe_prediction"] - group["actual_CAP"]).mean()
            ),
        })

    pd.DataFrame(count_rows).to_csv(
        OUTPUT_DIR / "probe_boundary_rates.csv", index=False
    )

    # 5. Save a compact row-level diagnostic for further inspection.
    merged[
        [
            "SEQN", "probe_type", "actual_CAP", "CAP_band",
            "LUXCPIQR", "baseline_prediction", "probe_prediction",
        ]
    ].to_csv(OUTPUT_DIR / "participant_diagnostics.csv", index=False)

    print("\n=== CAP boundary summary ===")
    print(pd.DataFrame(audit_rows).to_string(index=False))

    print("\n=== Probe-specific boundary rates ===")
    print(pd.DataFrame(count_rows).to_string(index=False))

    print("\n=== CAP IQR comparison ===")
    print(pd.DataFrame(iqr_rows).to_string(index=False))

    print(f"\nSaved diagnostic files to:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
