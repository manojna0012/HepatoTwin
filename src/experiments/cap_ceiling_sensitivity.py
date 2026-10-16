
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "processed" / "nhanes_merged.csv"

OOF_PATH = (
    ROOT
    / "data"
    / "results"
    / "experiments"
    / "probe_aware_model"
    / "oof_predictions.csv"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "results"
    / "experiments"
    / "cap_ceiling_sensitivity"
)

ID_COL = "SEQN"
TARGET = "LUXCAPM"
OOF_TARGET = "actual_CAP"
PROBE_COL = "LUAPNME"
IQR_COL = "LUXCPIQR"

PREDICTION_COLUMNS = {
    "Baseline": "baseline_prediction",
    "Probe-aware": "probe_prediction",
}


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(df, prediction_col):
    """Calculate regression metrics for a group."""

    group = df[[TARGET, prediction_col]].dropna()

    if group.empty:
        return {
            "n": 0,
            "R2": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan,
            "Bias": np.nan,
        }

    actual = group[TARGET].to_numpy(dtype=float)
    predicted = group[prediction_col].to_numpy(dtype=float)

    errors = predicted - actual

    return {
        "n": len(group),
        "R2": (
            r2_score(actual, predicted)
            if len(group) >= 2 and np.unique(actual).size >= 2
            else np.nan
        ),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "MAE": float(mean_absolute_error(actual, predicted)),
        "Bias": float(np.mean(errors)),
    }


def get_cap_band(cap):
    """Assign an observation to a CAP range."""

    if pd.isna(cap):
        return "missing"
    elif cap < 390:
        return "below_390"
    elif cap < 400:
        return "390_to_399"
    elif cap == 400:
        return "exactly_400"
    else:
        return "above_400"


def summarize_groups(df, group_columns):
    """Calculate metrics for each group and each model."""

    rows = []

    if group_columns:
        grouped = df.groupby(
            group_columns,
            dropna=False,
            observed=True,
        )
    else:
        grouped = [((), df)]

    for group_key, group in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        group_info = dict(zip(group_columns, group_key))

        for model_name, prediction_col in PREDICTION_COLUMNS.items():
            metrics = calculate_metrics(group, prediction_col)

            rows.append({
                **group_info,
                "model": model_name,
                **metrics,
            })

    return pd.DataFrame(rows)


# ============================================================
# LOAD AND VALIDATE DATA
# ============================================================

def load_data():
    """Load OOF predictions and join probe metadata by participant ID."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    if not OOF_PATH.exists():
        raise FileNotFoundError(
            f"OOF predictions not found:\n{OOF_PATH}\n"
            "Ensure the probe-aware experiment has been run."
        )

    # Read the source dataset.
    source = pd.read_csv(DATA_PATH)

    # Read existing out-of-fold predictions.
    oof = pd.read_csv(OOF_PATH)

    # Validate required columns.
    required_source = {ID_COL, TARGET, PROBE_COL}
    missing_source = required_source - set(source.columns)

    if missing_source:
        similar_columns = [
            col for col in source.columns
            if "LUA" in col.upper()
            or "PROBE" in col.upper()
            or "LUX" in col.upper()
        ]

        raise ValueError(
            f"Missing columns in source dataset: "
            f"{sorted(missing_source)}\n"
            f"Available related columns: {similar_columns}"
        )

    required_oof = {
        ID_COL,
        OOF_TARGET,
        "baseline_prediction",
        "probe_prediction",
    }
    missing_oof = required_oof - set(oof.columns)

    if missing_oof:
        raise ValueError(
            f"Missing columns in OOF predictions: "
            f"{sorted(missing_oof)}"
        )

    # Keep only the metadata needed for the analysis.
    metadata_columns = [ID_COL, PROBE_COL]

    if IQR_COL in source.columns:
        metadata_columns.append(IQR_COL)

    metadata = source[metadata_columns].copy()

    # Remove duplicate participant IDs before merging.
    metadata = metadata.drop_duplicates(subset=ID_COL)

    # Remove metadata columns from OOF data if already present.
    # This prevents pandas from creating LUAPNME_x / LUAPNME_y.
    columns_to_remove = [
        col for col in [PROBE_COL, IQR_COL]
        if col in oof.columns
    ]

    oof = oof.drop(columns=columns_to_remove, errors="ignore")

    # Merge by SEQN so each prediction retains the correct participant.
    df = oof.merge(
        metadata,
        on=ID_COL,
        how="left",
        validate="one_to_one",
    )

    # Standardize the target column name.
    df = df.rename(columns={OOF_TARGET: TARGET})

    if PROBE_COL not in df.columns:
        raise ValueError(
            f"{PROBE_COL} is missing after merging.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    # Ensure metric columns are numeric.
    numeric_columns = [
        TARGET,
        "baseline_prediction",
        "probe_prediction",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep only rows with actual values and both predictions.
    df = df.dropna(subset=numeric_columns).copy()

    # Add diagnostic categories.
    df["cap_band"] = df[TARGET].apply(get_cap_band)

    df["ceiling_group"] = np.where(
        df[TARGET] == 400,
        "exactly_400",
        "not_400",
    )

    # Attach CAP IQR if available.
    if IQR_COL in df.columns:
        df["cap_iqr"] = pd.to_numeric(
            df[IQR_COL],
            errors="coerce",
        )

    print("\nData loaded successfully.")
    print(f"OOF participants: {len(df)}")
    print(f"Missing probe types: {df[PROBE_COL].isna().sum()}")
    print(f"CAP exactly 400: {(df[TARGET] == 400).sum()}")
    print(f"CAP below 400: {(df[TARGET] < 400).sum()}")
    print(f"CAP above 400: {(df[TARGET] > 400).sum()}")

    return df


# ============================================================
# 1. FULL COHORT VS EXCLUDING CAP = 400
# ============================================================

def analyze_overall_sensitivity(df):
    """Compare full-cohort performance with performance excluding CAP=400."""

    full_cohort = df.copy()
    no_400 = df[df[TARGET] != 400].copy()

    full_summary = summarize_groups(full_cohort, [])
    full_summary["cohort"] = "full_oof_cohort"

    no_400_summary = summarize_groups(no_400, [])
    no_400_summary["cohort"] = "excluding_CAP_400"

    result = pd.concat(
        [full_summary, no_400_summary],
        ignore_index=True,
    )

    # Put the cohort column first.
    columns = ["cohort", "model", "n", "R2", "RMSE", "MAE", "Bias"]
    result = result[columns]

    result.to_csv(
        OUTPUT_DIR / "overall_sensitivity.csv",
        index=False,
    )

    return result


# ============================================================
# 2. PERFORMANCE BY CAP BAND
# ============================================================

def analyze_cap_bands(df):
    result = summarize_groups(df, ["cap_band"])

    result.to_csv(
        OUTPUT_DIR / "metrics_by_cap_band.csv",
        index=False,
    )

    return result


# ============================================================
# 3. PERFORMANCE BY PROBE TYPE
# ============================================================

def analyze_probe_types(df):
    result = summarize_groups(df, [PROBE_COL])

    result.to_csv(
        OUTPUT_DIR / "metrics_by_probe.csv",
        index=False,
    )

    return result


# ============================================================
# 4. PERFORMANCE BY PROBE TYPE AND CAP BAND
# ============================================================

def analyze_probe_and_cap_band(df):
    result = summarize_groups(
        df,
        [PROBE_COL, "cap_band"],
    )

    result.to_csv(
        OUTPUT_DIR / "metrics_by_probe_and_cap_band.csv",
        index=False,
    )

    return result


# ============================================================
# 5. CAP IQR BY CAP BAND
# ============================================================

def analyze_cap_iqr(df):
    if "cap_iqr" not in df.columns:
        print(
            f"\nSkipping IQR analysis: {IQR_COL} "
            "was not available in the source dataset."
        )
        return None

    result = (
        df.groupby(
            "cap_band",
            dropna=False,
            observed=True,
        )["cap_iqr"]
        .agg(
            n="count",
            mean="mean",
            median="median",
            std="std",
            minimum="min",
            maximum="max",
        )
        .reset_index()
    )

    result.to_csv(
        OUTPUT_DIR / "cap_iqr_by_band.csv",
        index=False,
    )

    return result


# ============================================================
# 6. PARTICIPANT-LEVEL ERROR ANALYSIS
# ============================================================

def save_participant_errors(df):
    result = df.copy()

    result["baseline_error"] = (
        result["baseline_prediction"] - result[TARGET]
    )

    result["probe_error"] = (
        result["probe_prediction"] - result[TARGET]
    )

    result["baseline_absolute_error"] = (
        result["baseline_error"].abs()
    )

    result["probe_absolute_error"] = (
        result["probe_error"].abs()
    )

    columns = [
        ID_COL,
        TARGET,
        PROBE_COL,
        "cap_band",
        "baseline_prediction",
        "probe_prediction",
        "baseline_error",
        "probe_error",
        "baseline_absolute_error",
        "probe_absolute_error",
    ]

    if "cap_iqr" in result.columns:
        columns.append("cap_iqr")

    result[columns].to_csv(
        OUTPUT_DIR / "participant_errors.csv",
        index=False,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_data()

    overall = analyze_overall_sensitivity(df)
    cap_bands = analyze_cap_bands(df)
    probes = analyze_probe_types(df)
    probe_bands = analyze_probe_and_cap_band(df)
    iqr_summary = analyze_cap_iqr(df)

    save_participant_errors(df)

    print("\n" + "=" * 75)
    print("1. OVERALL SENSITIVITY")
    print("=" * 75)
    print(overall.round(4).to_string(index=False))

    print("\n" + "=" * 75)
    print("2. METRICS BY CAP BAND")
    print("=" * 75)
    print(cap_bands.round(4).to_string(index=False))

    print("\n" + "=" * 75)
    print("3. METRICS BY PROBE TYPE")
    print("=" * 75)
    print(probes.round(4).to_string(index=False))

    print("\n" + "=" * 75)
    print("4. METRICS BY PROBE TYPE AND CAP BAND")
    print("=" * 75)
    print(probe_bands.round(4).to_string(index=False))

    if iqr_summary is not None:
        print("\n" + "=" * 75)
        print("5. CAP IQR BY CAP BAND")
        print("=" * 75)
        print(iqr_summary.round(4).to_string(index=False))

    print("\n" + "=" * 75)
    print("ANALYSIS COMPLETE")
    print("=" * 75)
    print(f"\nAll results saved to:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
