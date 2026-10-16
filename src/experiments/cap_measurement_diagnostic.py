
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "processed" / "nhanes_merged.csv"

OUTPUT_DIR = (
    ROOT
    / "data"
    / "results"
    / "experiments"
    / "cap_measurement_diagnostic"
)

TARGET = "LUXCAPM"

CANDIDATE_COLUMNS = [
    "SEQN",
    "LUXCAPM",
    "LUAPNME",
    "LUXCPIQR",
    "LUANMVGP",
    "LUANMTGP",
    "LUAXSTAT",
]

NUMERIC_COLUMNS = [
    "LUXCAPM",
    "LUXCPIQR",
    "LUANMVGP",
    "LUANMTGP",
    "LUAXSTAT",
]


def normalize_probe(value):
    """Normalize probe labels, including byte-string values."""
    if pd.isna(value):
        return "Missing"

    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")

    value = str(value).strip()

    if value.startswith("b'") and value.endswith("'"):
        value = value[2:-1]

    return value


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    # Read only relevant columns that exist in the dataset.
    header = pd.read_csv(DATA_PATH, nrows=0).columns.tolist()

    available = [
        column for column in CANDIDATE_COLUMNS
        if column in header
    ]

    required = {"SEQN", TARGET}
    missing = required - set(available)

    if missing:
        raise ValueError(
            f"Required columns are missing: {sorted(missing)}\n"
            f"Available related columns: "
            f"{[c for c in header if 'LUX' in c or 'LUA' in c]}"
        )

    df = pd.read_csv(DATA_PATH, usecols=available)

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    df = df.dropna(subset=[TARGET]).copy()

    if "LUAPNME" in df.columns:
        df["probe"] = df["LUAPNME"].apply(normalize_probe)
    else:
        df["probe"] = "Unavailable"

    df["cap_group"] = np.where(
        df[TARGET] == 400,
        "exactly_400",
        "not_400",
    )

    return df


def summarize_numeric(df, group_columns):
    """Summarize available measurement-quality variables."""
    measurement_columns = [
        column for column in [
            "LUXCPIQR",
            "LUANMVGP",
            "LUANMTGP",
        ]
        if column in df.columns
    ]

    rows = []

    grouped = df.groupby(
        group_columns,
        dropna=False,
        observed=True,
    )

    for group_key, group in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        group_info = dict(zip(group_columns, group_key))

        for column in measurement_columns:
            values = group[column].dropna()

            rows.append({
                **group_info,
                "variable": column,
                "n_nonmissing": len(values),
                "mean": values.mean() if len(values) else np.nan,
                "median": values.median() if len(values) else np.nan,
                "q1": values.quantile(0.25) if len(values) else np.nan,
                "q3": values.quantile(0.75) if len(values) else np.nan,
                "minimum": values.min() if len(values) else np.nan,
                "maximum": values.max() if len(values) else np.nan,
            })

    return pd.DataFrame(rows)


def summarize_counts(df, group_columns):
    """Count participants and estimate the proportion at CAP=400."""
    result = (
        df.groupby(
            group_columns,
            dropna=False,
            observed=True,
        )
        .agg(
            participants=(TARGET, "size"),
            mean_CAP=(TARGET, "mean"),
            median_CAP=(TARGET, "median"),
            exactly_400=(TARGET, lambda x: int((x == 400).sum())),
        )
        .reset_index()
    )

    result["pct_exactly_400"] = (
        100 * result["exactly_400"] / result["participants"]
    )

    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()

    print("\n=== Dataset overview ===")
    print(f"Rows with observed CAP: {len(df)}")
    print(f"CAP exactly 400: {(df[TARGET] == 400).sum()}")
    print(f"CAP below 400: {(df[TARGET] < 400).sum()}")
    print(f"CAP above 400: {(df[TARGET] > 400).sum()}")

    # 1. Compare all participants at 400 versus not at 400.
    overall_counts = summarize_counts(df, ["cap_group"])
    overall_counts.to_csv(
        OUTPUT_DIR / "cap_group_counts.csv",
        index=False,
    )

    # 2. Compare measurement variables at 400 versus not at 400.
    overall_quality = summarize_numeric(df, ["cap_group"])
    overall_quality.to_csv(
        OUTPUT_DIR / "measurement_quality_by_cap_group.csv",
        index=False,
    )

    # 3. Compare CAP=400 frequency by probe type.
    probe_counts = summarize_counts(df, ["probe"])
    probe_counts.to_csv(
        OUTPUT_DIR / "cap_boundary_by_probe.csv",
        index=False,
    )

    # 4. Compare measurement variables by both probe and CAP group.
    probe_quality = summarize_numeric(
        df,
        ["probe", "cap_group"],
    )
    probe_quality.to_csv(
        OUTPUT_DIR / "measurement_quality_by_probe_and_cap.csv",
        index=False,
    )

    # 5. Cross-tabulate exam status, if available.
    if "LUAXSTAT" in df.columns:
        status_table = pd.crosstab(
            df["LUAXSTAT"],
            df["cap_group"],
            margins=True,
            dropna=False,
        )
        status_table.to_csv(
            OUTPUT_DIR / "exam_status_by_cap_group.csv"
        )

    # 6. Export participant-level boundary cases for inspection.
    boundary = df[df[TARGET] == 400].copy()

    participant_columns = [
        column for column in [
            "SEQN",
            "LUXCAPM",
            "LUAPNME",
            "probe",
            "LUXCPIQR",
            "LUANMVGP",
            "LUANMTGP",
            "LUAXSTAT",
        ]
        if column in boundary.columns
    ]

    boundary[participant_columns].to_csv(
        OUTPUT_DIR / "participants_at_cap_400.csv",
        index=False,
    )

    # 7. Print the key comparisons.
    print("\n=== CAP boundary frequency by probe ===")
    print(probe_counts.round(3).to_string(index=False))

    print("\n=== Measurement quality by CAP group ===")
    print(overall_quality.round(3).to_string(index=False))

    print("\n=== Measurement quality by probe and CAP group ===")
    print(probe_quality.round(3).to_string(index=False))

    if "LUAXSTAT" in df.columns:
        print("\n=== Exam status by CAP group ===")
        print(status_table.to_string())

    print(f"\nResults saved to:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
