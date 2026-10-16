
"""Audit the apparent upper boundary of NHANES CAP values.

Run from the HepatoTwin project root:
    python audit_cap_ceiling.py
"""

import numpy as np
import pandas as pd

from src.train_final_model import ROOT, TARGET


SOURCE = ROOT / "data/processed/nhanes_merged.csv"


def main():
    df = pd.read_csv(SOURCE)

    required = ["SEQN", "LUAXSTAT", TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Same eligible cohort used by the project.
    df = df.loc[
        df["LUAXSTAT"].eq(1) & df[TARGET].notna()
    ].copy()

    if df.empty:
        raise ValueError("No eligible CAP observations found.")

    y = pd.to_numeric(df[TARGET], errors="raise")

    print("\n=== 1. CAP distribution ===")
    print(f"Eligible rows: {len(df)}")
    print(f"Minimum: {y.min():.1f}")
    print(f"Maximum: {y.max():.1f}")
    print(f"Mean: {y.mean():.2f}")
    print(f"Median: {y.median():.2f}")

    print("\n=== 2. Upper-tail counts ===")
    for threshold in [350, 360, 370, 380, 390, 395, 399, 400]:
        count = int((y >= threshold).sum())
        print(
            f"CAP >= {threshold}: {count:4d} "
            f"({count / len(df) * 100:.2f}%)"
        )

    print("\n=== 3. Most frequent CAP values ===")
    print(
        y.value_counts()
        .sort_values(ascending=False)
        .head(20)
        .rename_axis("CAP")
        .to_string()
    )

    print("\n=== 4. CAP IQR comparison ===")
    if "LUXCPIQR" in df.columns:
        df["LUXCPIQR"] = pd.to_numeric(
            df["LUXCPIQR"], errors="coerce"
        )

        for label, mask in [
            ("CAP exactly 400", y.eq(400)),
            ("CAP below 400", y.lt(400)),
            ("CAP >= 390", y.ge(390)),
        ]:
            values = df.loc[mask, "LUXCPIQR"].dropna()
            print(f"\n{label}:")
            print(f"Rows with nonmissing IQR: {len(values)}")

            if len(values):
                print(values.describe().round(2).to_string())
            else:
                print("No nonmissing IQR values available.")
    else:
        print("LUXCPIQR is absent from the merged CSV.")

    print("\n=== 5. Probe comparison ===")
    # LUAPNME is the NHANES probe-type variable in relevant cycles.
    probe_col = "LUAPNME"

    if probe_col in df.columns:
        table = (
            df.assign(
                at_400=y.eq(400),
                at_least_390=y.ge(390),
            )
            .groupby(probe_col, dropna=False)
            .agg(
                rows=(TARGET, "size"),
                cap_400=("at_400", "sum"),
                cap_ge_390=("at_least_390", "sum"),
                mean_cap=(TARGET, "mean"),
                max_cap=(TARGET, "max"),
            )
        )

        table["pct_at_400"] = (
            100 * table["cap_400"] / table["rows"]
        )

        print(table.round(2).to_string())
    else:
        print(
            f"{probe_col} is absent from the merged CSV; "
            "probe-specific analysis is unavailable."
        )

    print("\n=== 6. Values at the upper boundary ===")
    print(
        df.loc[y.ge(390)]
        .sort_values(TARGET, ascending=False)
        .head(25)[
            ["SEQN", TARGET]
            + [
                c for c in ["LUXCPIQR", "LUAPNME"]
                if c in df.columns
            ]
        ]
        .to_string(index=False)
    )

    print("\nAudit complete. No files or models were modified.")


if __name__ == "__main__":
    main()
