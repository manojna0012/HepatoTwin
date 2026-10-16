
from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/processed/nhanes_merged.csv")
TARGET = "LUXCAPM"

df = pd.read_csv(DATA_PATH)

if TARGET not in df.columns:
    raise ValueError(
        f"{TARGET} not found. Available columns: {df.columns.tolist()}"
    )

cap = pd.to_numeric(df[TARGET], errors="coerce").dropna()

print("\n--- CAP target audit ---")
print(f"Rows in dataset: {len(df)}")
print(f"Valid CAP values: {len(cap)}")
print(f"Missing CAP values: {df[TARGET].isna().sum()}")

print("\n--- Distribution ---")
print(cap.describe(percentiles=[0.10, 0.25, 0.50, 0.75, 0.90, 0.95]))

print("\n--- Extreme values ---")
print(f"Values equal to 400: {(cap == 400).sum()}")
print(f"Values greater than 390: {(cap > 390).sum()}")
print(f"Values below 200: {(cap < 200).sum()}")

print("\n--- Provisional CAP-stage balance ---")
print("Exploratory cutoffs only; verify against the chosen clinical reference.")

bins = [-float("inf"), 248, 268, 280, float("inf")]
labels = ["S0", "S1", "S2", "S3"]

stages = pd.cut(
    cap,
    bins=bins,
    labels=labels,
    right=False,
)

print(stages.value_counts().reindex(labels))
print("\nClass proportions:")
print((stages.value_counts(normalize=True).reindex(labels) * 100).round(2))
