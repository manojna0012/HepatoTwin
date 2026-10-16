from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.base import clone

from src.train_final_model import (
    ROOT, NUMERIC, CATEGORICAL, TARGET, load_cohort, make_pipeline
)

OUT = ROOT / "data/results/experiments/hba1c_ablation"
OUT.mkdir(parents=True, exist_ok=True)

data = load_cohort(ROOT / "data/processed/nhanes_merged.csv")
y = data[TARGET]

# Compare on participants with observed HbA1c, so both models use
# precisely the same people and folds.
eligible = data["LBXGH"].notna()
data = data.loc[eligible].reset_index(drop=True)
y = data[TARGET]

features_with = NUMERIC + CATEGORICAL
features_without = [c for c in features_with if c != "LBXGH"]

cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Use the selected model configuration, not the generic default.
# Read the saved OOF file to confirm the selected model's parameters
# before running this experiment.
params_path = ROOT / "data/results/engineered/metrics.json"
if not params_path.exists():
    raise FileNotFoundError(
        f"Cannot find {params_path}. Stop and inspect the saved results "
        "to confirm the selected model parameters."
    )

with open(params_path, "r", encoding="utf-8") as f:
    saved = json.load(f)

print("Saved metrics file loaded:", params_path)
print("Keys:", list(saved.keys()))
print("Eligible participants with HbA1c:", len(data))
print("This script intentionally stops before training so we can verify "
      "the exact selected-model parameters first.")
