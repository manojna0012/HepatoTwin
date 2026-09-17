"""
HepatoTwin end-to-end demo.

Fixes versus the previous demo:
  * loads the artifact train_final_model.py actually writes
  * uses a HELD-OUT participant (from the saved test SEQN list), so the
    reported error is a genuine generalisation error, not a training error
  * no manual one-hot construction and no re-reading the dataset to
    impute - the saved pipeline does both, exactly as at training time
  * reports the steatosis grade alongside the point estimate, with the
    model's typical error stated, because a single CAP number implies
    more precision than the model has

Usage:
    python demo.py                 # first held-out participant
    python demo.py --seqn 123456   # a specific held-out participant
    python demo.py --random        # a random held-out participant
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from train_final_model import (
    CATEGORICAL,
    engineer_features,
    FEATURES,
    TARGET,
)

MODEL_FILE = Path("data/results/final_patient_state_pipeline.joblib")
DATA_FILE = Path("data/processed/nhanes_merged.csv")

# Features worth showing a clinician, in reading order.
DISPLAY_FEATURES = [
    ("RIDAGEYR", "Age (years)"),
    ("RIAGENDR", "Sex (1=M, 2=F)"),
    ("RIDRETH3", "Race/ethnicity code"),
    ("BMXBMI", "BMI (kg/m2)"),
    ("BMXWAIST", "Waist circumference (cm)"),
    ("WHtR", "Waist-to-height ratio"),
    ("LBXSATSI", "ALT (U/L)"),
    ("LBXSASSI", "AST (U/L)"),
    ("LBXSGTSI", "GGT (U/L)"),
    ("LBXGH", "HbA1c (%)"),
    ("LBXSTR", "Triglycerides (mg/dL)"),
    ("LBXTC", "Total cholesterol (mg/dL)"),
    ("LBDHDD", "HDL cholesterol (mg/dL)"),
    ("FLI", "Fatty Liver Index"),
    ("DR1TKCAL", "Energy intake (kcal)"),
    ("pct_carb", "Carbohydrate (% energy)"),
    ("pct_fat", "Fat (% energy)"),
    ("pct_prot", "Protein (% energy)"),
]


# ============================================================
# PLACEHOLDER: EVIDENCE LAYER
# ============================================================
# Not implemented. These strings are stand-ins so the end-to-end path
# runs; the real layer must retrieve cited findings from the evidence
# base rather than hardcode statements.

EVIDENCE = {
    "S3": {
        "finding": "Severe steatosis range (CAP >= 280 dB/m)",
        "statement": "Predicted attenuation is in the range associated with "
                     "severe hepatic fat accumulation.",
        "source": "PLACEHOLDER - no citation, hardcoded",
    },
    "S2+": {
        "finding": "Moderate steatosis range (CAP 268-280 dB/m)",
        "statement": "Predicted attenuation is in the range associated with "
                     "moderate hepatic fat accumulation.",
        "source": "PLACEHOLDER - no citation, hardcoded",
    },
    "S1+": {
        "finding": "Mild steatosis range (CAP 248-268 dB/m)",
        "statement": "Predicted attenuation is in the range associated with "
                     "mild hepatic fat accumulation.",
        "source": "PLACEHOLDER - no citation, hardcoded",
    },
    "S0": {
        "finding": "No significant steatosis (CAP < 248 dB/m)",
        "statement": "Predicted attenuation is below the threshold commonly "
                     "used for significant hepatic steatosis.",
        "source": "PLACEHOLDER - no citation, hardcoded",
    },
}


# ============================================================
# PLACEHOLDER: NUTRITWIN RECIPE LAYER
# ============================================================

RECIPE = {
    "name": "PLACEHOLDER - recipe from recipe dataset",
    "original_ingredient": "PLACEHOLDER - original ingredient",
    "substitution": "PLACEHOLDER - substitution",
    "reason": "PLACEHOLDER - rationale for the substitution",
}


# ============================================================
# LOADING
# ============================================================

def load_bundle(model_file=MODEL_FILE):
    if not model_file.exists():
        raise FileNotFoundError(
            f"Model not found: {model_file}\nRun: python train_final_model.py"
        )
    return joblib.load(model_file)


def load_patient(bundle, seqn=None, random=False, data_file=DATA_FILE):
    """Pick a participant from the saved held-out test set."""
    df = pd.read_csv(data_file)
    df = df[df["LUAXSTAT"] == 1].dropna(subset=[TARGET])

    held_out = set(bundle["test_seqn"])
    df = df[df["SEQN"].astype(int).isin(held_out)].copy()

    if df.empty:
        raise ValueError("No held-out participants found in the dataset.")

    if seqn is not None:
        row = df[df["SEQN"].astype(int) == int(seqn)]
        if row.empty:
            raise ValueError(
                f"SEQN {seqn} is not in the held-out test set. "
                "Using a training participant would report a training error."
            )
        df = row
    elif random:
        df = df.sample(1, random_state=None)

    return engineer_features(df).iloc[0]


# ============================================================
# PREDICTION
# ============================================================

def predict_cap(bundle, patient):
    pipeline = bundle["pipeline"]
    cap_min, cap_max = bundle["cap_range"]

    # One row, columns in the exact training order. Missing values and
    # categorical encoding are handled inside the pipeline.
    X = patient[FEATURES].to_frame().T
    for col in FEATURES:
        if col not in CATEGORICAL:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    prediction = float(pipeline.predict(X)[0])
    return min(max(prediction, cap_min), cap_max)


def grade(cap, thresholds):
    if cap >= thresholds["S3"]:
        return "S3"
    if cap >= thresholds["S2+"]:
        return "S2+"
    if cap >= thresholds["S1+"]:
        return "S1+"
    return "S0"


# ============================================================
# DISPLAY
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqn", type=int, default=None)
    parser.add_argument("--random", action="store_true")
    args = parser.parse_args()

    print("=" * 64)
    print("HEPATOTWIN END-TO-END DEMO")
    print("=" * 64)

    bundle = load_bundle()
    patient = load_patient(bundle, seqn=args.seqn, random=args.random)
    thresholds = bundle["cap_thresholds"]
    typical_error = bundle["test_metrics"]["mae"]

    # --------------------------------------------------------
    # PATIENT
    # --------------------------------------------------------
    print("\nPATIENT  (held-out test participant - not seen during training)")
    print("-" * 64)
    print(f"{'SEQN':32s} {int(patient['SEQN'])}")

    for key, label in DISPLAY_FEATURES:
        value = patient.get(key)
        if pd.isna(value):
            shown = "missing (imputed by pipeline)"
        elif key in ("pct_prot", "pct_carb", "pct_fat"):
            shown = f"{100 * value:.1f}"
        elif isinstance(value, float):
            shown = f"{value:.1f}"
        else:
            shown = str(value)
        print(f"{label:32s} {shown}")

    # --------------------------------------------------------
    # TWIN STATE
    # --------------------------------------------------------
    actual_cap = float(patient[TARGET])
    predicted_cap = predict_cap(bundle, patient)

    print("\nTWIN STATE")
    print("-" * 64)
    print(f"{'Actual CAP':32s} {actual_cap:.1f} dB/m   (grade {grade(actual_cap, thresholds)})")
    print(f"{'Predicted CAP':32s} {predicted_cap:.1f} dB/m   (grade {grade(predicted_cap, thresholds)})")
    print(f"{'Absolute error':32s} {abs(actual_cap - predicted_cap):.1f} dB/m")
    print(f"\nModel's typical error on held-out data is {typical_error:.0f} dB/m, so the "
          f"grade\nmatters more than the point estimate. AUROC for CAP >= "
          f"{thresholds['S2+']} dB/m is "
          f"{bundle['test_metrics'].get('auroc_S2+', float('nan')):.3f}.")

    # --------------------------------------------------------
    # EVIDENCE  (placeholder layer)
    # --------------------------------------------------------
    evidence = EVIDENCE[grade(predicted_cap, thresholds)]

    print("\nEVIDENCE  [PLACEHOLDER LAYER - NOT IMPLEMENTED]")
    print("-" * 64)
    print(f"{'Finding':12s} {evidence['finding']}")
    print(f"{'Statement':12s} {evidence['statement']}")
    print(f"{'Source':12s} {evidence['source']}")

    # --------------------------------------------------------
    # NUTRITWIN  (placeholder layer)
    # --------------------------------------------------------
    print("\nNUTRITWIN  [PLACEHOLDER LAYER - NOT IMPLEMENTED]")
    print("-" * 64)
    print(f"{'Recipe':22s} {RECIPE['name']}")
    print(f"{'Original ingredient':22s} {RECIPE['original_ingredient']}")
    print(f"{'Substitution':22s} {RECIPE['substitution']}")
    print(f"{'Reason':22s} {RECIPE['reason']}")

    print("\n" + "=" * 64)
    print("END-TO-END DEMO COMPLETED")
    print("=" * 64)


if __name__ == "__main__":
    main()
