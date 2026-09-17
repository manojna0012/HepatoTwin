from pathlib import Path
import argparse
import pandas as pd

try:
    from .predict_cap import predict_patients, MODEL_FILE
except ImportError:
    from predict_cap import predict_patients, MODEL_FILE


# ============================================================
# CONFIG
# ============================================================

DATA_FILE = Path(__file__).resolve().parents[1] / "data/processed/nhanes_merged.csv"


# Raw patient features
RAW_FEATURES = [
    "RIDAGEYR",
    "RIAGENDR",
    "RIDRETH3",
    "BMXBMI",
    "BMXWAIST",
    "LBXSATSI",
    "LBXSASSI",
    "LBXGH",
    "LBXTC",
    "LBDHDD",
    "DR1TKCAL",
    "DR1TPROT",
    "DR1TCARB",
    "DR1TTFAT",
]

TARGET = "LUXCAPM"


# ============================================================
# EVIDENCE LAYER WILL IMPLEMENTED LATER (THIS IS ONLY A PLACEHOLDER, NEED TO IMPLEMENT THE ACTUAL LATER)
# ============================================================

EVIDENCE = {
    "higher_cap": {
        "finding": "Higher predicted CAP",
        "statement": (
            "The predicted CAP value indicates a higher "
            "level of hepatic fat accumulation."
        ),
        "source": "Hardcoded Week 4 evidence entry",
    },

    "lower_cap": {
        "finding": "Lower predicted CAP",
        "statement": (
            "The predicted CAP value is lower and does not "
            "indicate the same level of hepatic fat accumulation."
        ),
        "source": "Hardcoded Week 4 evidence entry",
    },
}


# ============================================================
# RECIPE LAYER WILL IMPLEMENTED LATER (THIS IS ONLY A PLACEHOLDER, NEED TO IMPLEMENT THE ACTUAL LATER)
# ============================================================

# Temporary placeholder for the future recipe layer.
# These values will be replaced with actual recipe data
# when the NutriTwin recipe evaluation layer is implemented.
RECIPE = {
    "name": "Real recipe from recipe dataset",
    "original_ingredient": "Original ingredient",
    "substitution": "Real substitution",
    "reason": "Reason for making the substitution",
}


# ============================================================
# LOAD PATIENT
# ============================================================

def load_patient(model_file=MODEL_FILE, seqn=None, random=False, data_file=DATA_FILE):

    df = pd.read_csv(data_file)
    manifest = pd.read_csv(Path(model_file).parent / 'split_manifest.csv')
    held_out = set(manifest.loc[manifest.partition.eq('test'), 'SEQN'])
    if seqn is not None and seqn not in held_out:
        raise ValueError(f'SEQN {seqn} is not in this model\'s held-out test set.')
    df = df[df.SEQN.isin(held_out)]
    if seqn is not None:
        df = df[df.SEQN.eq(seqn)]

    # Same LUX eligibility filter as previous preprocessing
    df = df[
        df["LUAXSTAT"] == 1.0
    ].copy()

    # CAP must exist
    df = df.dropna(
        subset=[TARGET]
    ).copy()

    if df.empty:
        raise ValueError(
            "No eligible held-out NHANES participants found."
        )

    patient = df.sample(1).iloc[0] if random else df.iloc[0]

    return patient


# ============================================================
# PREPARE MODEL INPUT
# ============================================================

def predict_cap(patient, model_file=MODEL_FILE):
    return float(predict_patients(pd.DataFrame([patient]), model_file).iloc[0])


# ============================================================
# EVIDENCE LOOKUP- THIS IS ALSO PLACEHOLDER CODE, NEED TO IMPLEMENT THE ACTUAL LATER
# ============================================================

def lookup_evidence(predicted_cap):

    if predicted_cap >= 280:

        return EVIDENCE[
            "higher_cap"
        ]

    return EVIDENCE[
        "lower_cap"
    ]


# ============================================================
# DISPLAY
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Demonstrate CAP prediction on a held-out participant.')
    parser.add_argument('--model', type=Path, default=MODEL_FILE)
    parser.add_argument('--seqn', type=int)
    parser.add_argument('--random', action='store_true')
    args = parser.parse_args()

    print("=" * 60)
    print("HEPATOTWIN END-TO-END DEMO")
    print("=" * 60)

    # --------------------------------------------------------
    # PATIENT
    # --------------------------------------------------------

    patient = load_patient(args.model, args.seqn, args.random)

    print("\nPATIENT (held-out test participant)")
    print(f'Model: {args.model}')
    print("-" * 60)

    print(
        f"SEQN: {int(patient['SEQN'])}"
    )

    for feature in RAW_FEATURES:

        print(
            f"{feature}: {patient[feature]}"
        )

    actual_cap = patient[
        TARGET
    ]

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    predicted_cap = predict_cap(
        patient, args.model
    )

    print("\nTWIN STATE")
    print("-" * 60)

    print(
        f"Actual CAP:     "
        f"{actual_cap:.2f} dB/m"
    )

    print(
        f"Predicted CAP:  "
        f"{predicted_cap:.2f} dB/m"
    )

    print(
        f"Absolute error: "
        f"{abs(actual_cap - predicted_cap):.2f} dB/m"
    )

    # --------------------------------------------------------
    # EVIDENCE- GETS OUTPUT FROM PLACEHOLDER CODE GETS OUTPUT FROM PLACEHOLDER CODE NEED TO IMPLEMENT THE ACTUAL LATER
    # --------------------------------------------------------

    evidence = lookup_evidence(
        predicted_cap
    )

    print("\nEVIDENCE [PLACEHOLDER - NOT IMPLEMENTED]")
    print("-" * 60)

    print(
        f"Finding: "
        f"{evidence['finding']}"
    )

    print(
        f"Statement: "
        f"{evidence['statement']}"
    )

    print(
        f"Source: "
        f"{evidence['source']}"
    )

    # --------------------------------------------------------
    # RECIPE-GETS OUTPUT FROM PLACEHOLDER CODE NEED TO IMPLEMENT THE ACTUAL LATER
    # --------------------------------------------------------

    print("\nNUTRITWIN [PLACEHOLDER - NOT IMPLEMENTED]")
    print("-" * 60)

    print(f"Recipe: " f"{RECIPE['name']}")

    print(f"Original ingredient: "f"{RECIPE['original_ingredient']}")

    print(
        f"Substitution: "
        f"{RECIPE['substitution']}"
    )

    print(
        f"Reason: "
        f"{RECIPE['reason']}"
    )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print(
        "END-TO-END DEMO COMPLETED"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
