from pathlib import Path
import argparse
import pandas as pd

import json

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.predict_cap import predict_patients, predict_steatosis, steatosis_classifiers, MODEL_FILE
from src.recipe_scoring.multi_criteria import explain
from src.twin_pipeline import run_twin


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data/processed/nhanes_merged.csv"


def data_file_for(model_file):
    """Dataset recorded in the model's metadata; older models used nhanes_merged.csv."""
    metadata = Path(model_file).parent / "metadata.json"
    if metadata.exists():
        recorded = json.loads(metadata.read_text(encoding="utf-8")).get("data_file")
        if recorded:
            return ROOT / recorded
    return DATA_FILE


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
# LOAD PATIENT
# ============================================================

def load_patient(model_file=MODEL_FILE, seqn=None, random=False, data_file=None):

    df = pd.read_csv(data_file or data_file_for(model_file))
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
# DISPLAY
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Demonstrate CAP prediction on a held-out participant.')
    parser.add_argument('--model', type=Path, default=MODEL_FILE)
    parser.add_argument('--seqn', type=int)
    parser.add_argument('--random', action='store_true')
    parser.add_argument('--cuisine', help='Requested cuisine, e.g. Asian or European')
    parser.add_argument('--diet', choices=['vegan', 'vegetarian', 'pescetarian'])
    parser.add_argument('--top', type=int, default=3, help='Recipes to show')
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

    patient_frame = pd.DataFrame([patient])
    for classifier_file in steatosis_classifiers(args.model):
        probability = predict_steatosis(patient_frame, classifier_file)
        print(f"{probability.name}: {probability.iloc[0]:.1%} (calibrated, research estimate)")

    # --------------------------------------------------------
    # LAYER 3: EVIDENCE-LINKED PRIORITIES
    # --------------------------------------------------------

    preferences = {"requested_cuisine": args.cuisine, "diet": args.diet}
    twin = run_twin(patient.to_dict(), model_file=args.model, preferences=preferences)

    print("\nNUTRITIONAL PRIORITIES (Layer 3; candidates, not prescriptions)")
    print("-" * 60)
    if not twin["nutritional_priorities"]:
        print("No liver-related priority: steatosis is unlikely at the model's thresholds.")
    for item in twin["nutritional_priorities"]:
        strengths = sorted({e["evidence_strength"] for e in item.get("supporting_evidence", [])})
        print(f"[{item['priority'].upper()}] {item['title']}")
        print(f"    basis: {item['priority_basis']}; evidence: {', '.join(strengths)}")

    # --------------------------------------------------------
    # LAYER 4: NUTRITWIN RECIPE RANKING
    # --------------------------------------------------------

    print("\nNUTRITWIN RECOMMENDATIONS (Layer 4)")
    print("-" * 60)
    print(f"Status: {twin['scoring_status']}")
    ranked = twin["ranked_recipes"]
    if ranked:
        meeting = sum(item["meets_all_references"] for item in ranked)
        print(f"{len(ranked)} candidate meals ranked; {meeting} meet every reference value; "
              f"{len(twin['not_ranked'])} held for data review.")
        for item in ranked[:args.top]:
            print(explain(item))

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
