from pathlib import Path
import joblib
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

MODEL_FILE = Path(
    "data/results/xgboost_cap_baseline.joblib"
)

DATA_FILE = Path(
    "data/processed/nhanes_merged.csv"
)


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

# Exact features used by XGBoost
MODEL_FEATURES = [
    "RIDAGEYR",
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
    "RIAGENDR_2.0",
    "RIDRETH3_2.0",
    "RIDRETH3_3.0",
    "RIDRETH3_4.0",
    "RIDRETH3_6.0",
    "RIDRETH3_7.0",
]

TARGET = "LUXCAPM"


# ============================================================
# EVIDENCE LAYER
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
# RECIPE LAYER
# ============================================================

# Temporary Week 4 real recipe/substitution entry.
# Replace these values with the actual recipe record
# and substitution from your recipe dataset.
RECIPE = {
    "name": "Real recipe from recipe dataset",
    "original_ingredient": "Original ingredient",
    "substitution": "Real substitution",
    "reason": "Reason for making the substitution",
}


# ============================================================
# LOAD PATIENT
# ============================================================

def load_patient():

    df = pd.read_csv(DATA_FILE)

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
            "No eligible NHANES participants found."
        )

    # Use one real NHANES participant
    patient = df.iloc[0]

    return patient


# ============================================================
# PREPARE MODEL INPUT
# ============================================================

def prepare_model_input(patient):

    # Start with numeric features
    model_input = {}

    numeric_features = [
        "RIDAGEYR",
        "BMXBMI",
        "BMXWAIST",
        "LBXSATSI",
        "LBXSASSI",
        "LBXGH",
        "LBXTC",
        "LBDHD",
        "DR1TKCAL",
        "DR1TPROT",
        "DR1TCARB",
        "DR1TTFAT",
    ]

    # NOTE:
    # Actual column is LBDHDD in the NHANES data.
    numeric_features = [
        "RIDAGEYR",
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

    for feature in numeric_features:

        value = patient[feature]

        # Median imputation will be handled below
        model_input[feature] = value

    # --------------------------------------------------------
    # SEX
    # --------------------------------------------------------

    model_input["RIAGENDR_2.0"] = (
        1.0
        if patient["RIAGENDR"] == 2.0
        else 0.0
    )

    # --------------------------------------------------------
    # RACE / ETHNICITY
    # --------------------------------------------------------

    for category in [
        2.0,
        3.0,
        4.0,
        6.0,
        7.0,
    ]:

        column = f"RIDRETH3_{category}"

        model_input[column] = (
            1.0
            if patient["RIDRETH3"] == category
            else 0.0
        )

    X = pd.DataFrame(
        [model_input],
        columns=MODEL_FEATURES
    )

    return X


# ============================================================
# IMPUTE MISSING VALUES
# ============================================================

def impute_missing_values(X):

    # Load the complete preprocessed training dataset.
    # Its medians reproduce the preprocessing used for
    # the current baseline model.
    preprocessed = pd.read_csv(
        "data/processed/nhanes_preprocessed.csv"
    )

    numeric_features = [
        "RIDAGEYR",
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

    for feature in numeric_features:

        if X[feature].isna().any():

            median_value = (
                preprocessed[feature]
                .median()
            )

            X[feature] = X[feature].fillna(
                median_value
            )

    return X


# ============================================================
# PREDICTION
# ============================================================

def predict_cap(patient):

    if not MODEL_FILE.exists():

        raise FileNotFoundError(
            f"Model not found: {MODEL_FILE}"
        )

    model = joblib.load(
        MODEL_FILE
    )

    X = prepare_model_input(
        patient
    )

    X = impute_missing_values(
        X
    )

    # Make absolutely sure feature order matches training
    X = X[
        MODEL_FEATURES
    ]

    prediction = model.predict(
        X
    )[0]

    return prediction


# ============================================================
# EVIDENCE LOOKUP
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

    print("=" * 60)
    print("HEPATOTWIN END-TO-END DEMO")
    print("=" * 60)

    # --------------------------------------------------------
    # PATIENT
    # --------------------------------------------------------

    patient = load_patient()

    print("\nPATIENT")
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
        patient
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
    # EVIDENCE
    # --------------------------------------------------------

    evidence = lookup_evidence(
        predicted_cap
    )

    print("\nEVIDENCE")
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
    # RECIPE
    # --------------------------------------------------------

    print("\nNUTRITWIN")
    print("-" * 60)

    print(
        f"Recipe: "
        f"{RECIPE['name']}"
    )

    print(
        f"Original ingredient: "
        f"{RECIPE['original_ingredient']}"
    )

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