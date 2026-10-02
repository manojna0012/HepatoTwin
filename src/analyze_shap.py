"""
SHAP explainability analysis for the finalized patient-state CAP model.

Purpose:
- Load the already-trained patient-state model.
- Use the exact fixed test split used for model evaluation.
- Reproduce the model's preprocessing pipeline.
- Compute SHAP values for the XGBoost model.
- Verify SHAP additivity against the pipeline predictions.
- Save SHAP Analysis outputs.

Important:
- The saved model is NEVER modified.
- SHAP uses an in-memory copy of the XGBoost model.
- This script processes all test participants.
"""

from pathlib import Path
import copy
import json

import joblib
import numpy as np
import pandas as pd
import shap


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "data" / "results" / "engineered" / "patient_state_model.joblib"
DATA_PATH = ROOT / "data" / "processed" / "nhanes_merged.csv"
SPLIT_PATH = ROOT / "data" / "results" / "engineered" / "split_manifest.csv"

OUTPUT_DIR = ROOT / "data" / "results"

SHAP_VALUES_PATH = OUTPUT_DIR / "shap_values.csv"
SHAP_IMPORTANCE_PATH = OUTPUT_DIR / "shap_feature_importance.csv"
ADDITIVITY_PATH = OUTPUT_DIR / "shap_additivity.csv"


# ============================================================
# SHAP ANALYSIS
# ============================================================


# the complete 1,805-person test set.
MAX_TEST_PATIENTS = None


# ============================================================
# Helper functions
# ============================================================

def normalize_xgboost_base_score(model):
    """
    Create an in-memory copy of an XGBoost model whose base_score
    representation is compatible with SHAP 0.49.1.

    XGBoost 3.x can store base_score as something like:

        "[2.561038E2]"

    while SHAP 0.49.1 expects:

        "2.561038E2"

    The original saved model is never modified.
    """

    shap_model = copy.deepcopy(model)

    booster = shap_model.get_booster()

    config = json.loads(booster.save_config())

    learner_params = config["learner"]["learner_model_param"]

    base_score = learner_params["base_score"]

    if isinstance(base_score, str) and base_score.startswith("["):
        base_score = base_score.strip("[]")
        learner_params["base_score"] = base_score

    booster.load_config(json.dumps(config))

    return shap_model


def get_feature_names(preprocessor):
    """
    Get the names of the 93 transformed features produced by
    the fitted ColumnTransformer.
    """

    return list(preprocessor.get_feature_names_out())


# ============================================================
# Main
# ============================================================

def main():

    print("Loading saved patient-state model...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    pipeline = joblib.load(MODEL_PATH)

    print(f"Model loaded from: {MODEL_PATH}")
    print()


    # --------------------------------------------------------
    # Load fixed test split
    # --------------------------------------------------------

    print("Loading fixed test split...")

    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"Split manifest not found:\n{SPLIT_PATH}"
        )

    split_manifest = pd.read_csv(SPLIT_PATH)

    test_ids = (
        split_manifest.loc[
            split_manifest["partition"].astype(str).str.lower() == "test",
            "SEQN"
        ]
        .astype(int)
        .tolist()
    )

    print(f"Total fixed test participants: {len(test_ids)}")

    if MAX_TEST_PATIENTS is not None:
        test_ids = test_ids[:MAX_TEST_PATIENTS]

    print(
        f"Using first {len(test_ids)} test participants "
        f"for SHAP Analysis."
    )
    print()


    # --------------------------------------------------------
    # Load NHANES data
    # --------------------------------------------------------

    print("Loading NHANES data...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"NHANES merged data not found:\n{DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    df["SEQN"] = df["SEQN"].astype(int)

    test_df = (
        df[df["SEQN"].isin(test_ids)]
        .copy()
        .set_index("SEQN")
        .loc[test_ids]
        .reset_index()
    )

    print(f"Loaded test rows: {len(test_df)}")
    print()


    if len(test_df) != len(test_ids):
        missing_ids = sorted(set(test_ids) - set(test_df["SEQN"]))

        raise ValueError(
            "Some test participants were not found in NHANES data.\n"
            f"Missing SEQN values: {missing_ids}"
        )


    # --------------------------------------------------------
    # Inspect pipeline
    # --------------------------------------------------------

    print("Pipeline components:")

    if "features" not in pipeline.named_steps:
        raise ValueError(
            "Expected outer pipeline step 'features' was not found."
        )

    if "regressor" not in pipeline.named_steps:
        raise ValueError(
            "Expected outer pipeline step 'regressor' was not found."
        )

    feature_transformer = pipeline.named_steps["features"]

    regressor_pipeline = pipeline.named_steps["regressor"]

    if "preprocess" not in regressor_pipeline.named_steps:
        raise ValueError(
            "Expected regressor step 'preprocess' was not found."
        )

    if "model" not in regressor_pipeline.named_steps:
        raise ValueError(
            "Expected regressor step 'model' was not found."
        )

    preprocessor = regressor_pipeline.named_steps["preprocess"]

    xgb_model = regressor_pipeline.named_steps["model"]

    print(
        f"  Feature transformer: "
        f"{feature_transformer.__class__.__name__}"
    )

    print(
        f"  Preprocessor: "
        f"{preprocessor.__class__.__name__}"
    )

    print(
        f"  Model: "
        f"{xgb_model.__class__.__name__}"
    )

    print()


    # --------------------------------------------------------
    # Identify raw model features
    # --------------------------------------------------------

    if not hasattr(pipeline, "feature_names_in_"):
        raise ValueError(
            "The saved pipeline does not contain feature_names_in_."
        )

    input_features = list(pipeline.feature_names_in_)

    missing_features = [
        feature
        for feature in input_features
        if feature not in test_df.columns
    ]

    if missing_features:
        raise ValueError(
            "The following model input features are missing "
            f"from the NHANES data:\n{missing_features}"
        )

    X_raw = test_df[input_features].copy()

    print(f"Raw input shape: {X_raw.shape}")
    print()


    # --------------------------------------------------------
    # Apply CAPFeatures
    # --------------------------------------------------------

    print("Applying CAPFeatures...")

    X_engineered = feature_transformer.transform(X_raw)

    print(f"Engineered shape: {X_engineered.shape}")
    print()


    # --------------------------------------------------------
    # Apply fitted preprocessing
    # --------------------------------------------------------

    print("Applying fitted preprocessing...")

    X_transformed = preprocessor.transform(X_engineered)

    print(f"Transformed shape: {X_transformed.shape}")

    feature_names = get_feature_names(preprocessor)
    transformed_df = pd.DataFrame(
    X_transformed,
    columns=feature_names)

    transformed_df.insert(
        0,
        "SEQN",
        test_df["SEQN"].values
    )

    transformed_df.to_csv(
        OUTPUT_DIR / "shap_transformed_features.csv",
        index=False
    )

    print(
        f"Number of transformed features: "
        f"{len(feature_names)}"
    )
    print()


    # --------------------------------------------------------
    # Check feature-name alignment
    # --------------------------------------------------------

    if X_transformed.shape[1] != len(feature_names):
        raise ValueError(
            "Number of transformed columns does not match "
            "number of transformed feature names."
        )


    # --------------------------------------------------------
    # Check predictions
    # --------------------------------------------------------

    print("Checking predictions...")

    pipeline_predictions = pipeline.predict(X_raw)

    print("First predictions:")

    for seqn, prediction in zip(
        test_df["SEQN"].head(5),
        pipeline_predictions[:5]
    ):
        print(
            f"  SEQN {int(seqn)}: "
            f"{prediction:.3f} dB/m"
        )

    print()


    # --------------------------------------------------------
    # Create SHAP-compatible in-memory model
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Create SHAP explainer
    # --------------------------------------------------------

    print("Creating SHAP explainer...")

    # Use the already-transformed feature matrix.
    # This avoids SHAP attempting to parse the XGBoost 3.x
    # internal model configuration.
    background = X_transformed

    def predict_transformed(X):
        return xgb_model.predict(X)


    explainer = shap.Explainer(
        predict_transformed,
        background,
        algorithm="permutation"
    )

    print("SHAP explainer created successfully.")
    print()

    


    # --------------------------------------------------------
    # Calculate SHAP values
    # --------------------------------------------------------

    print("Calculating SHAP values...")

    shap_explanation = explainer(
        X_transformed
    )

    shap_values = np.asarray(
        shap_explanation.values
    )

    print(f"SHAP shape: {shap_values.shape}")
    print()


    # --------------------------------------------------------
    # Validate SHAP dimensions
    # --------------------------------------------------------

    expected_shape = X_transformed.shape

    if shap_values.shape != expected_shape:
        raise ValueError(
            "SHAP value shape does not match transformed "
            "feature matrix.\n"
            f"Expected: {expected_shape}\n"
            f"Actual:   {shap_values.shape}"
        )


    # --------------------------------------------------------
    # SHAP additivity check
    # --------------------------------------------------------

    print("Checking SHAP additivity...")

    expected_value = float(
    np.asarray(shap_explanation.base_values).mean())

    shap_reconstructed_predictions = (
        expected_value + shap_values.sum(axis=1)
    )

    additivity_difference = (
        pipeline_predictions
        - shap_reconstructed_predictions
    )

    max_abs_difference = np.max(
        np.abs(additivity_difference)
    )

    mean_abs_difference = np.mean(
        np.abs(additivity_difference)
    )

    print(
        f"SHAP expected value: "
        f"{expected_value:.6f}"
    )

    print(
        f"Maximum absolute difference: "
        f"{max_abs_difference:.10f}"
    )

    print(
        f"Mean absolute difference: "
        f"{mean_abs_difference:.10f}"
    )

    print()


    # --------------------------------------------------------
    # Save additivity results
    # --------------------------------------------------------

    additivity_df = pd.DataFrame({
        "SEQN": test_df["SEQN"].values,
        "pipeline_prediction": pipeline_predictions,
        "shap_reconstructed_prediction": (
            shap_reconstructed_predictions
        ),
        "difference": additivity_difference,
        "absolute_difference": np.abs(additivity_difference),
    })

    additivity_df.to_csv(
        ADDITIVITY_PATH,
        index=False
    )


    # --------------------------------------------------------
    # Save individual SHAP values
    # --------------------------------------------------------

    shap_df = pd.DataFrame(
        shap_values,
        columns=feature_names
    )

    shap_df.insert(
        0,
        "SEQN",
        test_df["SEQN"].values
    )

    shap_df.to_csv(
        SHAP_VALUES_PATH,
        index=False
    )


    # --------------------------------------------------------
    # Calculate global SHAP importance
    # --------------------------------------------------------

    mean_abs_shap = np.mean(
        np.abs(shap_values),
        axis=0
    )

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    })

    importance_df = (
        importance_df
        .sort_values(
            "mean_abs_shap",
            ascending=False
        )
        .reset_index(drop=True)
    )

    importance_df.insert(
        0,
        "rank",
        np.arange(1, len(importance_df) + 1)
    )

    importance_df.to_csv(
        SHAP_IMPORTANCE_PATH,
        index=False
    )


    # --------------------------------------------------------
    # Print top SHAP features
    # --------------------------------------------------------

    print("Top SHAP features:")

    print(
        importance_df
        .head(20)
        .to_string(index=False)
    )

    print()


    # --------------------------------------------------------
    # Output paths
    # --------------------------------------------------------

    print("SHAP Analysis outputs saved:")

    print(
        f"  SHAP values: "
        f"{SHAP_VALUES_PATH}"
    )

    print(
        f"  SHAP importance: "
        f"{SHAP_IMPORTANCE_PATH}"
    )

    print(
        f"  Additivity check: "
        f"{ADDITIVITY_PATH}"
    )

    print()

    print("SHAP analysis completed successfully.")


if __name__ == "__main__":
    main()