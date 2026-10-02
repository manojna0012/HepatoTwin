"""
Calibration analysis for the finalized patient-state CAP model.

The model predicts continuous CAP (dB/m), so this script evaluates
regression calibration rather than probability calibration.

Calibration is fitted using training OOF predictions from the exact
selected model and evaluated on the fixed 1,805-person test set.

The saved patient-state model is never modified.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT
    / "data"
    / "results"
    / "engineered"
    / "patient_state_model.joblib"
)

DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "nhanes_merged.csv"
)

OOF_PATH = (
    ROOT
    / "data"
    / "results"
    / "engineered"
    / "oof_predictions.csv"
)

SPLIT_PATH = (
    ROOT
    / "data"
    / "results"
    / "engineered"
    / "split_manifest.csv"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "results"
    / "calibration"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Model configuration
# ============================================================

SELECTED_OOF_COLUMN = "ratios_600_trees_missing_flags"


# ============================================================
# Helper
# ============================================================

def regression_metrics(actual, predicted):
    """Return basic regression metrics."""

    return {
        "MAE": mean_absolute_error(
            actual,
            predicted
        ),
        "RMSE": np.sqrt(
            mean_squared_error(
                actual,
                predicted
            )
        ),
        "R2": r2_score(
            actual,
            predicted
        ),
        "Mean_error_pred_minus_actual": np.mean(
            predicted - actual
        ),
    }


# ============================================================
# Main
# ============================================================

def main():

    print("Starting CAP regression calibration analysis...")
    print()


    # --------------------------------------------------------
    # 1. Load OOF predictions
    # --------------------------------------------------------

    print("Loading training OOF predictions...")

    oof = pd.read_csv(OOF_PATH)

    if SELECTED_OOF_COLUMN not in oof.columns:
        raise ValueError(
            f"Selected model column not found: "
            f"{SELECTED_OOF_COLUMN}"
        )

    oof = oof[
        ["SEQN", "Actual_CAP", SELECTED_OOF_COLUMN]
    ].dropna()

    print(
        f"OOF rows: {len(oof)}"
    )

    print(
        f"Using model: {SELECTED_OOF_COLUMN}"
    )

    print()


    # --------------------------------------------------------
    # 2. Fit calibration relationship on OOF predictions
    # --------------------------------------------------------

    print(
        "Fitting calibration relationship "
        "using OOF predictions..."
    )

    X_oof = oof[
        [SELECTED_OOF_COLUMN]
    ]

    y_oof = oof["Actual_CAP"]

    calibration_model = LinearRegression()

    calibration_model.fit(
        X_oof,
        y_oof
    )

    calibration_slope = float(
        calibration_model.coef_[0]
    )

    calibration_intercept = float(
        calibration_model.intercept_
    )

    print(
        f"Calibration intercept: "
        f"{calibration_intercept:.6f}"
    )

    print(
        f"Calibration slope: "
        f"{calibration_slope:.6f}"
    )

    print()


    # --------------------------------------------------------
    # 3. Load finalized model
    # --------------------------------------------------------

    print("Loading finalized patient-state model...")

    model = joblib.load(MODEL_PATH)

    print(
        f"Model loaded from: {MODEL_PATH}"
    )

    print()


    # --------------------------------------------------------
    # 4. Load NHANES data
    # --------------------------------------------------------

    print("Loading NHANES data...")

    df = pd.read_csv(DATA_PATH)

    df["SEQN"] = df["SEQN"].astype(int)

    print(
        f"Total NHANES rows: {len(df)}"
    )

    print()


    # --------------------------------------------------------
    # 5. Identify fixed test participants
    # --------------------------------------------------------

    print("Loading fixed test split...")

    split = pd.read_csv(SPLIT_PATH)

    test_ids = (
        split.loc[
            split["partition"].astype(str).str.lower()
            == "test",
            "SEQN"
        ]
        .astype(int)
        .tolist()
    )

    print(
        f"Fixed test participants: {len(test_ids)}"
    )

    print()


    # --------------------------------------------------------
    # 6. Construct exact test dataset
    # --------------------------------------------------------

    test_df = (
        df[
            df["SEQN"].isin(test_ids)
        ]
        .copy()
        .set_index("SEQN")
        .loc[test_ids]
        .reset_index()
    )

    if len(test_df) != len(test_ids):
        raise ValueError(
            "Mismatch between split manifest and "
            "NHANES data."
        )

    print(
        f"Test rows loaded: {len(test_df)}"
    )

    print()


    # --------------------------------------------------------
    # 7. Generate predictions from exact saved model
    # --------------------------------------------------------

    print(
        "Generating predictions from the "
        "finalized saved model..."
    )

    if not hasattr(model, "feature_names_in_"):
        raise ValueError(
            "Saved model does not contain "
            "feature_names_in_."
        )

    input_features = list(
        model.feature_names_in_
    )

    missing_features = [
        feature
        for feature in input_features
        if feature not in test_df.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing model features:\n"
            f"{missing_features}"
        )

    X_test = test_df[
        input_features
    ].copy()

    y_test = test_df[
        "LUXCAPM"
    ].copy()

    predictions = model.predict(
        X_test
    )


    # --------------------------------------------------------
    # 8. Apply OOF-derived calibration
    # --------------------------------------------------------

    calibrated_predictions = (
        calibration_intercept
        + calibration_slope * predictions
    )


    # --------------------------------------------------------
    # 9. Calculate metrics
    # --------------------------------------------------------

    raw_metrics = regression_metrics(
        y_test,
        predictions
    )

    calibrated_metrics = regression_metrics(
        y_test,
        calibrated_predictions
    )

    print("Raw test performance:")

    for name, value in raw_metrics.items():
        print(
            f"  {name}: {value:.6f}"
        )

    print()

    print("Calibrated test performance:")

    for name, value in calibrated_metrics.items():
        print(
            f"  {name}: {value:.6f}"
        )

    print()


    # --------------------------------------------------------
    # 10. Save test predictions
    # --------------------------------------------------------

    results = pd.DataFrame({
        "SEQN": test_df["SEQN"].values,
        "Actual_CAP": y_test.values,
        "Raw_prediction": predictions,
        "Calibrated_prediction": calibrated_predictions,
        "Raw_error": predictions - y_test.values,
        "Calibrated_error": (
            calibrated_predictions
            - y_test.values
        ),
    })

    results_path = (
        OUTPUT_DIR
        / "calibration_test_predictions.csv"
    )

    results.to_csv(
        results_path,
        index=False
    )


    # --------------------------------------------------------
    # 11. Save calibration parameters
    # --------------------------------------------------------

    parameters = pd.DataFrame({
        "parameter": [
            "calibration_intercept",
            "calibration_slope",
            "oof_rows",
            "test_rows",
        ],
        "value": [
            calibration_intercept,
            calibration_slope,
            len(oof),
            len(test_df),
        ],
    })

    parameters_path = (
        OUTPUT_DIR
        / "calibration_parameters.csv"
    )

    parameters.to_csv(
        parameters_path,
        index=False
    )


    # --------------------------------------------------------
    # 12. Save metrics
    # --------------------------------------------------------

    metrics_df = pd.DataFrame({
        "metric": list(raw_metrics.keys()),
        "raw": list(raw_metrics.values()),
        "calibrated": [
            calibrated_metrics[key]
            for key in raw_metrics.keys()
        ],
    })

    metrics_path = (
        OUTPUT_DIR
        / "calibration_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False
    )


    # --------------------------------------------------------
    # 13. Calibration bins
    # --------------------------------------------------------

    results["prediction_bin"] = pd.qcut(
        results["Raw_prediction"],
        q=10,
        duplicates="drop"
    )

    calibration_bins = (
        results
        .groupby(
            "prediction_bin",
            observed=True
        )
        .agg(
            n=("Actual_CAP", "size"),
            mean_predicted=(
                "Raw_prediction",
                "mean"
            ),
            mean_observed=(
                "Actual_CAP",
                "mean"
            ),
            mean_calibrated=(
                "Calibrated_prediction",
                "mean"
            ),
        )
        .reset_index()
    )

    bins_path = (
        OUTPUT_DIR
        / "calibration_bins.csv"
    )

    calibration_bins.to_csv(
        bins_path,
        index=False
    )


    # --------------------------------------------------------
    # 14. Calibration plot
    # --------------------------------------------------------

    print("Creating calibration plot...")

    plt.figure(
        figsize=(8, 7)
    )

    plt.plot(
        calibration_bins["mean_predicted"],
        calibration_bins["mean_observed"],
        marker="o",
        label="Raw prediction"
    )

    plt.plot(
        calibration_bins["mean_calibrated"],
        calibration_bins["mean_observed"],
        marker="o",
        label="Calibrated prediction"
    )

    minimum = min(
        calibration_bins["mean_predicted"].min(),
        calibration_bins["mean_calibrated"].min(),
        calibration_bins["mean_observed"].min(),
    )

    maximum = max(
        calibration_bins["mean_predicted"].max(),
        calibration_bins["mean_calibrated"].max(),
        calibration_bins["mean_observed"].max(),
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
        label="Perfect calibration"
    )

    plt.xlabel(
        "Mean predicted CAP (dB/m)"
    )

    plt.ylabel(
        "Mean observed CAP (dB/m)"
    )

    plt.title(
        "CAP Regression Calibration"
    )

    plt.legend()

    plt.tight_layout()

    plot_path = (
        OUTPUT_DIR
        / "cap_calibration_plot.png"
    )

    plt.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    # --------------------------------------------------------
    # 15. Final output
    # --------------------------------------------------------

    print()
    print("Calibration analysis completed.")

    print(
        f"Calibration parameters: "
        f"{parameters_path}"
    )

    print(
        f"Test predictions: "
        f"{results_path}"
    )

    print(
        f"Metrics: "
        f"{metrics_path}"
    )

    print(
        f"Calibration bins: "
        f"{bins_path}"
    )

    print(
        f"Calibration plot: "
        f"{plot_path}"
    )


if __name__ == "__main__":
    main()