"""
HepatoTwin engineered-feature ablation experiment.

Models:
A. Raw features only
B. Raw + simple ratios
C. Raw + FLI/HSI
D. Raw + all engineered features

All models use:
- Same train/test participants
- Same 5-fold CV
- Same XGBoost parameters
- Same preprocessing
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_predict
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from xgboost import XGBRegressor

from src.cap_features import (
    CAPFeatures,
    RAW_FEATURES,
    ENGINEERED,
)

from src.train_final_model import ROOT, TARGET


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

PARAMS = dict(
    n_estimators=600,
    learning_rate=0.02,
    max_depth=4,
    min_child_weight=8,
    subsample=0.8,
    colsample_bytree=0.6,
    reg_lambda=3.0,
    reg_alpha=0.5,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=2,
)


OUTPUT_DIR = ROOT / "data/results/engineered_ablation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------

def calculate_metrics(y_true, y_pred):

    return {
        "MAE": mean_absolute_error(
            y_true,
            y_pred,
        ),

        "RMSE": np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        ),

        "R2": r2_score(
            y_true,
            y_pred,
        ),
    }


# ---------------------------------------------------------
# LOAD SAME SPLIT
# ---------------------------------------------------------

def load_data():

    source = (
        ROOT /
        "data/processed/nhanes_merged.csv"
    )

    data = (
        pd.read_csv(source)
        .set_index("SEQN")
    )

    data = data.loc[
        data["LUAXSTAT"].eq(1)
        & data[TARGET].notna()
    ].copy()

    split_dir = ROOT / "data/results/second_pass"

    train_ids = pd.read_csv(
        split_dir / "oof_predictions.csv"
    )["SEQN"].to_numpy()

    test_ids = pd.read_csv(
        split_dir / "test_predictions.csv"
    )["SEQN"].to_numpy()

    train = data.loc[train_ids].copy()
    test = data.loc[test_ids].copy()

    print(
        f"Training participants: {len(train)}"
    )

    print(
        f"Test participants: {len(test)}"
    )

    return train, test


# ---------------------------------------------------------
# BUILD ENGINEERED REPRESENTATION
# ---------------------------------------------------------

def build_features(train, test):

    transformer = CAPFeatures()

    train_transformed = transformer.fit_transform(
        train
    )

    test_transformed = transformer.transform(
        test
    )

    columns = (
        transformer
        .get_feature_names_out()
    )

    X_train = pd.DataFrame(
        train_transformed,
        index=train.index,
        columns=columns,
    )

    X_test = pd.DataFrame(
        test_transformed,
        index=test.index,
        columns=columns,
    )

    return X_train, X_test


# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------

def make_model():

    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    keep_empty_features=True,
                ),
            ),

            (
                "model",
                XGBRegressor(
                    **PARAMS
                ),
            ),
        ]
    )


# ---------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------

def evaluate(
    name,
    selected_features,
    X_train,
    y_train,
    X_test,
    y_test,
):

    print("\n" + "=" * 65)

    print(name)

    print("=" * 65)

    print(
        f"Number of features: "
        f"{len(selected_features)}"
    )

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    model = make_model()

    # -----------------------------------------------------
    # Training-only CV
    # -----------------------------------------------------

    cv_predictions = cross_val_predict(
        model,
        X_train[selected_features],
        y_train,
        cv=cv,
        n_jobs=1,
    )

    cv_metrics = calculate_metrics(
        y_train,
        cv_predictions,
    )

    # -----------------------------------------------------
    # Final fit on training data
    # -----------------------------------------------------

    model.fit(
        X_train[selected_features],
        y_train,
    )

    test_predictions = model.predict(
        X_test[selected_features]
    )

    test_metrics = calculate_metrics(
        y_test,
        test_predictions,
    )

    print(
        f"CV R²:   {cv_metrics['R2']:.4f}"
    )

    print(
        f"CV RMSE: {cv_metrics['RMSE']:.4f}"
    )

    print(
        f"CV MAE:  {cv_metrics['MAE']:.4f}"
    )

    print(
        f"Test R²: {test_metrics['R2']:.4f}"
    )

    print(
        f"Test RMSE: "
        f"{test_metrics['RMSE']:.4f}"
    )

    print(
        f"Test MAE: "
        f"{test_metrics['MAE']:.4f}"
    )

    return {
        "Model": name,
        "N_features": len(selected_features),

        "CV_R2": cv_metrics["R2"],
        "CV_RMSE": cv_metrics["RMSE"],
        "CV_MAE": cv_metrics["MAE"],

        "Test_R2": test_metrics["R2"],
        "Test_RMSE": test_metrics["RMSE"],
        "Test_MAE": test_metrics["MAE"],
    }


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    train, test = load_data()

    y_train = train[TARGET]
    y_test = test[TARGET]

    X_train, X_test = build_features(
        train,
        test,
    )

    print(
        f"\nTotal available features: "
        f"{len(X_train.columns)}"
    )

    # -----------------------------------------------------
    # Feature groups
    # -----------------------------------------------------

    raw = list(RAW_FEATURES)

    ratios = [
        "WHtR",
        "AST_ALT",
        "TG_HDL",
        "TC_HDL",
    ]

    indices = [
        "HSI_partial",
        "FLI_proxy",
    ]

    # The remaining engineered nutritional features
    remaining_engineered = [
        "energy_fraction_protein",
        "energy_fraction_carb",
        "energy_fraction_fat",
        "kcal_per_kg",
    ]

    all_engineered = (
        ratios
        + indices
        + remaining_engineered
    )

    # -----------------------------------------------------
    # Models
    # -----------------------------------------------------

    feature_sets = {

        "A - Raw only":
            raw,

        "B - Raw + ratios":
            raw + ratios,

        "C - Raw + FLI/HSI":
            raw + indices,

        "D - Raw + all engineered":
            raw + all_engineered,
    }

    results = []

    for name, features in feature_sets.items():

        results.append(
            evaluate(
                name,
                features,
                X_train,
                y_train,
                X_test,
                y_test,
            )
        )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_DIR /
        "engineered_feature_comparison.csv",
        index=False,
    )

    print("\n" + "=" * 65)

    print(
        "ENGINEERED FEATURE ABLATION RESULTS"
    )

    print("=" * 65)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print(
        "\nSaved to:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()