"""
HepatoTwin LDL incremental feature experiment.

Compares:
A. Existing 46-feature model
B. Existing model + LDL
C. Existing model + LDL + missingness indicator

The original train/test participants are preserved.
Feature selection is based on training CV only.
The existing model/results are not modified.
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

from src.cap_features import CAPFeatures
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


OUTPUT_DIR = ROOT / "data/results/ldl_experiment"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------

def calculate_metrics(y_true, y_pred):

    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(
            mean_squared_error(y_true, y_pred)
        ),
        "R2": r2_score(y_true, y_pred),
    }


# ---------------------------------------------------------
# LOAD SAME TRAIN / TEST SPLIT
# ---------------------------------------------------------

def load_data():

    source = ROOT / "data/processed/nhanes_merged.csv"

    data = (
        pd.read_csv(source)
        .set_index("SEQN")
    )

    data = data.loc[
        data["LUAXSTAT"].eq(1)
        & data[TARGET].notna()
    ].copy()

    previous_dir = ROOT / "data/results/second_pass"

    train_ids = pd.read_csv(
        previous_dir / "oof_predictions.csv"
    )["SEQN"].to_numpy()

    test_ids = pd.read_csv(
        previous_dir / "test_predictions.csv"
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
# BUILD BASE 46 FEATURES
# ---------------------------------------------------------

def build_features(train, test):

    transformer = CAPFeatures()

    train_features = transformer.fit_transform(
        train
    )

    test_features = transformer.transform(
        test
    )

    columns = transformer.get_feature_names_out()

    train_features = pd.DataFrame(
        train_features,
        index=train.index,
        columns=columns,
    )

    test_features = pd.DataFrame(
        test_features,
        index=test.index,
        columns=columns,
    )

    return train_features, test_features


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
                XGBRegressor(**PARAMS),
            ),
        ]
    )


# ---------------------------------------------------------
# EVALUATE
# ---------------------------------------------------------

def evaluate(
    name,
    X_train,
    y_train,
    X_test,
    y_test,
):

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    model = make_model()

    # Training-only CV
    cv_pred = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=cv,
        n_jobs=1,
    )

    cv_metrics = calculate_metrics(
        y_train,
        cv_pred,
    )

    # Fit on full training set
    model.fit(
        X_train,
        y_train,
    )

    test_pred = model.predict(
        X_test
    )

    test_metrics = calculate_metrics(
        y_test,
        test_pred,
    )

    print(
        f"CV R²:    {cv_metrics['R2']:.4f}"
    )

    print(
        f"CV RMSE:  {cv_metrics['RMSE']:.4f}"
    )

    print(
        f"CV MAE:   {cv_metrics['MAE']:.4f}"
    )

    print(
        f"Test R²:  {test_metrics['R2']:.4f}"
    )

    print(
        f"Test RMSE:{test_metrics['RMSE']:.4f}"
    )

    print(
        f"Test MAE: {test_metrics['MAE']:.4f}"
    )

    return {
        "Model": name,

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

    # -----------------------------------------------------
    # Base features
    # -----------------------------------------------------

    X_train, X_test = build_features(
        train,
        test,
    )

    print(
        f"\nBase feature count: "
        f"{X_train.shape[1]}"
    )

    results = []

    # -----------------------------------------------------
    # MODEL A: CURRENT MODEL
    # -----------------------------------------------------

    results.append(
        evaluate(
            "A - Current 46 features",
            X_train,
            y_train,
            X_test,
            y_test,
        )
    )

    # -----------------------------------------------------
    # MODEL B: ADD LDL
    # -----------------------------------------------------

    X_train_ldl = X_train.copy()
    X_test_ldl = X_test.copy()

    X_train_ldl["LDL"] = train["LBDLDL"]
    X_test_ldl["LDL"] = test["LBDLDL"]

    results.append(
        evaluate(
            "B - Current + LDL",
            X_train_ldl,
            y_train,
            X_test_ldl,
            y_test,
        )
    )

    # -----------------------------------------------------
    # MODEL C: LDL + MISSINGNESS INDICATOR
    # -----------------------------------------------------

    X_train_ldl_missing = X_train_ldl.copy()
    X_test_ldl_missing = X_test_ldl.copy()

    X_train_ldl_missing[
        "LDL_missing"
    ] = train["LBDLDL"].isna().astype(int)

    X_test_ldl_missing[
        "LDL_missing"
    ] = test["LBDLDL"].isna().astype(int)

    results.append(
        evaluate(
            "C - Current + LDL + missing indicator",
            X_train_ldl_missing,
            y_train,
            X_test_ldl_missing,
            y_test,
        )
    )

    # -----------------------------------------------------
    # SAVE
    # -----------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_DIR / "ldl_comparison.csv",
        index=False,
    )

    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)

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