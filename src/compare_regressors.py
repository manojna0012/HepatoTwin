"""
Compare XGBoost and HistGradientBoosting
using the same 40-feature representation.

Feature set:
- 36 raw features
- WHtR
- AST_ALT
- TG_HDL
- TC_HDL

Model selection is based ONLY on 5-fold CV
on the training partition.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor

from src.cap_features import CAPFeatures, RAW_FEATURES
from src.train_final_model import ROOT, TARGET


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

XGB_PARAMS = dict(
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


HGB_PARAMS = dict(
    learning_rate=0.05,
    max_iter=300,
    max_leaf_nodes=31,
    max_depth=None,
    min_samples_leaf=20,
    l2_regularization=1.0,
    random_state=42,
)


OUTPUT_DIR = ROOT / "data/results/regressor_comparison"
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
# LOAD SAME TRAIN / TEST SPLIT
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
# BUILD SAME 40 FEATURES
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

    ratios = [
        "WHtR",
        "AST_ALT",
        "TG_HDL",
        "TC_HDL",
    ]

    selected_features = (
        list(RAW_FEATURES)
        + ratios
    )

    return (
        X_train[selected_features],
        X_test[selected_features],
        selected_features,
    )


# ---------------------------------------------------------
# MODELS
# ---------------------------------------------------------

def make_xgb():

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
                    **XGB_PARAMS
                ),
            ),
        ]
    )


def make_hgb():

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
                HistGradientBoostingRegressor(
                    **HGB_PARAMS
                ),
            ),
        ]
    )


# ---------------------------------------------------------
# EVALUATE MODEL
# ---------------------------------------------------------

def evaluate(
    name,
    model,
    X_train,
    y_train,
    X_test,
    y_test,
):

    print("\n" + "=" * 65)
    print(name)
    print("=" * 65)

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    # -----------------------------------------------------
    # TRAINING-ONLY CV
    # -----------------------------------------------------

    cv_predictions = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=cv,
        n_jobs=1,
    )

    cv_metrics = calculate_metrics(
        y_train,
        cv_predictions,
    )

    # -----------------------------------------------------
    # FIT ON FULL TRAINING SET
    # -----------------------------------------------------

    model.fit(
        X_train,
        y_train,
    )

    test_predictions = model.predict(
        X_test
    )

    test_metrics = calculate_metrics(
        y_test,
        test_predictions,
    )

    print(
        f"CV R2:     {cv_metrics['R2']:.4f}"
    )

    print(
        f"CV RMSE:   {cv_metrics['RMSE']:.4f}"
    )

    print(
        f"CV MAE:    {cv_metrics['MAE']:.4f}"
    )

    print(
        f"Test R2:   {test_metrics['R2']:.4f}"
    )

    print(
        f"Test RMSE: {test_metrics['RMSE']:.4f}"
    )

    print(
        f"Test MAE:  {test_metrics['MAE']:.4f}"
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

    X_train, X_test, features = build_features(
        train,
        test,
    )

    print(
        f"\nNumber of features: "
        f"{len(features)}"
    )

    print("\nFeatures:")
    print(features)

    # -----------------------------------------------------
    # Compare regressors
    # -----------------------------------------------------

    models = {
        "XGBoost - 40 features":
            make_xgb(),

        "HistGradientBoosting - 40 features":
            make_hgb(),
    }

    results = []

    for name, model in models.items():

        results.append(
            evaluate(
                name,
                model,
                X_train,
                y_train,
                X_test,
                y_test,
            )
        )

    results_df = pd.DataFrame(results)

    # Selection criterion:
    # lowest CV RMSE
    results_df = results_df.sort_values(
        "CV_RMSE"
    )

    results_df.to_csv(
        OUTPUT_DIR /
        "regressor_comparison.csv",
        index=False,
    )

    print("\n" + "=" * 65)
    print("REGRESSOR COMPARISON")
    print("=" * 65)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print(
        "\nSelected by lowest CV RMSE:",
        results_df.iloc[0]["Model"],
    )

    print(
        "\nSaved to:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()