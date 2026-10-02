"""
HepatoTwin sample-weighting experiment.

Goal:
Test whether giving additional training weight to CAP extremes
reduces regression-to-the-mean behavior.

Model selection is based ONLY on 5-fold CV on the training set.
The original test set is used only for final diagnostic comparison.
"""

import numpy as np
import pandas as pd

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

RATIOS = [
    "WHtR",
    "AST_ALT",
    "TG_HDL",
    "TC_HDL",
]


# ---------------------------------------------------------
# LOAD DATA
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

    split_dir = ROOT / "data/results/second_pass"

    train_ids = pd.read_csv(
        split_dir / "oof_predictions.csv"
    )["SEQN"].to_numpy()

    test_ids = pd.read_csv(
        split_dir / "test_predictions.csv"
    )["SEQN"].to_numpy()

    train = data.loc[train_ids].copy()
    test = data.loc[test_ids].copy()

    return train, test


# ---------------------------------------------------------
# FEATURES
# ---------------------------------------------------------

def build_features(train, test):

    transformer = CAPFeatures()

    train_transformed = transformer.fit_transform(train)
    test_transformed = transformer.transform(test)

    columns = transformer.get_feature_names_out()

    X_train_all = pd.DataFrame(
        train_transformed,
        index=train.index,
        columns=columns,
    )

    X_test_all = pd.DataFrame(
        test_transformed,
        index=test.index,
        columns=columns,
    )

    features = list(RAW_FEATURES) + RATIOS

    return (
        X_train_all[features],
        X_test_all[features],
    )


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
# SAMPLE WEIGHTS
# ---------------------------------------------------------

def make_weights(y, strength):

    """
    Give larger weights to observations farther from
    the training-set median CAP.

    strength = 0 means ordinary XGBoost.

    We normalize the weights so the average weight
    remains approximately 1.
    """

    median = np.median(y)

    distance = np.abs(y - median)

    # Scale distance using MAD for robustness.
    mad = np.median(np.abs(y - median))

    if mad == 0:
        return np.ones(len(y))

    scaled = distance / mad

    weights = 1.0 + strength * scaled

    # Avoid excessively large weights.
    weights = np.clip(
        weights,
        1.0,
        4.0,
    )

    # Normalize mean weight to 1.
    weights = weights / weights.mean()

    return weights


# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------

def metrics(y_true, y_pred):

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
# CV EXPERIMENT
# ---------------------------------------------------------

def evaluate_strength(
    name,
    strength,
    X_train,
    y_train,
    X_test,
    y_test,
):

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    # -----------------------------------------------------
    # IMPORTANT:
    # weights are computed separately inside each training
    # fold, so validation information is never used.
    # -----------------------------------------------------

    oof = np.zeros(len(y_train))

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X_train),
        start=1,
    ):

        X_fold = X_train.iloc[train_idx]
        y_fold = y_train.iloc[train_idx]

        X_val = X_train.iloc[val_idx]

        weights = make_weights(
            y_fold.to_numpy(),
            strength,
        )

        model = make_model()

        model.fit(
            X_fold,
            y_fold,
            model__sample_weight=weights,
        )

        oof[val_idx] = model.predict(
            X_val
        )

    cv_result = metrics(
        y_train,
        oof,
    )

    # -----------------------------------------------------
    # Final training fit
    # -----------------------------------------------------

    final_model = make_model()

    final_weights = make_weights(
        y_train.to_numpy(),
        strength,
    )

    final_model.fit(
        X_train,
        y_train,
        model__sample_weight=final_weights,
    )

    test_pred = final_model.predict(
        X_test
    )

    test_result = metrics(
        y_test,
        test_pred,
    )

    print("\n" + "=" * 65)
    print(name)
    print("=" * 65)

    print(f"Strength: {strength}")

    print(
        f"CV R2:     {cv_result['R2']:.4f}"
    )
    print(
        f"CV RMSE:   {cv_result['RMSE']:.4f}"
    )
    print(
        f"CV MAE:    {cv_result['MAE']:.4f}"
    )

    print(
        f"Test R2:   {test_result['R2']:.4f}"
    )
    print(
        f"Test RMSE: {test_result['RMSE']:.4f}"
    )
    print(
        f"Test MAE:  {test_result['MAE']:.4f}"
    )

    return {
        "Model": name,
        "Strength": strength,

        "CV_R2": cv_result["R2"],
        "CV_RMSE": cv_result["RMSE"],
        "CV_MAE": cv_result["MAE"],

        "Test_R2": test_result["R2"],
        "Test_RMSE": test_result["RMSE"],
        "Test_MAE": test_result["MAE"],
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
        f"Training participants: {len(train)}"
    )

    print(
        f"Test participants: {len(test)}"
    )

    print(
        f"Features: {X_train.shape[1]}"
    )

    # -----------------------------------------------------
    # Compare weighting strengths
    # -----------------------------------------------------

    # 0 = ordinary XGBoost
    # Higher values increasingly emphasize extremes.

    strengths = [
        0.0,
        0.25,
        0.50,
        0.75,
    ]

    results = []

    for strength in strengths:

        results.append(
            evaluate_strength(
                f"XGBoost - weighting strength {strength}",
                strength,
                X_train,
                y_train,
                X_test,
                y_test,
            )
        )

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "CV_RMSE"
    )

    output_dir = (
        ROOT /
        "data/results/sample_weighting"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        output_dir /
        "sample_weighting_comparison.csv",
        index=False,
    )

    print("\n" + "=" * 65)
    print("SAMPLE WEIGHTING RESULTS")
    print("=" * 65)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print(
        "\nSelected by lowest CV RMSE:"
    )

    print(
        results_df.iloc[0]["Model"]
    )

    print(
        "\nSaved to:",
        output_dir,
    )


if __name__ == "__main__":
    main()