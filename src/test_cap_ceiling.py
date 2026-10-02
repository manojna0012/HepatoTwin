"""
HepatoTwin CAP ceiling sensitivity analysis.

Compares model performance:
1. On the full eligible cohort
2. After excluding CAP == 400
3. Separately reports performance for CAP < 400 and CAP == 400

Uses the current best feature configuration:
36 raw features + 4 ratios.

IMPORTANT:
This is a sensitivity analysis.
CAP == 400 observations are NOT being removed from the main model.
"""

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

from src.cap_features import CAPFeatures, RAW_FEATURES
from src.train_final_model import ROOT, TARGET


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


def load_data():

    data = (
        pd.read_csv(
            ROOT /
            "data/processed/nhanes_merged.csv"
        )
        .set_index("SEQN")
    )

    data = data.loc[
        data["LUAXSTAT"].eq(1)
        & data[TARGET].notna()
    ].copy()

    split_dir = (
        ROOT /
        "data/results/second_pass"
    )

    train_ids = pd.read_csv(
        split_dir /
        "oof_predictions.csv"
    )["SEQN"].to_numpy()

    test_ids = pd.read_csv(
        split_dir /
        "test_predictions.csv"
    )["SEQN"].to_numpy()

    train = data.loc[train_ids].copy()
    test = data.loc[test_ids].copy()

    return train, test


def build_features(train, test):

    transformer = CAPFeatures()

    train_data = transformer.fit_transform(
        train
    )

    test_data = transformer.transform(
        test
    )

    columns = (
        transformer
        .get_feature_names_out()
    )

    train_features = pd.DataFrame(
        train_data,
        index=train.index,
        columns=columns,
    )

    test_features = pd.DataFrame(
        test_data,
        index=test.index,
        columns=columns,
    )

    ratios = [
        "WHtR",
        "AST_ALT",
        "TG_HDL",
        "TC_HDL",
    ]

    selected = (
        list(RAW_FEATURES)
        + ratios
    )

    return (
        train_features[selected],
        test_features[selected],
    )


def make_model():

    return Pipeline([
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
    ])


def main():

    train, test = load_data()

    X_train, X_test = build_features(
        train,
        test,
    )

    y_train = train[TARGET]
    y_test = test[TARGET]

    print(
        f"Training participants: "
        f"{len(train)}"
    )

    print(
        f"Test participants: "
        f"{len(test)}"
    )

    print(
        f"Training CAP=400: "
        f"{(y_train == 400).sum()}"
    )

    print(
        f"Test CAP=400: "
        f"{(y_test == 400).sum()}"
    )

    # --------------------------------------------------
    # MODEL
    # --------------------------------------------------

    model = make_model()

    model.fit(
        X_train,
        y_train,
    )

    test_pred = model.predict(
        X_test
    )

    # --------------------------------------------------
    # FULL TEST SET
    # --------------------------------------------------

    full_metrics = metrics(
        y_test,
        test_pred,
    )

    print("\n" + "=" * 60)
    print("FULL TEST SET")
    print("=" * 60)

    print(
        f"n = {len(y_test)}"
    )

    print(
        f"MAE  = {full_metrics['MAE']:.4f}"
    )

    print(
        f"RMSE = {full_metrics['RMSE']:.4f}"
    )

    print(
        f"R²   = {full_metrics['R2']:.4f}"
    )

    # --------------------------------------------------
    # TEST WITHOUT CAP = 400
    # --------------------------------------------------

    non_ceiling = y_test < 400

    non_ceiling_metrics = metrics(
        y_test[non_ceiling],
        test_pred[non_ceiling],
    )

    print("\n" + "=" * 60)
    print("TEST SET: CAP < 400")
    print("=" * 60)

    print(
        f"n = {non_ceiling.sum()}"
    )

    print(
        f"MAE  = "
        f"{non_ceiling_metrics['MAE']:.4f}"
    )

    print(
        f"RMSE = "
        f"{non_ceiling_metrics['RMSE']:.4f}"
    )

    print(
        f"R²   = "
        f"{non_ceiling_metrics['R2']:.4f}"
    )

    # --------------------------------------------------
    # CAP = 400 ONLY
    # --------------------------------------------------

    ceiling = y_test == 400

    if ceiling.sum() > 0:

        ceiling_errors = (
            np.abs(
                y_test[ceiling]
                - test_pred[ceiling]
            )
        )

        print("\n" + "=" * 60)
        print("TEST SET: CAP = 400")
        print("=" * 60)

        print(
            f"n = {ceiling.sum()}"
        )

        print(
            f"Mean predicted CAP = "
            f"{test_pred[ceiling].mean():.4f}"
        )

        print(
            f"Mean absolute error = "
            f"{ceiling_errors.mean():.4f}"
        )

        print(
            f"Median absolute error = "
            f"{np.median(ceiling_errors):.4f}"
        )

    # --------------------------------------------------
    # ERROR BY CAP RANGE
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("ERROR BY TRUE CAP RANGE")
    print("=" * 60)

    bins = [
        0,
        180,
        220,
        260,
        300,
        350,
        400,
    ]

    labels = [
        "<=180",
        "181-220",
        "221-260",
        "261-300",
        "301-350",
        "351-400",
    ]

    ranges = pd.cut(
        y_test,
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    error_df = pd.DataFrame({
        "CAP": y_test,
        "Prediction": test_pred,
        "Range": ranges,
    })

    error_df["AbsoluteError"] = (
        error_df["CAP"]
        - error_df["Prediction"]
    ).abs()

    grouped = (
        error_df
        .groupby(
            "Range",
            observed=False,
        )
        .agg(
            n=("AbsoluteError", "size"),
            MAE=("AbsoluteError", "mean"),
        )
    )

    print(grouped)


if __name__ == "__main__":
    main()