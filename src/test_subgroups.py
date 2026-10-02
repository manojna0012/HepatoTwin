import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.cap_features import CAPFeatures, RAW_FEATURES
from src.train_final_model import ROOT, TARGET

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor


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


def metrics(y, pred):

    return {
        "MAE": mean_absolute_error(y, pred),
        "RMSE": np.sqrt(
            mean_squared_error(y, pred)
        ),
        "R2": r2_score(y, pred),
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

    split_dir = ROOT / "data/results/second_pass"

    train_ids = pd.read_csv(
        split_dir /
        "oof_predictions.csv"
    )["SEQN"].to_numpy()

    test_ids = pd.read_csv(
        split_dir /
        "test_predictions.csv"
    )["SEQN"].to_numpy()

    return (
        data.loc[train_ids].copy(),
        data.loc[test_ids].copy(),
    )


def build_features(train, test):

    transformer = CAPFeatures()

    train_data = transformer.fit_transform(train)
    test_data = transformer.transform(test)

    columns = transformer.get_feature_names_out()

    X_train = pd.DataFrame(
        train_data,
        index=train.index,
        columns=columns,
    )

    X_test = pd.DataFrame(
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

    selected = list(RAW_FEATURES) + ratios

    return (
        X_train[selected],
        X_test[selected],
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


def evaluate_group(
    name,
    mask,
    y,
    predictions,
):

    if mask.sum() < 10:
        print(
            f"{name}: too few participants"
        )
        return

    actual = y[mask]
    predicted = predictions[mask]

    result = metrics(
        actual,
        predicted,
    )

    residual = actual - predicted

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(f"n = {mask.sum()}")

    print(
        f"Actual mean CAP = "
        f"{actual.mean():.2f}"
    )

    print(
        f"Predicted mean CAP = "
        f"{predicted.mean():.2f}"
    )

    print(
        f"Mean residual = "
        f"{residual.mean():.2f}"
    )

    print(
        f"MAE = "
        f"{result['MAE']:.4f}"
    )

    print(
        f"RMSE = "
        f"{result['RMSE']:.4f}"
    )

    print(
        f"R² = "
        f"{result['R2']:.4f}"
    )


def main():

    train, test = load_data()

    X_train, X_test = build_features(
        train,
        test,
    )

    y_train = train[TARGET]
    y_test = test[TARGET]

    model = make_model()

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_test
    )

    # --------------------------------------------------
    # Overall
    # --------------------------------------------------

    evaluate_group(
        "ALL TEST PARTICIPANTS",
        np.ones(len(test), dtype=bool),
        y_test,
        predictions,
    )

    # --------------------------------------------------
    # AGE GROUPS
    # --------------------------------------------------

    age = test["RIDAGEYR"]

    evaluate_group(
        "AGE < 20",
        age < 20,
        y_test,
        predictions,
    )

    evaluate_group(
        "AGE 20-39",
        (age >= 20) & (age < 40),
        y_test,
        predictions,
    )

    evaluate_group(
        "AGE 40-59",
        (age >= 40) & (age < 60),
        y_test,
        predictions,
    )

    evaluate_group(
        "AGE >= 60",
        age >= 60,
        y_test,
        predictions,
    )

    # --------------------------------------------------
    # SEX
    # --------------------------------------------------

    sex = test["RIAGENDR"]

    # NHANES coding:
    # 1 = male
    # 2 = female

    evaluate_group(
        "MALE",
        sex == 1,
        y_test,
        predictions,
    )

    evaluate_group(
        "FEMALE",
        sex == 2,
        y_test,
        predictions,
    )

    # --------------------------------------------------
    # BMI GROUPS
    # --------------------------------------------------

    bmi = test["BMXBMI"]

    evaluate_group(
        "BMI < 25",
        bmi < 25,
        y_test,
        predictions,
    )

    evaluate_group(
        "BMI 25-30",
        (bmi >= 25) & (bmi < 30),
        y_test,
        predictions,
    )

    evaluate_group(
        "BMI >= 30",
        bmi >= 30,
        y_test,
        predictions,
    )


if __name__ == "__main__":
    main()