from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_predict
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

OUTPUT_DIR = ROOT / "data/results/ratio_ablation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def metrics(y, pred):
    return {
        "MAE": mean_absolute_error(y, pred),
        "RMSE": np.sqrt(mean_squared_error(y, pred)),
        "R2": r2_score(y, pred),
    }


def load_data():

    data = (
        pd.read_csv(ROOT / "data/processed/nhanes_merged.csv")
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

    return (
        data.loc[train_ids].copy(),
        data.loc[test_ids].copy(),
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


def evaluate(name, features, X_train, y_train, X_test, y_test):

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    model = make_model()

    cv_pred = cross_val_predict(
        model,
        X_train[features],
        y_train,
        cv=cv,
        n_jobs=1,
    )

    cv_m = metrics(y_train, cv_pred)

    model.fit(
        X_train[features],
        y_train,
    )

    test_pred = model.predict(
        X_test[features]
    )

    test_m = metrics(y_test, test_pred)

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print(f"CV R²:   {cv_m['R2']:.4f}")
    print(f"CV RMSE: {cv_m['RMSE']:.4f}")
    print(f"CV MAE:  {cv_m['MAE']:.4f}")
    print(f"Test R²: {test_m['R2']:.4f}")
    print(f"Test RMSE: {test_m['RMSE']:.4f}")
    print(f"Test MAE: {test_m['MAE']:.4f}")

    return {
        "Model": name,
        "CV_R2": cv_m["R2"],
        "CV_RMSE": cv_m["RMSE"],
        "CV_MAE": cv_m["MAE"],
        "Test_R2": test_m["R2"],
        "Test_RMSE": test_m["RMSE"],
        "Test_MAE": test_m["MAE"],
    }


def main():

    train, test = load_data()

    y_train = train[TARGET]
    y_test = test[TARGET]

    transformer = CAPFeatures()

    train_transformed = transformer.fit_transform(train)
    test_transformed = transformer.transform(test)

    columns = transformer.get_feature_names_out()

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

    results = []

    # Baseline
    results.append(
        evaluate(
            "Raw only",
            list(RAW_FEATURES),
            X_train,
            y_train,
            X_test,
            y_test,
        )
    )

    # Each ratio individually
    for ratio in ratios:

        results.append(
            evaluate(
                f"Raw + {ratio}",
                list(RAW_FEATURES) + [ratio],
                X_train,
                y_train,
                X_test,
                y_test,
            )
        )

    # All ratios
    results.append(
        evaluate(
            "Raw + all four ratios",
            list(RAW_FEATURES) + ratios,
            X_train,
            y_train,
            X_test,
            y_test,
        )
    )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_DIR / "individual_ratio_results.csv",
        index=False,
    )

    print("\nFINAL RESULTS")
    print("=" * 60)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


if __name__ == "__main__":
    main()