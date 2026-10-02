"""
Systematic feature-selection experiment for HepatoTwin CAP regression.

This script:
1. Uses the SAME fixed train/test participants as the current model.
2. Builds the existing 46-feature engineered representation.
3. Uses training-only CV to rank features.
4. Evaluates several top-k feature sets using the SAME CV folds.
5. Evaluates the selected feature sets once on the original holdout.

IMPORTANT:
- Do not use the holdout test set to choose the number of features.
- This script is an experiment; it does not overwrite the existing model.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor

from src.cap_features import CAPFeatures, RAW_FEATURES, ENGINEERED
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

TOP_K_VALUES = [46, 40, 35, 30, 25, 20, 15, 12, 10]

OUTPUT_DIR = ROOT / "data/results/rfe_experiment"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------

def metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------

def make_model(selected_features):
    """
    Build model after CAPFeatures transformation.

    We create the full engineered representation first,
    then select the requested columns.
    """

    numeric_features = selected_features

    preprocessing = ColumnTransformer(
        [
            (
                "numeric",
                SimpleImputer(
                    strategy="median",
                    keep_empty_features=True,
                ),
                numeric_features,
            )
        ],
        remainder="drop",
    )

    model = XGBRegressor(**PARAMS)

    return Pipeline(
        [
            ("preprocess", preprocessing),
            ("model", model),
        ]
    )


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

def load_data():

    source = ROOT / "data/processed/nhanes_merged.csv"

    data = pd.read_csv(source).set_index("SEQN")

    # Required cohort
    data = data.loc[
        data["LUAXSTAT"].eq(1)
        & data[TARGET].notna()
    ].copy()

    # Existing fixed split
    previous_dir = ROOT / "data/results/second_pass"

    old_oof = pd.read_csv(
        previous_dir / "oof_predictions.csv"
    )

    old_test = pd.read_csv(
        previous_dir / "test_predictions.csv"
    )

    train_ids = old_oof["SEQN"].to_numpy()
    test_ids = old_test["SEQN"].to_numpy()

    train = data.loc[train_ids].copy()
    test = data.loc[test_ids].copy()

    print(f"Training participants: {len(train)}")
    print(f"Test participants: {len(test)}")

    return train, test


# ---------------------------------------------------------
# CREATE ENGINEERED FEATURES
# ---------------------------------------------------------

def create_engineered_data(train, test):

    transformer = CAPFeatures()

    train_features = transformer.fit_transform(train)
    test_features = transformer.transform(test)

    train_features = pd.DataFrame(
        train_features,
        index=train.index,
        columns=transformer.get_feature_names_out(),
    )

    test_features = pd.DataFrame(
        test_features,
        index=test.index,
        columns=transformer.get_feature_names_out(),
    )

    return train_features, test_features


# ---------------------------------------------------------
# INITIAL FEATURE IMPORTANCE
# ---------------------------------------------------------

def rank_features(X, y):

    print("\nTraining full 46-feature model...")

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    fold_importances = []

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X), 1):

        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]

        model = Pipeline(
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

        model.fit(X_train, y_train)

        importance = model.named_steps[
            "model"
        ].feature_importances_

        fold_importances.append(importance)

        print(f"Fold {fold} completed")

    mean_importance = np.mean(
        fold_importances,
        axis=0,
    )

    ranking = pd.DataFrame(
        {
            "Feature": X.columns,
            "Importance": mean_importance,
        }
    ).sort_values(
        "Importance",
        ascending=False,
    )

    ranking["Rank"] = range(
        1,
        len(ranking) + 1,
    )

    return ranking


# ---------------------------------------------------------
# EVALUATE FEATURE COUNTS
# ---------------------------------------------------------

def evaluate_feature_sets(
    X_train,
    y_train,
    X_test,
    y_test,
    ranking,
):

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    results = []

    for k in TOP_K_VALUES:

        selected = ranking.head(k)["Feature"].tolist()

        print(
            f"\nEvaluating top {k} features..."
        )

        model = make_model(selected)

        oof_prediction = cross_val_predict(
            model,
            X_train[selected],
            y_train,
            cv=cv,
            n_jobs=1,
        )

        cv_metrics = metrics(
            y_train,
            oof_prediction,
        )

        # Fit ONLY after CV evaluation
        model.fit(
            X_train[selected],
            y_train,
        )

        test_prediction = model.predict(
            X_test[selected]
        )

        test_metrics = metrics(
            y_test,
            test_prediction,
        )

        results.append(
            {
                "N_features": k,

                "CV_MAE": cv_metrics["MAE"],
                "CV_RMSE": cv_metrics["RMSE"],
                "CV_R2": cv_metrics["R2"],

                "Test_MAE": test_metrics["MAE"],
                "Test_RMSE": test_metrics["RMSE"],
                "Test_R2": test_metrics["R2"],
            }
        )

        print(
            f"CV R2 = {cv_metrics['R2']:.4f}"
        )

        print(
            f"Test R2 = {test_metrics['R2']:.4f}"
        )

    return pd.DataFrame(results)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    train, test = load_data()

    y_train = train[TARGET]
    y_test = test[TARGET]

    print("\nCreating engineered features...")

    X_train, X_test = create_engineered_data(
        train,
        test,
    )

    print(
        f"Total engineered features: "
        f"{len(X_train.columns)}"
    )

    print("\nRanking features using training CV...")

    ranking = rank_features(
        X_train,
        y_train,
    )

    ranking.to_csv(
        OUTPUT_DIR / "feature_ranking.csv",
        index=False,
    )

    print("\nTOP FEATURES")
    print(
        ranking.head(20).to_string(
            index=False
        )
    )

    results = evaluate_feature_sets(
        X_train,
        y_train,
        X_test,
        y_test,
        ranking,
    )

    results.to_csv(
        OUTPUT_DIR / "feature_count_comparison.csv",
        index=False,
    )

    print("\n==============================")
    print("FEATURE COUNT COMPARISON")
    print("==============================")

    print(
        results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print(
        "\nSaved results to:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()