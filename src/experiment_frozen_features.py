"""Frozen-feature comparison for the HepatoTwin CAP model.

Uses the existing train/test participant IDs and the selected model's
feature engineering and missing-value indicator configuration.
Does not evaluate or tune on the holdout test set.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor

from src.cap_features import CAPFeatures, RAW_FEATURES, RAW_NUMERIC, ENGINEERED
from src.train_final_model import ROOT, TARGET, load_cohort, make_pipeline, metrics
from src.train_engineered_model import SUPPLIED_PARAMS, build_model


SOURCE = ROOT / "data/processed/nhanes_merged.csv"
SPLIT_DIR = ROOT / "data/results/second_pass"
SELECTED_DIR = ROOT / "data/results/engineered"
OUT = ROOT / "data/results/experiments/frozen_features"


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # Reuse the established participant split.
    data = pd.read_csv(SOURCE).set_index("SEQN")
    cohort = load_cohort(SOURCE)
    data = data.loc[cohort.SEQN]

    train_ids = pd.read_csv(SPLIT_DIR / "oof_predictions.csv")["SEQN"]
    test_ids = pd.read_csv(SPLIT_DIR / "test_predictions.csv")["SEQN"]

    train = data.loc[train_ids]
    test = data.loc[test_ids]

    if set(train.index) & set(test.index):
        raise ValueError("Train/test participant IDs overlap.")
    if len(train) != 7216 or len(test) != 1805:
        raise ValueError(
            f"Unexpected split sizes: train={len(train)}, test={len(test)}"
        )

    selection = json.loads((SELECTED_DIR / "selection.json").read_text())
    selected_features = selection["features"]
    indicators = selection["selected_model"] == "ratios_600_trees_missing_flags"

    X = train[RAW_FEATURES]
    y = train[TARGET]
    cv = list(KFold(n_splits=5, shuffle=True, random_state=42).split(X))

    # The current selected model is the reference.
    reference = joblib.load(SELECTED_DIR / "patient_state_model.joblib")

    # Keep the same engineered transforms and missingness-indicator setup
    # for tree models. Ridge gets scaling after the same imputation/encoding.
    def make_tree_pipeline(estimator):
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder

        preprocessing = ColumnTransformer([
            ("numeric", SimpleImputer(
                strategy="median", keep_empty_features=True,
                add_indicator=indicators
            ), RAW_NUMERIC + ENGINEERED),
            ("categorical", Pipeline([
                ("impute", SimpleImputer(
                    strategy="most_frequent", keep_empty_features=True
                )),
                ("encode", OneHotEncoder(
                    categories=[[1., 2.], [1., 2., 3., 4., 6., 7.]],
                    drop="first", handle_unknown="error", sparse_output=False
                )),
            ]), ["RIAGENDR", "RIDRETH3"]),
        ])
        return Pipeline([
            ("features", CAPFeatures()),
            ("preprocess", preprocessing),
            ("model", estimator),
        ])

    # Use the same XGBoost settings as the selected 600-tree model.
    xgb = XGBRegressor(
        **{**SUPPLIED_PARAMS, "n_estimators": 600},
        objective="reg:squarederror", random_state=42, n_jobs=2
    )

    models = {
        "Selected XGBoost reference": clone(reference),
        "Random Forest": make_tree_pipeline(RandomForestRegressor(
            n_estimators=500, min_samples_leaf=3, max_features=0.8,
            random_state=42, n_jobs=2
        )),
        "Extra Trees": make_tree_pipeline(ExtraTreesRegressor(
            n_estimators=500, min_samples_leaf=3, max_features=0.8,
            random_state=42, n_jobs=2
        )),
    }

    # Ridge requires scaled transformed inputs. The preprocessing and
    # transformations are fitted independently within each CV fold.
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder

    ridge_preprocessing = ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(
                strategy="median", keep_empty_features=True,
                add_indicator=indicators
            )),
            ("scale", StandardScaler()),
        ]), RAW_NUMERIC + ENGINEERED),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(
                strategy="most_frequent", keep_empty_features=True
            )),
            ("encode", OneHotEncoder(
                categories=[[1., 2.], [1., 2., 3., 4., 6., 7.]],
                drop="first", handle_unknown="error", sparse_output=False
            )),
        ]), ["RIAGENDR", "RIDRETH3"]),
    ])
    models["Ridge"] = Pipeline([
        ("features", CAPFeatures()),
        ("preprocess", ridge_preprocessing),
        ("model", Ridge(alpha=100.0)),
    ])

    rows = []
    predictions = pd.DataFrame({
        "SEQN": train.index,
        "Actual_CAP": y.to_numpy(),
    })

    with threadpool_limits(limits=2):
        for name, model in models.items():
            print(f"Running five-fold CV: {name}", flush=True)
            oof_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=1)
            predictions[name] = oof_pred
            rows.append({"Model": name, **metrics(y, oof_pred)})
            pd.DataFrame(rows).sort_values("RMSE").to_csv(
                OUT / "cv_metrics.csv", index=False
            )
            predictions.to_csv(OUT / "oof_predictions.csv", index=False)
            print(rows[-1], flush=True)

    pd.DataFrame(rows).sort_values("RMSE").to_csv(
        OUT / "cv_metrics.csv", index=False
    )
    (OUT / "experiment_metadata.json").write_text(json.dumps({
        "target": TARGET,
        "units": "dB/m",
        "train_n": len(train),
        "holdout_n": len(test),
        "cv": "KFold(n_splits=5, shuffle=True, random_state=42)",
        "raw_features": RAW_FEATURES,
        "engineered_features": ENGINEERED,
        "missing_indicators": indicators,
        "holdout_used_for_selection": False,
        "note": "CV results are development estimates; holdout not evaluated here."
    }, indent=2))
    print("\nCV comparison (lower RMSE/MAE is better; higher R2 is better):")
    print(pd.DataFrame(rows).sort_values("RMSE").to_string(index=False))
    print(f"\nResults saved to: {OUT}")


if __name__ == "__main__":
    main()
