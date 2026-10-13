"""Leakage-aware stacking experiment for HepatoTwin CAP prediction."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, StackingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor

from src.cap_features import CAPFeatures, RAW_NUMERIC, ENGINEERED
from src.train_final_model import ROOT, TARGET, load_cohort, make_pipeline, metrics
from src.train_engineered_model import SUPPLIED_PARAMS, build_model

SOURCE = ROOT / "data/processed/nhanes_merged.csv"
SPLIT_DIR = ROOT / "data/results/second_pass"
MODEL_DIR = ROOT / "data/results/engineered"
OUT = ROOT / "data/results/experiments/stacking"


def make_forest_pipeline(estimator):
    """Use the same engineered features and missing-value indicators."""
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder

    preprocessing = ColumnTransformer([
        ("numeric", SimpleImputer(
            strategy="median", keep_empty_features=True,
            add_indicator=True
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


def main():
    OUT.mkdir(parents=True, exist_ok=True)

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
        raise ValueError("Unexpected train/test split sizes.")

    selection = json.loads((MODEL_DIR / "selection.json").read_text())
    reference = joblib.load(MODEL_DIR / "patient_state_model.joblib")
    X, y = train[selection["features"]], train[TARGET]
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    xgb = clone_model = build_model(
        {**SUPPLIED_PARAMS, "n_estimators": 600},
        engineered=True,
        indicators=True,
    )

    extra_trees = make_forest_pipeline(ExtraTreesRegressor(
        n_estimators=200,
        min_samples_leaf=3,
        max_features=0.8,
        random_state=42,
        n_jobs=1,
    ))

    random_forest = make_forest_pipeline(RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=3,
        max_features=0.8,
        random_state=42,
        n_jobs=1,
    ))

    # Each base estimator's predictions for a training row are generated
    # without fitting that estimator on that row.
    stack = StackingRegressor(
        estimators=[
            ("xgb", xgb),
            ("extra_trees", extra_trees),
            ("random_forest", random_forest),
        ],
        final_estimator=Ridge(alpha=10.0),
        cv=KFold(n_splits=5, shuffle=True, random_state=17),
        passthrough=False,
        n_jobs=1,
    )

    print("Running five-fold CV: stacking ensemble", flush=True)
    with threadpool_limits(limits=2):
        pred = cross_val_predict(stack, X, y, cv=cv, n_jobs=1)

    result = {"Model": "Stacking (XGBoost + Extra Trees + Random Forest)",
              **metrics(y, pred)}

    pd.DataFrame([result]).to_csv(OUT / "cv_metrics.csv", index=False)
    pd.DataFrame({
        "SEQN": train.index,
        "Actual_CAP": y.to_numpy(),
        "Stacking_prediction": pred,
    }).to_csv(OUT / "oof_predictions.csv", index=False)

    (OUT / "metadata.json").write_text(json.dumps({
        "target": TARGET,
        "units": "dB/m",
        "train_n": len(train),
        "holdout_n": len(test),
        "outer_cv": "KFold(5, shuffle=True, random_state=42)",
        "stack_cv": "KFold(5, shuffle=True, random_state=17)",
        "meta_model": "Ridge(alpha=10.0)",
        "base_models": ["XGBoost", "Extra Trees", "Random Forest"],
        "holdout_used_for_selection": False,
        "note": "Stacked out-of-fold predictions; holdout not evaluated."
    }, indent=2))

    print("\nStacking CV results:")
    print(pd.DataFrame([result]).to_string(index=False))
    print(f"Saved results to: {OUT}", flush=True)


if __name__ == "__main__":
    main()
