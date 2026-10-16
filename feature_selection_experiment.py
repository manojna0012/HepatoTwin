"""
Leakage-safe feature-selection experiment for HepatoTwin.

Run from the project root:
    python feature_selection_experiment.py

Compares:
  1. Existing XGBoost pipeline with raw + engineered features
  2. LASSO-based selection on preprocessed features, then XGBoost
  3. RFE using ExtraTrees feature importance, then XGBoost

All feature selection is fitted inside each training fold.
Outputs are written to data/results/experiments/feature_selection/.
This script does not overwrite the selected model or existing results.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import TransformedTargetRegressor
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor

from src.cap_features import CAPFeatures, RAW_FEATURES, RAW_NUMERIC, ENGINEERED
from src.train_final_model import ROOT, TARGET, load_cohort, make_pipeline

SEED = 42
N_SPLITS = 5
MODEL_PARAMS = dict(
    n_estimators=600,
    learning_rate=0.02,
    max_depth=4,
    min_child_weight=8,
    subsample=0.8,
    colsample_bytree=0.6,
    reg_lambda=3.0,
    reg_alpha=0.5,
)
XGB_PARAMS = {**MODEL_PARAMS, "objective": "reg:squarederror", "random_state": SEED, "n_jobs": 2}


def score(y, pred):
    return {
        "R2": float(r2_score(y, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "MAE": float(mean_absolute_error(y, pred)),
        "Bias_pred_minus_actual": float(np.mean(pred - np.asarray(y))),
    }


def make_preprocessor():
    numeric = RAW_NUMERIC + ENGINEERED
    categorical = ["RIAGENDR", "RIDRETH3"]
    return ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median", keep_empty_features=True), numeric),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                        (
                            "encode",
                            OneHotEncoder(
                                categories=[[1.0, 2.0], [1.0, 2.0, 3.0, 4.0, 6.0, 7.0]],
                                drop="first",
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_xgb():
    return XGBRegressor(**XGB_PARAMS)


def make_baseline():
    # Use the same project's feature engineering and preprocessing conventions.
    regressor = make_pipeline(
        MODEL_PARAMS,
        numeric=RAW_NUMERIC + ENGINEERED,
        add_indicator=True,
    )
    return Pipeline([("features", CAPFeatures()), ("regressor", regressor)])


def make_selected_pipeline(selector, scale_for_selection=False):
    # Selection occurs after fold-specific imputation/encoding and before XGBoost.
    # LASSO receives standardized inputs; RFE uses the preprocessed features directly.
    steps = [
        ("features", CAPFeatures()),
        ("preprocess", make_preprocessor()),
    ]
    if scale_for_selection:
        steps.append(("scale", StandardScaler()))
    steps.extend([
        ("selector", selector),
        ("model", make_xgb()),
    ])
    return Pipeline(steps)


def main():
    source = ROOT / "data/processed/nhanes_merged.csv"
    out = ROOT / "data/results/experiments/feature_selection"
    out.mkdir(parents=True, exist_ok=True)

    cohort = load_cohort(source)
    all_data = pd.read_csv(source).set_index("SEQN")
    data = all_data.loc[cohort["SEQN"]].reset_index()
    X = data[RAW_FEATURES].copy()
    y = data[TARGET].astype(float).copy()

    # Confirm this cohort/order matches the established OOF record set when available.
    previous_oof_path = ROOT / "data/results/second_pass/oof_predictions.csv"
    if previous_oof_path.exists():
        old_oof = pd.read_csv(previous_oof_path)
        if "SEQN" in old_oof.columns:
            indexed = data.set_index("SEQN")
            if old_oof["SEQN"].isin(indexed.index).all():
                ordered = indexed.loc[old_oof["SEQN"]].reset_index()
                if len(ordered) == len(old_oof):
                    data = ordered
                    X = data[RAW_FEATURES].copy()
                    y = data[TARGET].astype(float).copy()
                    if "Actual_CAP" in old_oof.columns:
                        np.testing.assert_allclose(y, old_oof["Actual_CAP"].to_numpy())

    cv = list(KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED).split(X))
    candidates = {
        "baseline_xgb": make_baseline(),
        "lasso_selected_xgb": make_selected_pipeline(
            SelectFromModel(
                estimator=LassoCV(cv=3, max_iter=20000, n_jobs=1, random_state=SEED),
                threshold="median",
            ),
            scale_for_selection=True,
        ),
        "rfe_extratrees_selected_xgb": make_selected_pipeline(
            RFE(
                estimator=ExtraTreesRegressor(
                    n_estimators=150,
                    min_samples_leaf=3,
                    max_features=0.8,
                    n_jobs=2,
                    random_state=SEED,
                ),
                n_features_to_select=40,
                step=0.2,
            )
        ),
    }

    summary_rows = []
    prediction_frame = pd.DataFrame(
        {"SEQN": data["SEQN"].to_numpy(), "Actual_CAP": y.to_numpy()}
    )
    fold_rows = []

    with threadpool_limits(limits=2):
        for name, model in candidates.items():
            print(f"\nRunning {name} ...", flush=True)
            oof = np.full(len(y), np.nan, dtype=float)
            for fold_number, (train_idx, valid_idx) in enumerate(cv, start=1):
                fold_model = clone(model)
                fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])
                pred = fold_model.predict(X.iloc[valid_idx])
                oof[valid_idx] = pred
                fold_metrics = score(y.iloc[valid_idx], pred)
                fold_rows.append(
                    {"Model": name, "Fold": fold_number, "N_valid": len(valid_idx), **fold_metrics}
                )
                print(
                    f"  Fold {fold_number}: R2={fold_metrics['R2']:.4f}, "
                    f"RMSE={fold_metrics['RMSE']:.3f}, MAE={fold_metrics['MAE']:.3f}",
                    flush=True,
                )

            if np.isnan(oof).any():
                raise RuntimeError(f"Missing OOF predictions for {name}")
            summary_rows.append({"Model": name, "N": len(y), **score(y, oof)})
            prediction_frame[f"pred_{name}"] = oof

    summary = pd.DataFrame(summary_rows).sort_values("R2", ascending=False)
    summary.to_csv(out / "summary_metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(out / "fold_metrics.csv", index=False)
    prediction_frame.to_csv(out / "oof_predictions.csv", index=False)

    report = {
        "target": TARGET,
        "n_rows": int(len(data)),
        "cv": {"type": "KFold", "n_splits": N_SPLITS, "shuffle": True, "random_state": SEED},
        "selection_is_fold_local": True,
        "feature_selection": {
            "lasso": "StandardScaler + LassoCV(cv=3), SelectFromModel threshold=median",
            "rfe": "ExtraTreesRegressor, 40 transformed features, step=0.2",
        },
        "xgb_params": XGB_PARAMS,
        "note": "Exploratory CV comparison. Feature selection is fitted within each fold. "
                "Do not replace the selected model based on a small CV difference alone.",
    }
    (out / "experiment_config.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n=== POOLED OUT-OF-FOLD RESULTS ===")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print(f"\nSaved results to: {out}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
