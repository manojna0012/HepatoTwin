from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor

from src.train_final_model import (
    ROOT,
    TARGET,
    make_pipeline,
)
from src.cap_features import (
    CAPFeatures,
    RAW_FEATURES,
    RAW_NUMERIC,
    ENGINEERED,
)


# ============================================================
# Configuration
# ============================================================

DATA_PATH = ROOT / "data" / "processed" / "nhanes_merged.csv"
OUTPUT_DIR = ROOT / "data" / "results" / "experiments" / "hybrid_regression"

# Exploratory cutoffs only. Verify against an appropriate source
# before describing these labels as clinical stages.
CAP_CUTOFFS = [248, 268, 280]

OUTER_FOLDS = 5
INNER_FOLDS = 5
SEED = 42

BASE_NUMERIC = list(dict.fromkeys(RAW_NUMERIC + ENGINEERED))


# ============================================================
# Feature transformer for regression with optional probabilities
# ============================================================

class HybridCAPFeatures(BaseEstimator, TransformerMixin):
    """
    Engineer CAP features from raw inputs and then append probability
    columns. This prevents CAPFeatures from receiving probability columns
    as if they were raw clinical inputs.
    """

    def __init__(self, extra_features=None):
        self.extra_features = extra_features

    def fit(self, X, y=None):
        self.extra_features_ = list(self.extra_features or [])
        missing = [c for c in self.extra_features_ if c not in X.columns]
        if missing:
            raise ValueError(
                f"Expected probability columns are missing: {missing}"
            )

        self.raw_columns_ = [
            c for c in X.columns if c not in self.extra_features_
        ]

        self.cap_features_ = CAPFeatures()
        self.cap_features_.fit(X[self.raw_columns_], y)
        return self

    def transform(self, X):
        missing_raw = [c for c in self.raw_columns_ if c not in X.columns]
        missing_extra = [
            c for c in self.extra_features_ if c not in X.columns
        ]
        if missing_raw or missing_extra:
            raise ValueError(
                f"Missing raw columns: {missing_raw}; "
                f"missing probability columns: {missing_extra}"
            )

        raw = X[self.raw_columns_]
        engineered = self.cap_features_.transform(raw)

        if isinstance(engineered, pd.DataFrame):
            result = engineered.copy()
        else:
            raise TypeError(
                "CAPFeatures.transform() must return a pandas DataFrame "
                "for this hybrid pipeline. Check src/cap_features.py."
            )

        if self.extra_features_:
            probabilities = X[self.extra_features_].copy()
            probabilities.index = result.index
            result = result.join(probabilities)

        return result


# ============================================================
# Model factories
# ============================================================

def make_classifier(binary=False):
    """Create the classification pipeline used to generate probabilities."""

    model = XGBClassifier(
        objective="binary:logistic" if binary else "multi:softprob",
        num_class=None if binary else 4,
        n_estimators=600,
        learning_rate=0.02,
        max_depth=4,
        min_child_weight=8,
        subsample=0.8,
        colsample_bytree=0.6,
        reg_lambda=3,
        reg_alpha=0.5,
        eval_metric="logloss" if binary else "mlogloss",
        random_state=SEED,
        n_jobs=-1,
    )

    preprocessing = make_pipeline(
        numeric=BASE_NUMERIC,
        estimator=model,
        add_indicator=True,
    )

    return Pipeline([
        ("features", CAPFeatures()),
        ("model", preprocessing),
    ])


def make_regressor(extra_numeric=None):
    """Create baseline or hybrid XGBoost regression pipeline."""

    extra_numeric = list(extra_numeric or [])
    numeric = list(dict.fromkeys(BASE_NUMERIC + extra_numeric))

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=600,
        learning_rate=0.02,
        max_depth=4,
        min_child_weight=8,
        subsample=0.8,
        colsample_bytree=0.6,
        reg_lambda=3,
        reg_alpha=0.5,
        random_state=SEED,
        n_jobs=-1,
    )

    preprocessing = make_pipeline(
        numeric=numeric,
        estimator=model,
        add_indicator=True,
    )

    return Pipeline([
        (
            "features",
            HybridCAPFeatures(extra_features=extra_numeric),
        ),
        ("model", preprocessing),
    ])


def fit_classifier(model, X, y):
    """Fit a classifier with balanced class weights on its training rows."""

    weights = compute_sample_weight(class_weight="balanced", y=y)

    inner_model = model.named_steps["model"]
    final_step = list(inner_model.named_steps)[-1]

    fit_kwargs = {
        f"model__{final_step}__sample_weight": weights
    }
    model.fit(X, y, **fit_kwargs)
    return model


# ============================================================
# Leakage-aware classifier probability generation
# ============================================================

def make_oof_probabilities(X_train, y_train, binary=False):
    """
    Generate training probabilities using inner-fold predictions:
    each row's classifier did not train on that row.
    """

    y_train = np.asarray(y_train)
    n_classes = 2 if binary else 4
    probabilities = np.zeros((len(X_train), n_classes), dtype=float)

    splitter = StratifiedKFold(
        n_splits=INNER_FOLDS,
        shuffle=True,
        random_state=SEED,
    )

    for inner_train, inner_valid in splitter.split(X_train, y_train):
        classifier = make_classifier(binary=binary)
        fit_classifier(
            classifier,
            X_train.iloc[inner_train],
            y_train[inner_train],
        )
        probabilities[inner_valid] = classifier.predict_proba(
            X_train.iloc[inner_valid]
        )

    return probabilities


def make_outer_features(X_train, y_train, X_valid, binary=False):
    """
    Training rows receive inner-fold OOF probabilities. Validation rows
    receive probabilities from a classifier fitted only on outer training.
    """

    train_prob = make_oof_probabilities(
        X_train, y_train, binary=binary
    )

    full_classifier = make_classifier(binary=binary)
    fit_classifier(full_classifier, X_train, y_train)
    valid_prob = full_classifier.predict_proba(X_valid)

    if binary:
        columns = ["binary_prob_S0", "binary_prob_elevated"]
    else:
        columns = [
            "multiclass_prob_S0",
            "multiclass_prob_S1",
            "multiclass_prob_S2",
            "multiclass_prob_S3",
        ]

    train_extra = pd.DataFrame(
        train_prob, columns=columns, index=X_train.index
    )
    valid_extra = pd.DataFrame(
        valid_prob, columns=columns, index=X_valid.index
    )

    return train_extra, valid_extra


# ============================================================
# Evaluation
# ============================================================

def calculate_metrics(actual, predicted):
    return {
        "r2": r2_score(actual, predicted),
        "rmse": np.sqrt(mean_squared_error(actual, predicted)),
        "mae": mean_absolute_error(actual, predicted),
        "mean_prediction_bias": float(
            np.mean(np.asarray(predicted) - np.asarray(actual))
        ),
    }


def main():
    warnings.filterwarnings("default")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load all columns needed for engineered features.
    df = pd.read_csv(DATA_PATH)

    required = ["SEQN", "LUAXSTAT", TARGET] + list(RAW_FEATURES)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    # Match the eligibility rule used by the existing regression pipeline.
    df = df.loc[
        (df["LUAXSTAT"] == 1) & df[TARGET].notna()
    ].copy()

    if df["SEQN"].duplicated().any():
        raise ValueError("Duplicate SEQN values found in eligible cohort.")

    cohort = df.reset_index(drop=True)

    X = cohort[RAW_FEATURES].copy()
    y = cohort[TARGET].astype(float).to_numpy()

    # Provisional labels, not confirmed clinical stages.
    classes = np.digitize(y, CAP_CUTOFFS).astype(int)
    binary_labels = (classes > 0).astype(int)

    print(f"Eligible participants: {len(cohort)}")
    print(f"CAP target: {TARGET}")
    print("Class counts (S0, S1, S2, S3):")
    print(pd.Series(classes).value_counts().sort_index().to_dict())
    print("\nStarting nested hybrid regression evaluation...")

    outer = KFold(
        n_splits=OUTER_FOLDS,
        shuffle=True,
        random_state=SEED,
    )

    model_names = [
        "baseline_xgboost",
        "binary_hybrid",
        "four_class_hybrid",
    ]

    predictions = {
        name: np.full(len(cohort), np.nan)
        for name in model_names
    }
    fold_rows = []

    for fold, (train_idx, valid_idx) in enumerate(outer.split(X), start=1):
        print(f"\nOuter fold {fold}/{OUTER_FOLDS}")

        X_train = X.iloc[train_idx].copy()
        X_valid = X.iloc[valid_idx].copy()
        y_train = y[train_idx]
        y_valid = y[valid_idx]

        # Baseline regression
        baseline = make_regressor()
        baseline.fit(X_train, y_train)
        predictions["baseline_xgboost"][valid_idx] = baseline.predict(X_valid)

        # Binary hybrid
        binary_train_prob, binary_valid_prob = make_outer_features(
            X_train, binary_labels[train_idx], X_valid, binary=True
        )
        X_train_binary = X_train.join(binary_train_prob)
        X_valid_binary = X_valid.join(binary_valid_prob)

        binary_regressor = make_regressor(
            extra_numeric=["binary_prob_S0", "binary_prob_elevated"]
        )
        binary_regressor.fit(X_train_binary, y_train)
        predictions["binary_hybrid"][valid_idx] = (
            binary_regressor.predict(X_valid_binary)
        )

        # Four-class hybrid
        multi_train_prob, multi_valid_prob = make_outer_features(
            X_train, classes[train_idx], X_valid, binary=False
        )
        X_train_multi = X_train.join(multi_train_prob)
        X_valid_multi = X_valid.join(multi_valid_prob)

        multi_regressor = make_regressor(
            extra_numeric=[
                "multiclass_prob_S0",
                "multiclass_prob_S1",
                "multiclass_prob_S2",
                "multiclass_prob_S3",
            ]
        )
        multi_regressor.fit(X_train_multi, y_train)
        predictions["four_class_hybrid"][valid_idx] = (
            multi_regressor.predict(X_valid_multi)
        )

        for name in model_names:
            fold_pred = predictions[name][valid_idx]
            fold_rows.append({
                "fold": fold,
                "model": name,
                **calculate_metrics(y_valid, fold_pred),
            })

        print("Fold results:")
        for name in model_names:
            fold_pred = predictions[name][valid_idx]
            print(f"  {name}: R2={r2_score(y_valid, fold_pred):.4f}")

    # Pooled out-of-fold metrics
    summary_rows = []
    for name in model_names:
        summary_rows.append({
            "model": name,
            "participants": len(y),
            **calculate_metrics(y, predictions[name]),
        })

    summary = pd.DataFrame(summary_rows)
    fold_results = pd.DataFrame(fold_rows)

    summary.to_csv(OUTPUT_DIR / "summary_metrics.csv", index=False)
    fold_results.to_csv(OUTPUT_DIR / "fold_metrics.csv", index=False)

    prediction_frame = pd.DataFrame({
        "SEQN": cohort["SEQN"].to_numpy(),
        "actual_CAP": y,
        **predictions,
    })
    prediction_frame.to_csv(
        OUTPUT_DIR / "oof_predictions.csv", index=False
    )

    metadata = {
        "target": TARGET,
        "eligible_participants": len(cohort),
        "outer_folds": OUTER_FOLDS,
        "inner_folds": INNER_FOLDS,
        "random_seed": SEED,
        "provisional_cutoffs": CAP_CUTOFFS,
        "cutoffs_clinically_verified": False,
        "evaluation": "Nested outer-fold OOF predictions",
        "models": model_names,
        "note": (
            "Internal cross-validation only; not external validation. "
            "Classification probabilities are not calibrated clinical risks. "
            "The class cutoffs are exploratory and require verification."
        ),
    }
    with open(OUTPUT_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n========== POOLED OOF RESULTS ==========")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    baseline_r2 = summary.loc[
        summary["model"] == "baseline_xgboost", "r2"
    ].iloc[0]

    print("\nR2 change relative to baseline:")
    for _, row in summary.iterrows():
        print(f"  {row['model']}: {row['r2'] - baseline_r2:+.4f}")

    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
