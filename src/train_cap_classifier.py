
"""
Exploratory CAP classification experiments for HepatoTwin.

Experiments:
1. Binary: S0 vs S1-S3
2. Multiclass: S0, S1, S2, S3

Uses the existing eligible NHANES cohort and CAPFeatures transformer.
All preprocessing is fitted inside each CV fold.

Run from the project root:
    python -m src.train_cap_classifier

IMPORTANT:
The CAP thresholds below are provisional and must be verified before
interpreting these classes as clinical stages.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

from src.train_final_model import (
    ROOT,
    TARGET,
    CATEGORICAL,
    load_cohort,
    make_pipeline,
)
from src.cap_features import CAPFeatures, RAW_FEATURES, RAW_NUMERIC, ENGINEERED


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DATA_PATH = ROOT / "data/processed/nhanes_merged.csv"

OUTPUT_DIR = ROOT / "data/results/experiments/classification"

RANDOM_STATE = 42
N_SPLITS = 5

# EXPLORATORY CUTS ONLY.
# Verify these against the intended clinical reference.
CAP_CUTOFFS = [248, 268, 280]
STAGE_NAMES = ["S0", "S1", "S2", "S3"]

XGB_PARAMS = {
    "n_estimators": 600,
    "learning_rate": 0.02,
    "max_depth": 4,
    "min_child_weight": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "reg_lambda": 3.0,
    "reg_alpha": 0.5,
    "random_state": RANDOM_STATE,
    "n_jobs": 2,
    "verbosity": 0,
}


# ---------------------------------------------------------
# Data loading
# ---------------------------------------------------------

def load_classification_data():
    """Load the established eligible cohort and its full features."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    # Reuse the exact cohort eligibility rules from the existing model.
    cohort = load_cohort(DATA_PATH)

    full_data = pd.read_csv(DATA_PATH)

    missing = sorted(set(RAW_FEATURES) - set(full_data.columns))
    if missing:
        raise ValueError(f"Missing required raw features: {missing}")

    # Keep the same eligible participants, but retrieve the expanded inputs.
    feature_data = cohort[["SEQN", TARGET]].merge(
        full_data[["SEQN"] + RAW_FEATURES],
        on="SEQN",
        how="left",
        validate="one_to_one",
    )

    if feature_data["SEQN"].duplicated().any():
        raise ValueError("Duplicate participant IDs found.")

    X = feature_data[RAW_FEATURES].copy()
    cap = feature_data[TARGET].to_numpy(dtype=float)

    # Derive exploratory stages from the observed CAP target.
    # bins are left-inclusive: S0 < 248, S1 [248,268),
    # S2 [268,280), S3 >= 280.
    y_stage = np.digitize(cap, CAP_CUTOFFS).astype(int)

    if set(np.unique(y_stage)) != {0, 1, 2, 3}:
        raise ValueError(
            "All four stages must be represented. Check CAP cutoffs."
        )

    y_binary = (y_stage > 0).astype(int)

    print("\n--- Cohort ---")
    print(f"Eligible participants: {len(X)}")
    print(f"Raw input features: {len(RAW_FEATURES)}")
    print(f"Engineered features: {len(ENGINEERED)}")

    print("\n--- Exploratory stage counts ---")
    for stage_id, stage_name in enumerate(STAGE_NAMES):
        count = int((y_stage == stage_id).sum())
        percentage = 100 * count / len(y_stage)
        print(f"{stage_name}: {count:,} ({percentage:.2f}%)")

    print("\n--- Binary class counts ---")
    print(f"S0: {(y_binary == 0).sum():,}")
    print(f"S1-S3: {(y_binary == 1).sum():,}")

    return feature_data, X, cap, y_stage, y_binary


# ---------------------------------------------------------
# Model construction
# ---------------------------------------------------------

def build_classifier(binary=True):
    """Build a fold-safe preprocessing and XGBoost pipeline."""

    params = dict(XGB_PARAMS)

    if binary:
        params.update({
            "objective": "binary:logistic",
            "eval_metric": "logloss",
        })
    else:
        params.update({
            "objective": "multi:softprob",
            "num_class": 4,
            "eval_metric": "mlogloss",
        })

    estimator = XGBClassifier(**params)

    # make_pipeline handles numeric imputation and categorical encoding.
    # CAPFeatures computes the existing deterministic engineered features.
    return Pipeline([
        ("features", CAPFeatures()),
        ("preprocessing_model", make_pipeline(
            numeric=RAW_NUMERIC + ENGINEERED,
            estimator=estimator,
            add_indicator=True,
        )),
    ])


# ---------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------

def run_experiment(name, X, y, binary=True):
    """Generate out-of-fold predictions and evaluate two baselines."""

    n_classes = 2 if binary else 4
    class_ids = list(range(n_classes))
    class_names = ["S0", "Elevated"] if binary else STAGE_NAMES

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    model = build_classifier(binary=binary)

    # Every participant receives predictions from a model that did not
    # train on that participant.
    oof_pred = np.full(len(y), -1, dtype=int)
    oof_proba = np.full((len(y), n_classes), np.nan)
    baseline_pred = np.full(len(y), -1, dtype=int)

    fold_rows = []

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]
        y_train = y[train_idx]
        y_valid = y[valid_idx]

        fitted_model = clone(model)

        # Compensate for class imbalance during training.
        sample_weights = compute_sample_weight(
            class_weight="balanced",
            y=y_train,
        )

        fitted_model.fit(
            X_train,
            y_train,
            preprocessing_model__model__sample_weight=sample_weights,
        )

        pred = fitted_model.predict(X_valid).astype(int)
        proba = fitted_model.predict_proba(X_valid)

        oof_pred[valid_idx] = pred
        oof_proba[valid_idx] = proba

        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X_train, y_train)
        baseline_pred[valid_idx] = dummy.predict(X_valid)

        fold_rows.append({
            "experiment": name,
            "fold": fold,
            "n_validation": len(valid_idx),
            "macro_f1": f1_score(
                y_valid, pred, average="macro", zero_division=0
            ),
            "balanced_accuracy": balanced_accuracy_score(y_valid, pred),
            "accuracy": accuracy_score(y_valid, pred),
        })

        print(
            f"{name} | Fold {fold}/{N_SPLITS} | "
            f"Macro F1={fold_rows[-1]['macro_f1']:.4f} | "
            f"Balanced accuracy="
            f"{fold_rows[-1]['balanced_accuracy']:.4f}",
            flush=True,
        )

    if (oof_pred < 0).any() or np.isnan(oof_proba).any():
        raise RuntimeError("Some observations lack out-of-fold predictions.")

    # Pooled out-of-fold metrics.
    metrics = {
        "experiment": name,
        "n_samples": len(y),
        "macro_f1": f1_score(
            y, oof_pred, average="macro", zero_division=0
        ),
        "weighted_f1": f1_score(
            y, oof_pred, average="weighted", zero_division=0
        ),
        "balanced_accuracy": balanced_accuracy_score(y, oof_pred),
        "accuracy": accuracy_score(y, oof_pred),
        "baseline_macro_f1": f1_score(
            y, baseline_pred, average="macro", zero_division=0
        ),
        "baseline_balanced_accuracy": balanced_accuracy_score(
            y, baseline_pred
        ),
        "baseline_accuracy": accuracy_score(y, baseline_pred),
    }

    if binary:
        metrics.update({
            "precision_elevated": precision_score(
                y, oof_pred, pos_label=1, zero_division=0
            ),
            "recall_elevated": recall_score(
                y, oof_pred, pos_label=1, zero_division=0
            ),
            "roc_auc": roc_auc_score(y, oof_proba[:, 1]),
            "pr_auc": average_precision_score(y, oof_proba[:, 1]),
        })
    else:
        metrics["roc_auc_ovr_macro"] = roc_auc_score(
            y,
            oof_proba,
            labels=class_ids,
            multi_class="ovr",
            average="macro",
        )

    report = classification_report(
        y,
        oof_pred,
        labels=class_ids,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y, oof_pred, labels=class_ids)

    print(f"\n--- {name}: pooled out-of-fold metrics ---")
    print(pd.Series(metrics).to_string())

    print("\nClassification report:")
    print(pd.DataFrame(report).T.round(4).to_string())

    print("\nConfusion matrix (rows=actual, columns=predicted):")
    print(pd.DataFrame(
        cm,
        index=[f"Actual_{c}" for c in class_names],
        columns=[f"Predicted_{c}" for c in class_names],
    ).to_string())

    return {
        "metrics": metrics,
        "fold_metrics": fold_rows,
        "report": report,
        "confusion_matrix": cm,
        "oof_predictions": oof_pred,
        "oof_probabilities": oof_proba,
        "baseline_predictions": baseline_pred,
        "class_names": class_names,
    }


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data, X, cap, y_stage, y_binary = load_classification_data()

    print("\nStarting binary classification experiment...")
    binary_result = run_experiment(
        "binary_s0_vs_s1_s3",
        X,
        y_binary,
        binary=True,
    )

    print("\nStarting four-class classification experiment...")
    multiclass_result = run_experiment(
        "multiclass_s0_s3",
        X,
        y_stage,
        binary=False,
    )

    # Aggregate metrics.
    pd.DataFrame([
        binary_result["metrics"],
        multiclass_result["metrics"],
    ]).to_csv(OUTPUT_DIR / "cv_metrics.csv", index=False)

    all_fold_metrics = (
        binary_result["fold_metrics"]
        + multiclass_result["fold_metrics"]
    )
    pd.DataFrame(all_fold_metrics).to_csv(
        OUTPUT_DIR / "fold_metrics.csv",
        index=False,
    )

    # Class-specific reports.
    pd.DataFrame(binary_result["report"]).T.to_csv(
        OUTPUT_DIR / "binary_classification_report.csv"
    )
    pd.DataFrame(multiclass_result["report"]).T.to_csv(
        OUTPUT_DIR / "multiclass_classification_report.csv"
    )

    pd.DataFrame(
        binary_result["confusion_matrix"],
        index=["Actual_S0", "Actual_Elevated"],
        columns=["Predicted_S0", "Predicted_Elevated"],
    ).to_csv(OUTPUT_DIR / "binary_confusion_matrix.csv")

    pd.DataFrame(
        multiclass_result["confusion_matrix"],
        index=[f"Actual_{s}" for s in STAGE_NAMES],
        columns=[f"Predicted_{s}" for s in STAGE_NAMES],
    ).to_csv(OUTPUT_DIR / "multiclass_confusion_matrix.csv")

    # Out-of-fold predictions. SEQN is retained for auditing, so review
    # this file carefully before sharing it outside the project.
    predictions = pd.DataFrame({
        "SEQN": data["SEQN"].to_numpy(),
        "Actual_CAP": cap,
        "Actual_stage": [STAGE_NAMES[i] for i in y_stage],
        "Actual_binary": y_binary,
        "Binary_predicted": binary_result["oof_predictions"],
        "Binary_probability_S0": binary_result["oof_probabilities"][:, 0],
        "Binary_probability_elevated": binary_result["oof_probabilities"][:, 1],
        "Multiclass_predicted": [
            STAGE_NAMES[i]
            for i in multiclass_result["oof_predictions"]
        ],
    })

    for i, stage in enumerate(STAGE_NAMES):
        predictions[f"Multiclass_probability_{stage}"] = (
            multiclass_result["oof_probabilities"][:, i]
        )

    predictions.to_csv(OUTPUT_DIR / "oof_predictions.csv", index=False)

    metadata = {
        "dataset": str(DATA_PATH),
        "target": TARGET,
        "n_samples": len(X),
        "raw_features": RAW_FEATURES,
        "engineered_features": ENGINEERED,
        "binary_definition": "S0 vs S1-S3",
        "multiclass_labels": STAGE_NAMES,
        "provisional_cutoffs": CAP_CUTOFFS,
        "cutoff_status": "Exploratory only; clinical reference not yet verified",
        "cv": {
            "method": "StratifiedKFold",
            "n_splits": N_SPLITS,
            "shuffle": True,
            "random_state": RANDOM_STATE,
        },
        "class_weighting": "balanced sample weights within each training fold",
        "xgboost_parameters": XGB_PARAMS,
        "versions": {
            "xgboost": xgboost.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "limitations": [
            "CAP categories use provisional thresholds and are not validated diagnoses.",
            "Probability calibration has not yet been performed.",
            "NHANES survey weights and complex survey design are not modeled.",
            "Out-of-fold predictions are internal validation, not external validation.",
            "Classification performance does not directly measure continuous CAP R2.",
        ],
    }

    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\n--- Final comparison ---")
    print(pd.DataFrame([
        binary_result["metrics"],
        multiclass_result["metrics"],
    ]).round(4).to_string(index=False))

    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
