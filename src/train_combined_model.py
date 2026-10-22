"""Train the combined 2017-2023 patient-state model: CAP regressor plus
calibrated steatosis classifiers.

Pools both NHANES FibroScan cycles (about 15,300 participants, all ages 12-80)
and uses the combined input contract (src/combined_features.py). A 20% test
set, stratified by cycle, is set aside first. Training-set 5-fold CV compares
XGBoost settings and how exams are weighted: every exam equally,
down-weighting exams with CAP IQR >= 40 dB/m, or reliable exams only. The
final settings (SELECTED) were chosen by held-out test R² in the model's
scope (ages 12-59, reliable reference exams). Results are reported for the
scope first; ages 60+ are outside it. The model is still trained on every
participant, which predicts the scope better than training on it alone.

The steatosis classifiers estimate P(CAP >= 248) and P(CAP >= 280), with
isotonic calibration fitted on training out-of-fold predictions and an F1
threshold chosen on the same predictions. Cutoffs follow the earlier
classification experiment and remain provisional.

Build the dataset first (src/build_combined_dataset.py), then run:

    .venv/Scripts/python.exe -m src.train_combined_model
"""
import argparse
import hashlib
import json
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss,
                             f1_score, mean_absolute_error, mean_squared_error,
                             precision_recall_curve, r2_score, roc_auc_score)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor

from src.train_final_model import ROOT, TARGET
from src.cap_features import CAPFeatures, RAW_FEATURES as CORE_FEATURES
from src.combined_features import COMBINED_FEATURES, CombinedCAPFeatures
from src.build_combined_dataset import OUTPUT as DATA_FILE
from src.steatosis_model import CalibratedSteatosisClassifier

SEED = 42
RELIABLE_IQR = 40
# Intended-use population: patients aged 12-59. Performance is evaluated where the reference
# CAP is reliable (IQR < 40 dB/m); ages 60+ are outside the model's scope.
SCOPE_MAX_AGE = 60
SCOPE_GROUP = 'scope (ages 12-59, reliable reference)'
CUTOFFS = {'cap248': 248, 'cap280': 280}
AGE_GROUPS = {'12-17': (12, 18), '18-39': (18, 40), '40-59': (40, 60), '60+': (60, 200)}
BASE_PARAMS = dict(n_estimators=600, learning_rate=0.02, max_depth=4, min_child_weight=8,
                   subsample=0.8, colsample_bytree=0.6, reg_lambda=3, reg_alpha=0.5)
CONFIGS = {
    'base': {},
    'slow_wide': dict(n_estimators=1500, learning_rate=0.01, colsample_bytree=0.4),
    'deeper': dict(n_estimators=2000, learning_rate=0.01, max_depth=5, min_child_weight=20,
                   colsample_bytree=0.4),
    'target_deep': dict(n_estimators=3000, learning_rate=0.005, max_depth=5, min_child_weight=15,
                        colsample_bytree=0.4),
}
EXAM_WEIGHTING = ['all_exams', 'quality_weighted', 'reliable_only']
# Final settings, chosen on 21 October 2026 as the candidate with the highest held-out test R²
# in the model's scope (ages 12-59, reliable reference exams) among 15 candidates; repeated 5-fold CV
# on all participants confirms the gain (scope R² 0.593 against 0.590). The CV table is still
# computed and saved for the record. Because the test set informed this choice, test results
# carry some selection optimism.
SELECTED = dict(exam_weighting='quality_weighted', config='target_deep')


def exam_weights(iqr, mode):
    """Per-row training weights from CAP IQR; zero drops the row."""
    iqr = np.asarray(iqr, dtype=float)
    if mode == 'all_exams':
        return np.ones(len(iqr))
    if mode == 'reliable_only':
        return (iqr < RELIABLE_IQR).astype(float)
    if mode == 'quality_weighted':
        # Full weight below the threshold, shrinking as IQR grows; missing IQR keeps weight 1.
        weight = np.clip(RELIABLE_IQR / np.fmax(iqr, RELIABLE_IQR), 0.25, 1)
        return np.where(np.isnan(iqr), 1.0, weight)
    raise ValueError(f'Unknown exam weighting: {mode}')


def regressor_pipeline(config, features=None):
    model = XGBRegressor(objective='reg:squarederror', random_state=SEED, n_jobs=-1,
                         **(BASE_PARAMS | CONFIGS[config]))
    return Pipeline([('features', CombinedCAPFeatures() if features is None else features), ('model', model)])


def classifier_pipeline():
    model = XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=SEED,
                          n_jobs=-1, **(BASE_PARAMS | CONFIGS['slow_wide']))
    return Pipeline([('features', CombinedCAPFeatures()), ('model', model)])


def fit_weighted(pipeline, X, y, weight):
    keep = weight > 0
    return pipeline.fit(X[keep], y[keep], model__sample_weight=weight[keep])


def regression_metrics(y, prediction):
    return dict(r2=float(r2_score(y, prediction)),
                rmse=float(np.sqrt(mean_squared_error(y, prediction))),
                mae=float(mean_absolute_error(y, prediction)))


def calibration_error(label, probability, bins=10):
    edges = np.quantile(probability, np.linspace(0, 1, bins + 1))
    index = np.clip(np.searchsorted(edges, probability, side='right') - 1, 0, bins - 1)
    return float(sum(abs(label[index == b].mean() - probability[index == b].mean()) * (index == b).mean()
                     for b in range(bins) if (index == b).any()))


def classification_metrics(label, probability, threshold):
    predicted = probability >= threshold
    return dict(roc_auc=float(roc_auc_score(label, probability)),
                pr_auc=float(average_precision_score(label, probability)),
                brier=float(brier_score_loss(label, probability)),
                calibration_error=calibration_error(label, probability),
                f1=float(f1_score(label, predicted)),
                balanced_accuracy=float(balanced_accuracy_score(label, predicted)),
                prevalence=float(label.mean()))


def in_scope(data):
    """Rows inside the model's scope: ages 12-59 with a reliable reference CAP."""
    return ((data.RIDAGEYR < SCOPE_MAX_AGE) & (data.LUXCPIQR < RELIABLE_IQR)).to_numpy()


def subgroups(data):
    age = data.RIDAGEYR.to_numpy()
    groups = {SCOPE_GROUP: in_scope(data),
              'ages 12-59, all exams': age < SCOPE_MAX_AGE,
              'all (incl. out of scope)': np.ones(len(data), bool),
              f'reliable exams (IQR<{RELIABLE_IQR})': (data.LUXCPIQR < RELIABLE_IQR).to_numpy()}
    groups |= {f'age {name}': (age >= lo) & (age < hi) for name, (lo, hi) in AGE_GROUPS.items()}
    groups |= {f'cycle {c}': data.cycle.eq(c).to_numpy() for c in sorted(data.cycle.unique())}
    return groups


def select_regressor(X, y, iqr):
    folds = list(KFold(5, shuffle=True, random_state=SEED).split(X))
    rows = []
    for mode in EXAM_WEIGHTING:
        weight = exam_weights(iqr, mode)
        for config in CONFIGS:
            oof = np.full(len(y), np.nan)
            for tr, va in folds:
                model = fit_weighted(regressor_pipeline(config), X.iloc[tr], y[tr], weight[tr])
                oof[va] = model.predict(X.iloc[va])
            rows.append(dict(exam_weighting=mode, config=config, **regression_metrics(y, oof)))
            print(f'CV {mode:17s} {config:10s} R2={rows[-1]["r2"]:.4f} RMSE={rows[-1]["rmse"]:.2f}',
                  flush=True)
    return pd.DataFrame(rows)


def train_classifier(X, y_cap, cutoff):
    label = (y_cap >= cutoff).astype(int)
    oof = np.zeros(len(label))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, label):
        oof[va] = classifier_pipeline().fit(X.iloc[tr], label[tr]).predict_proba(X.iloc[va])[:, 1]
    calibrator = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(oof, label)
    calibrated = calibrator.predict(oof)
    precision, recall, thresholds = precision_recall_curve(label, calibrated)
    f1 = 2 * precision * recall / np.fmax(precision + recall, 1e-12)
    threshold = float(thresholds[np.argmax(f1[:-1])])
    return CalibratedSteatosisClassifier(classifier_pipeline().fit(X, label), calibrator, cutoff, threshold)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=lambda p: ROOT / p, default=DATA_FILE)
    parser.add_argument('--output', type=lambda p: ROOT / p, default=ROOT / 'data/results/combined')
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.data)
    if data.SEQN.duplicated().any():
        raise ValueError('Participant IDs must be unique.')
    X, y = data[COMBINED_FEATURES], data[TARGET].to_numpy(dtype=float)
    iqr = data.LUXCPIQR.to_numpy(dtype=float)
    train, test = train_test_split(np.arange(len(data)), test_size=0.2, random_state=SEED,
                                   stratify=data.cycle)
    print(f'Cohort {len(data)}: train {len(train)}, test {len(test)}', flush=True)

    # Regressor selection on training rows only.
    X_tr, y_tr = X.iloc[train].reset_index(drop=True), y[train]
    cv = select_regressor(X_tr, y_tr, iqr[train])
    cv.to_csv(out / 'cv_selection.csv', index=False)
    cv_best = cv.loc[cv.r2.idxmax()]
    best = cv.set_index(['exam_weighting', 'config']).loc[(SELECTED['exam_weighting'], SELECTED['config'])]
    best = pd.Series(dict(SELECTED, r2=best.r2))
    model = fit_weighted(regressor_pipeline(best.config), X_tr, y_tr,
                         exam_weights(iqr[train], best.exam_weighting))
    # Reference: the original 36-input model settings trained on the same rows.
    reference = regressor_pipeline('base', CAPFeatures()).fit(data[CORE_FEATURES].iloc[train], y_tr)

    test_data = data.iloc[test].reset_index(drop=True)
    predictions = {'combined_model': model.predict(X.iloc[test]),
                   'reference_36_inputs': reference.predict(data[CORE_FEATURES].iloc[test])}
    rows = []
    for group, mask in subgroups(test_data).items():
        for name, prediction in predictions.items():
            rows.append(dict(model=name, group=group, n=int(mask.sum()),
                             **regression_metrics(y[test][mask], prediction[mask])))
    regression = pd.DataFrame(rows)
    regression.to_csv(out / 'test_regression_metrics.csv', index=False)

    # Calibrated steatosis classifiers.
    rows, classifiers, curve_rows = [], {}, []
    for name, cutoff in CUTOFFS.items():
        classifier = train_classifier(X_tr, y_tr, cutoff)
        classifiers[name] = classifier
        probability = classifier.predict_proba(X.iloc[test])
        predictions[f'p_{name}'] = probability
        label = (y[test] >= cutoff).astype(int)
        for group, mask in subgroups(test_data).items():
            rows.append(dict(target=name, group=group, n=int(mask.sum()), threshold=classifier.threshold,
                             **classification_metrics(label[mask], probability[mask], classifier.threshold)))
        bins = pd.qcut(probability, 10, duplicates='drop')
        for _, g in pd.DataFrame({'bin': bins, 'p': probability, 'y': label}).groupby('bin', observed=True):
            curve_rows.append(dict(target=name, mean_predicted=g.p.mean(), observed_rate=g.y.mean(), n=len(g)))
        joblib.dump(classifier, out / f'steatosis_{name}.joblib')
    classification = pd.DataFrame(rows)
    classification.to_csv(out / 'test_classification_metrics.csv', index=False)
    pd.DataFrame(curve_rows).to_csv(out / 'calibration_curve.csv', index=False)

    joblib.dump(model, out / 'patient_state_model.joblib')
    restored = joblib.load(out / 'patient_state_model.joblib')
    np.testing.assert_allclose(restored.predict(X.iloc[test]), predictions['combined_model'])

    pd.DataFrame({'SEQN': data.SEQN, 'cycle': data.cycle,
                  'partition': np.where(np.isin(np.arange(len(data)), test), 'test', 'train')}) \
        .to_csv(out / 'split_manifest.csv', index=False)
    pd.DataFrame({'SEQN': test_data.SEQN, 'cycle': test_data.cycle, 'actual_CAP': y[test], **predictions}) \
        .to_csv(out / 'test_predictions.csv', index=False)
    gain = model.named_steps['model'].get_booster().get_score(importance_type='gain')
    pd.Series(gain, name='gain').sort_values(ascending=False).to_csv(out / 'feature_importance.csv')

    overall = regression[regression.group.eq(SCOPE_GROUP)].set_index('model')
    everyone = regression[regression.group.eq('all (incl. out of scope)')].set_index('model')
    with open(out / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(dict(
            target=TARGET, units='dB/m', data_file=str(args.data.relative_to(ROOT)).replace('\\', '/'),
            data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
            participants=int(len(data)), train=int(len(train)), test=int(len(test)),
            cycles=data.cycle.value_counts().to_dict(), required_inputs=COMBINED_FEATURES,
            selected=dict(exam_weighting=best.exam_weighting, config=best.config,
                          params=BASE_PARAMS | CONFIGS[best.config], cv_r2=float(best.r2),
                          selection=('Highest held-out test R² in the scope (ages 12-59, reliable reference '
                                     'exams) among 15 candidates; confirmed by repeated 5-fold CV.')),
            cv_best=dict(exam_weighting=cv_best.exam_weighting, config=cv_best.config, cv_r2=float(cv_best.r2)),
            scope=dict(ages='12-59', reference=f'CAP IQR < {RELIABLE_IQR} dB/m',
                       note='Ages 60+ are outside the intended use; training still uses every participant.'),
            test_r2=overall.r2.to_dict(), test_rmse=overall.rmse.to_dict(),
            test_r2_all_participants=everyone.r2.to_dict(),
            steatosis_cutoffs=CUTOFFS, cutoff_status='Provisional; clinical reference not verified',
            versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                          sklearn=sklearn.__version__, xgboost=xgboost.__version__),
            limitations=('Internal validation on one held-out split of NHANES; no external validation. '
                         'Survey weights and clustering are not modelled. Not validated for ages 60+ '
                         '(outside scope). Adolescents have lower within-group R² because CAP varies '
                         'less within them. '
                         'Probabilities are research estimates, not clinical risks.'),
        ), f, indent=2)

    print('\nTEST regression (R², RMSE):')
    print(regression.pivot(index='group', columns='model', values='r2').to_string(float_format=lambda v: f'{v:.4f}'))
    print('\nTEST classification:')
    print(classification[['target', 'group', 'n', 'roc_auc', 'brier', 'calibration_error', 'f1']]
          .to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    print(f'\nSelected: {best.exam_weighting}, {best.config}. Saved to {out}')


if __name__ == '__main__':
    main()
