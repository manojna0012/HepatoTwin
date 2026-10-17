"""Additional validation of the combined 2017-2023 patient-state model.

Addresses the limitations listed in data/results/combined/metadata.json, using
the saved artifacts and the same 80/20 split as train_combined_model.py:

1. Uncertainty: 95% bootstrap intervals (participant resampling of the test set)
   for regression and classification metrics, overall and by age group.
2. Survey weights: test metrics weighted by NHANES examination weights
   (WTMECPRP for 2017-March 2020, WTMEC2YR for 2021-2023), pooled in
   proportion to cycle length (3.2 and 2 years) as CDC recommends.
3. Older adults: age-group-specific isotonic calibration, fitted on the same
   training out-of-fold predictions as the global calibrator, compared with
   the global calibration on the test set.
4. Reference-standard sensitivity: a third classifier for CAP >= 302 dB/m.
5. Temporal validation: train on every 2017-March 2020 participant with the
   selected settings and test on every 2021-2023 participant.

The test set is not used to choose anything. Survey strata and PSUs are not
used, so intervals reflect participant sampling, not the full survey design.
Run from the project root after train_combined_model.py:

    .venv/Scripts/python.exe -m src.validate_combined_model
"""
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.train_final_model import ROOT, TARGET
from src.combined_features import COMBINED_FEATURES
from src.build_combined_dataset import OUTPUT as DATA_FILE
from src.train_combined_model import (
    AGE_GROUPS, CUTOFFS, SEED, classifier_pipeline, exam_weights, fit_weighted,
    regressor_pipeline, train_classifier)

MODEL_DIR = ROOT / 'data' / 'results' / 'combined'
DEMO_FILES = {'2017-2020': (ROOT / 'data/raw/P_DEMO.xpt', 'WTMECPRP', 3.2),
              '2021-2023': (ROOT / 'data/raw_2021/DEMO_L.xpt', 'WTMEC2YR', 2.0)}
STRICT_CUTOFF = 302
BOOTSTRAPS = 1000


def survey_weights(data):
    """Pooled MEC weights: each cycle's weight scaled by its share of the 5.2 years."""
    total = sum(years for _, _, years in DEMO_FILES.values())
    parts = []
    for cycle, (path, column, years) in DEMO_FILES.items():
        demo = pd.read_sas(path)[['SEQN', column]].rename(columns={column: 'weight'})
        demo['weight'] *= years / total
        parts.append(demo.assign(cycle=cycle))
    weights = data[['SEQN', 'cycle']].merge(pd.concat(parts), on=['SEQN', 'cycle'], how='left',
                                            validate='one_to_one').weight.to_numpy()
    if np.isnan(weights).any() or (weights <= 0).any():
        raise ValueError('Every participant needs a positive examination weight.')
    return weights


def r2(y, prediction, w):
    mean = np.average(y, weights=w)
    return 1 - np.sum(w * (y - prediction) ** 2) / np.sum(w * (y - mean) ** 2)


def rmse(y, prediction, w):
    return float(np.sqrt(np.average((y - prediction) ** 2, weights=w)))


def ece(label, probability, w, bins=10):
    edges = np.quantile(probability, np.linspace(0, 1, bins + 1))
    index = np.clip(np.searchsorted(edges, probability, side='right') - 1, 0, bins - 1)
    total = w.sum()
    return float(sum(abs(np.average(label[m], weights=w[m]) - np.average(probability[m], weights=w[m]))
                     * w[m].sum() / total for m in (index == b for b in range(bins)) if m.any()))


def regression_scores(y, prediction, w):
    return dict(r2=r2(y, prediction, w), rmse=rmse(y, prediction, w))


def classification_scores(label, probability, threshold, w):
    return dict(roc_auc=roc_auc_score(label, probability, sample_weight=w),
                brier=brier_score_loss(label, probability, sample_weight=w),
                calibration_error=ece(label, probability, w),
                f1=f1_score(label, probability >= threshold, sample_weight=w))


def with_intervals(score, arrays, w, groups, rng):
    """Point estimate and 95% percentile bootstrap interval per group."""
    rows = []
    n = len(w)
    samples = [rng.integers(0, n, n) for _ in range(BOOTSTRAPS)]
    for group, mask in groups.items():
        point = score(*(a[mask] for a in arrays), w[mask])
        draws = []
        for index in samples:
            index = index[mask[index]]
            try:
                draws.append(score(*(a[index] for a in arrays), w[index]))
            except ValueError:  # a resample with one class only
                continue
        draws = pd.DataFrame(draws)
        for metric, value in point.items():
            low, high = np.percentile(draws[metric], [2.5, 97.5])
            rows.append(dict(group=group, n=int(mask.sum()), metric=metric,
                             estimate=float(value), ci_low=float(low), ci_high=float(high)))
    return rows


def age_groups(age):
    groups = {'all': np.ones(len(age), bool)}
    return groups | {f'age {name}': (age >= lo) & (age < hi) for name, (lo, hi) in AGE_GROUPS.items()}


def age_index(age):
    edges = [lo for lo, _ in AGE_GROUPS.values()][1:]
    return np.searchsorted(edges, age, side='right')


def oof_probabilities(X, label):
    oof = np.zeros(len(label))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, label):
        oof[va] = classifier_pipeline().fit(X.iloc[tr], label[tr]).predict_proba(X.iloc[va])[:, 1]
    return oof


def isotonic(raw, label):
    return IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(raw, label)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=lambda p: ROOT / p, default=MODEL_DIR / 'validation')
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    data = pd.read_csv(DATA_FILE)
    X, y = data[COMBINED_FEATURES], data[TARGET].to_numpy(dtype=float)
    iqr = data.LUXCPIQR.to_numpy(dtype=float)
    weight = survey_weights(data)
    train, test = train_test_split(np.arange(len(data)), test_size=0.2, random_state=SEED,
                                   stratify=data.cycle)
    manifest = pd.read_csv(MODEL_DIR / 'split_manifest.csv')
    if not np.array_equal(manifest.partition.eq('test').to_numpy().nonzero()[0], np.sort(test)):
        raise ValueError('Split does not match the saved model; retrain first.')
    X_tr, y_tr, X_te, y_te = X.iloc[train].reset_index(drop=True), y[train], X.iloc[test], y[test]
    age_tr, age_te = data.RIDAGEYR.to_numpy()[train], data.RIDAGEYR.to_numpy()[test]
    w_te = weight[test]
    groups = age_groups(age_te)
    ones = np.ones(len(test))
    metadata = json.loads((MODEL_DIR / 'metadata.json').read_text(encoding='utf-8'))

    # 1-2. Saved regressor and classifiers: bootstrap intervals, unweighted and survey-weighted.
    regressor = joblib.load(MODEL_DIR / 'patient_state_model.joblib')
    prediction = regressor.predict(X_te)
    interval_rows = []
    for weighting, w in [('unweighted', ones), ('survey_weighted', w_te)]:
        for row in with_intervals(regression_scores, (y_te, prediction), w, groups, rng):
            interval_rows.append(dict(model='CAP regressor', weighting=weighting, **row))
    probabilities = {}
    for name, cutoff in CUTOFFS.items():
        classifier = joblib.load(MODEL_DIR / f'steatosis_{name}.joblib')
        probabilities[name] = classifier.predict_proba(X_te)
        label = (y_te >= cutoff).astype(int)
        threshold = classifier.threshold
        score = lambda l, p, w, t=threshold: classification_scores(l, p, t, w)
        for weighting, w in [('unweighted', ones), ('survey_weighted', w_te)]:
            for row in with_intervals(score, (label, probabilities[name]), w, groups, rng):
                interval_rows.append(dict(model=f'P(CAP >= {cutoff})', weighting=weighting, **row))
        print(f'Intervals done for CAP >= {cutoff}', flush=True)

    # 3-4. Age-specific calibration for every cutoff, plus the stricter CAP >= 302 classifier.
    calibration_rows, strict = [], None
    for name, cutoff in (CUTOFFS | {f'cap{STRICT_CUTOFF}': STRICT_CUTOFF}).items():
        label_tr, label_te = (y_tr >= cutoff).astype(int), (y_te >= cutoff).astype(int)
        oof = oof_probabilities(X_tr, label_tr)
        if cutoff == STRICT_CUTOFF:
            strict = train_classifier(X_tr, y_tr, cutoff)
            pipeline, threshold = strict.pipeline, strict.threshold
            joblib.dump(strict, out / f'steatosis_{name}.joblib')
        else:
            saved = joblib.load(MODEL_DIR / f'steatosis_{name}.joblib')
            pipeline, threshold = saved.pipeline, saved.threshold
        raw_te = pipeline.predict_proba(X_te)[:, 1]
        global_cal = isotonic(oof, label_tr)
        by_age_tr, by_age_te = age_index(age_tr), age_index(age_te)
        age_cal = {g: isotonic(oof[by_age_tr == g], label_tr[by_age_tr == g]) for g in np.unique(by_age_tr)}
        p_global = global_cal.predict(raw_te)
        p_age = np.empty(len(raw_te))
        for g, calibrator in age_cal.items():
            p_age[by_age_te == g] = calibrator.predict(raw_te[by_age_te == g])
        if cutoff == STRICT_CUTOFF:
            probabilities[name] = p_global
            score = lambda l, p, w, t=threshold: classification_scores(l, p, t, w)
            for weighting, w in [('unweighted', ones), ('survey_weighted', w_te)]:
                for row in with_intervals(score, (label_te, p_global), w, groups, rng):
                    interval_rows.append(dict(model=f'P(CAP >= {cutoff})', weighting=weighting, **row))
        for method, p in [('global isotonic', p_global), ('age-group isotonic', p_age)]:
            for group, mask in groups.items():
                calibration_rows.append(dict(
                    target=name, calibration=method, group=group, n=int(mask.sum()),
                    calibration_error=ece(label_te[mask], p[mask], ones[mask]),
                    brier=brier_score_loss(label_te[mask], p[mask]),
                    roc_auc=roc_auc_score(label_te[mask], p[mask])))
        print(f'Calibration comparison done for CAP >= {cutoff}', flush=True)

    # 5. Temporal validation: fit on 2017-2020, test on 2021-2023.
    early, late = data.cycle.eq('2017-2020').to_numpy(), data.cycle.eq('2021-2023').to_numpy()
    X_early, y_early = X[early].reset_index(drop=True), y[early]
    X_late, y_late, age_late = X[late], y[late], data.RIDAGEYR.to_numpy()[late]
    selected = metadata['selected']
    temporal_regressor = fit_weighted(regressor_pipeline(selected['config']), X_early, y_early,
                                      exam_weights(iqr[early], selected['exam_weighting']))
    late_groups, late_ones = age_groups(age_late), np.ones(late.sum())
    temporal_rows = []
    for row in with_intervals(regression_scores, (y_late, temporal_regressor.predict(X_late)),
                              late_ones, late_groups, rng):
        temporal_rows.append(dict(model='CAP regressor', **row))
    for name, cutoff in CUTOFFS.items():
        classifier = train_classifier(X_early, y_early, cutoff)
        label = (y_late >= cutoff).astype(int)
        score = lambda l, p, w, t=classifier.threshold: classification_scores(l, p, t, w)
        for row in with_intervals(score, (label, classifier.predict_proba(X_late)), late_ones,
                                  late_groups, rng):
            temporal_rows.append(dict(model=f'P(CAP >= {cutoff})', **row))
    print('Temporal validation done', flush=True)

    intervals = pd.DataFrame(interval_rows)
    calibration = pd.DataFrame(calibration_rows)
    temporal = pd.DataFrame(temporal_rows)
    intervals.to_csv(out / 'test_metrics_with_intervals.csv', index=False)
    calibration.to_csv(out / 'age_calibration_comparison.csv', index=False)
    temporal.to_csv(out / 'temporal_validation.csv', index=False)
    pd.DataFrame({'SEQN': data.SEQN.iloc[test].to_numpy(), 'survey_weight': w_te,
                  **{f'p_{k}': v for k, v in probabilities.items()}}) \
        .to_csv(out / 'test_probabilities.csv', index=False)
    with open(out / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(dict(
            bootstraps=BOOTSTRAPS, seed=SEED, interval='95% percentile, participant resampling of the test set',
            survey_weights={c: dict(column=col, years=yrs) for c, (_, col, yrs) in DEMO_FILES.items()},
            strict_cutoff=STRICT_CUTOFF, strict_threshold=strict.threshold,
            temporal=dict(train='2017-2020', train_n=int(early.sum()), test='2021-2023', test_n=int(late.sum()),
                          settings=selected),
            limitations=('Survey strata and PSUs are not used, so intervals do not reflect the full design. '
                         'Temporal validation uses the same survey programme, not an external population.'),
        ), f, indent=2)

    show = lambda frame: print(frame.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    print('\nTest metrics with 95% intervals (all participants):')
    show(intervals.query("group == 'all'")[['model', 'weighting', 'metric', 'estimate', 'ci_low', 'ci_high']])
    print('\nCalibration error by age, global vs age-group isotonic:')
    show(calibration.pivot_table(index=['target', 'group'], columns='calibration',
                                 values='calibration_error').reset_index())
    print('\nTemporal validation (train 2017-2020, test 2021-2023):')
    show(temporal.query("group == 'all'")[['model', 'metric', 'estimate', 'ci_low', 'ci_high']])
    print(f'\nSaved to {out}')


if __name__ == '__main__':
    main()
