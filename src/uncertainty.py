"""Patient-level uncertainty for Layer 2 (Proposal section 17.4).

CAP interval: conformal prediction. Five-fold out-of-fold predictions on the training partition,
made with the selected settings, give one absolute error per training patient inside the scope; the
interval is the prediction plus or minus their (1 - alpha) quantile with the finite-sample correction.
Because the final model is refitted on all training patients, this is the cross-conformal form of split
conformal prediction; its coverage is checked on the held-out test set, as the proposal requires.

Probability bands: the isotonic calibrators are refitted on 200 bootstrap resamples of the training
out-of-fold scores, and each calibrated probability gets the 5th-95th percentile band of those refits.
This covers the uncertainty of the calibration step, not of the underlying classifier.

    .venv/Scripts/python.exe -m src.uncertainty
"""
import json
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold, StratifiedKFold

from src.build_combined_dataset import OUTPUT as DATA_FILE
from src.combined_features import COMBINED_FEATURES
from src.train_combined_model import (CUTOFFS, SEED, SELECTED, TARGET, classifier_pipeline, exam_weights,
                                      fit_weighted, in_scope, regressor_pipeline)
from src.train_final_model import ROOT

MODEL_DIR = ROOT / 'data' / 'results' / 'combined'
OUTPUT = MODEL_DIR / 'uncertainty'
LEVELS = (0.8, 0.9, 0.95)
TARGET_COVERAGE = 0.9
BOOTSTRAPS = 200
GRID = np.round(np.linspace(0, 1, 401), 4)
CAP_RANGE = (100, 400)


def conformal_quantile(scores, coverage):
    scores = np.sort(np.asarray(scores, dtype=float))
    rank = int(np.ceil((len(scores) + 1) * coverage))
    return float(scores[min(rank, len(scores)) - 1])


def split(data):
    manifest = pd.read_csv(MODEL_DIR / 'split_manifest.csv').set_index('SEQN').partition
    part = data.SEQN.map(manifest)
    return data[part.eq('train')].reset_index(drop=True), data[part.eq('test')].reset_index(drop=True)


def fit(train):
    X, y = train[COMBINED_FEATURES], train[TARGET].to_numpy(dtype=float)
    iqr = train.LUXCPIQR.to_numpy(dtype=float)
    oof = np.full(len(y), np.nan)
    for tr, va in KFold(5, shuffle=True, random_state=SEED).split(X):
        model = fit_weighted(regressor_pipeline(SELECTED['config']), X.iloc[tr], y[tr],
                             exam_weights(iqr[tr], SELECTED['exam_weighting']))
        oof[va] = model.predict(X.iloc[va])
        print('regressor fold done', flush=True)
    scores = np.abs(y - oof)[in_scope(train)]
    conformal = {'n_calibration': int(len(scores)),
                 'half_width': {str(level): conformal_quantile(scores, level) for level in LEVELS}}
    bands = []
    rng = np.random.default_rng(SEED)
    for name, cutoff in CUTOFFS.items():
        label = (y >= cutoff).astype(int)
        raw = np.zeros(len(label))
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, label):
            raw[va] = classifier_pipeline().fit(X.iloc[tr], label[tr]).predict_proba(X.iloc[va])[:, 1]
        curves = []
        for _ in range(BOOTSTRAPS):
            idx = rng.integers(0, len(label), len(label))
            iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(raw[idx], label[idx])
            curves.append(iso.predict(GRID))
        curves = np.array(curves)
        bands.append(pd.DataFrame({'classifier': name, 'raw_score': GRID,
                                   'low': np.percentile(curves, 5, axis=0),
                                   'high': np.percentile(curves, 95, axis=0)}))
        print(f'{name} bootstrap bands done', flush=True)
    return conformal, pd.concat(bands, ignore_index=True)


@lru_cache(maxsize=None)
def load():
    conformal = json.loads((OUTPUT / 'conformal.json').read_text(encoding='utf-8'))
    bands = pd.read_csv(OUTPUT / 'probability_bands.csv')
    return conformal, {name: g.sort_values('raw_score') for name, g in bands.groupby('classifier')}


def cap_interval(predicted_cap, coverage=TARGET_COVERAGE):
    half = load()[0]['half_width'][str(coverage)]
    return (float(np.clip(predicted_cap - half, *CAP_RANGE)), float(np.clip(predicted_cap + half, *CAP_RANGE)))


def probability_band(classifier_name, raw_score):
    band = load()[1][classifier_name]
    return (float(np.interp(raw_score, band.raw_score, band.low)),
            float(np.interp(raw_score, band.raw_score, band.high)))


def main():
    import joblib
    OUTPUT.mkdir(parents=True, exist_ok=True)
    train, test = split(pd.read_csv(DATA_FILE))
    conformal, bands = fit(train)
    model = joblib.load(MODEL_DIR / 'patient_state_model.joblib')
    scope = in_scope(test)
    prediction = model.predict(test[COMBINED_FEATURES])
    error = np.abs(test[TARGET].to_numpy(dtype=float) - prediction)
    coverage = {}
    for level, half in conformal['half_width'].items():
        coverage[level] = {'scope': float((error[scope] <= half).mean()), 'all_test': float((error <= half).mean())}
        for name, (lo, hi) in {'12-17': (12, 18), '18-39': (18, 40), '40-59': (40, 60)}.items():
            mask = scope & test.RIDAGEYR.between(lo, hi - 1).to_numpy()
            coverage[level][f'scope ages {name}'] = float((error[mask] <= half).mean())
    conformal['test_coverage'] = coverage
    widths = {}
    for name in CUTOFFS:
        classifier = joblib.load(MODEL_DIR / f'steatosis_{name}.joblib')
        raw = classifier.pipeline.predict_proba(test[COMBINED_FEATURES])[:, 1][scope]
        b = bands[bands.classifier.eq(name)]
        widths[name] = float(np.mean(np.interp(raw, b.raw_score, b.high) - np.interp(raw, b.raw_score, b.low)))
    conformal['mean_probability_band_width_scope'] = widths
    (OUTPUT / 'conformal.json').write_text(json.dumps(conformal, indent=2), encoding='utf-8')
    bands.to_csv(OUTPUT / 'probability_bands.csv', index=False)
    print(json.dumps(conformal, indent=2))


if __name__ == '__main__':
    main()
