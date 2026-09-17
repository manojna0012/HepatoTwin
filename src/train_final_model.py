"""Reproducible CAP model selection using training-only cross-validation."""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
NUMERIC = ['RIDAGEYR', 'BMXBMI', 'BMXWAIST', 'LBXSATSI', 'LBXSASSI',
           'LBXGH', 'LBXTC', 'LBDHDD', 'DR1TKCAL', 'DR1TPROT', 'DR1TCARB', 'DR1TTFAT']
CATEGORICAL = ['RIAGENDR', 'RIDRETH3']
FEATURES = NUMERIC + CATEGORICAL
TARGET = 'LUXCAPM'
BASELINE_PARAMS = dict(n_estimators=300, learning_rate=0.05, max_depth=6,
                       subsample=0.8, colsample_bytree=0.8)


def load_cohort(path):
    data = pd.read_csv(path)
    required = FEATURES + [TARGET, 'SEQN', 'LUAXSTAT']
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    data = data.loc[data.LUAXSTAT.eq(1) & data[TARGET].notna(), required].copy()
    if data.empty or data.SEQN.isna().any() or data.SEQN.duplicated().any():
        raise ValueError('Cohort must contain unique, nonmissing participant IDs.')
    if np.isinf(data[FEATURES + [TARGET]].to_numpy(dtype=float)).any():
        raise ValueError('Infinite feature or target values are not supported.')
    return data.reset_index(drop=True)


def make_pipeline(params=None, numeric=None, estimator=None, add_indicator=False):
    numeric = NUMERIC if numeric is None else numeric
    preprocessing = ColumnTransformer([
        ('numeric', SimpleImputer(strategy='median', keep_empty_features=True,
                                  add_indicator=add_indicator), numeric),
        ('categorical', Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent', keep_empty_features=True)),
            ('encode', OneHotEncoder(categories=[[1., 2.], [1., 2., 3., 4., 6., 7.]],
                                     drop='first', handle_unknown='error', sparse_output=False)),
        ]), CATEGORICAL),
    ])
    return Pipeline([('preprocess', preprocessing), ('model', estimator if estimator is not None else XGBRegressor(
        objective='reg:squarederror', random_state=42, n_jobs=2,
        **(params or BASELINE_PARAMS)))])


def metrics(y, prediction):
    return dict(MAE=float(mean_absolute_error(y, prediction)),
                RMSE=float(np.sqrt(mean_squared_error(y, prediction))),
                R2=float(r2_score(y, prediction)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/processed/nhanes_merged.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/results/final')
    parser.add_argument('--iterations', type=int, default=40)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error('--iterations must be positive')
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    data = load_cohort(args.data)
    train, test = train_test_split(np.arange(len(data)), test_size=0.2, random_state=42)
    X, y = data[FEATURES], data[TARGET]
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {'rmse': 'neg_root_mean_squared_error', 'mae': 'neg_mean_absolute_error', 'r2': 'r2'}
    baseline = make_pipeline()
    print(f'Cohort: {len(data)}; train: {len(train)}; test: {len(test)}', flush=True)
    baseline_cv = cross_validate(baseline, X.iloc[train], y.iloc[train], cv=cv, scoring=scoring)
    distributions = {
        'model__n_estimators': [200, 350, 500, 750, 1000],
        'model__learning_rate': [0.015, 0.025, 0.04, 0.06],
        'model__max_depth': [2, 3, 4, 5],
        'model__min_child_weight': [5, 10, 20, 40],
        'model__subsample': [0.7, 0.85, 1.0],
        'model__colsample_bytree': [0.7, 0.85, 1.0],
        'model__reg_alpha': [0, 0.1, 1, 5],
        'model__reg_lambda': [1, 5, 15, 40],
    }
    search = RandomizedSearchCV(make_pipeline(), distributions, n_iter=args.iterations,
        scoring=scoring, refit='rmse', cv=cv, random_state=42, n_jobs=2,
        error_score='raise', return_train_score=True, verbose=1)
    search.fit(X.iloc[train], y.iloc[train])
    pd.DataFrame(search.cv_results_).to_csv(out / 'cv_search.csv', index=False)
    baseline_score = float(-baseline_cv['test_rmse'].mean())
    tuned_score = float(-search.best_score_)
    # Selection is frozen before computing any holdout predictions.
    selected_name = 'Tuned XGBoost' if tuned_score < baseline_score else 'Baseline XGBoost'
    baseline.fit(X.iloc[train], y.iloc[train])
    selected = search.best_estimator_ if selected_name == 'Tuned XGBoost' else baseline
    dummy = DummyRegressor().fit(X.iloc[train], y.iloc[train])
    models = {'Mean predictor': dummy, 'Baseline XGBoost': baseline,
              'Tuned XGBoost': search.best_estimator_}
    predictions = pd.DataFrame({'SEQN': data.iloc[test].SEQN.to_numpy(), 'Actual_CAP': y.iloc[test].to_numpy()})
    rows = []
    for name, model in models.items():
        pred = model.predict(X.iloc[test])
        predictions[name] = pred
        rows.append({'Model': name, **metrics(y.iloc[test], pred)})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(out / 'model_comparison.csv', index=False)
    predictions.to_csv(out / 'test_predictions.csv', index=False)
    joblib.dump(selected, out / 'patient_state_model.joblib')
    # Keep the evaluated artifact trained on the development partition only.
    reloaded = joblib.load(out / 'patient_state_model.joblib')
    np.testing.assert_allclose(reloaded.predict(X.iloc[test]), selected.predict(X.iloc[test]))
    folds = np.full(len(data), -1)
    for fold, (_, validation) in enumerate(cv.split(train)):
        folds[train[validation]] = fold
    pd.DataFrame({'SEQN': data.SEQN, 'partition': np.where(folds < 0, 'test', 'train'),
                  'cv_validation_fold': folds}).to_csv(out / 'split_manifest.csv', index=False)
    names = selected.named_steps['preprocess'].get_feature_names_out()
    pd.DataFrame({'Feature': names, 'Importance': selected.named_steps['model'].feature_importances_})\
        .sort_values('Importance', ascending=False).to_csv(out / 'feature_importance.csv', index=False)
    subgroup_rows = []
    pred = selected.predict(X.iloc[test])
    test_data = data.iloc[test].reset_index(drop=True)
    for column in ['RIAGENDR', 'RIDRETH3']:
        for value, group in test_data.groupby(column):
            subgroup_rows.append({'Group': column, 'Value': value, 'N': len(group),
                **metrics(group[TARGET], pred[group.index])})
    pd.DataFrame(subgroup_rows).to_csv(out / 'subgroup_metrics.csv', index=False)
    # Paired bootstrap describes uncertainty of the fixed-model holdout difference.
    rng = np.random.default_rng(42)
    actual = predictions.Actual_CAP.to_numpy()
    base_error = (actual - predictions['Baseline XGBoost'].to_numpy()) ** 2
    final_error = (actual - pred) ** 2
    deltas = []
    for _ in range(2000):
        indices = rng.integers(0, len(test), size=len(test))
        deltas.append(float(np.sqrt(base_error[indices].mean()) - np.sqrt(final_error[indices].mean())))
    metadata = {
        'selected_model': selected_name, 'target': TARGET, 'units': 'dB/m',
        'features': FEATURES, 'selection_criterion': 'lowest mean five-fold training CV RMSE',
        'baseline_cv_rmse': baseline_score, 'tuned_cv_rmse': tuned_score,
        'baseline_cv_fold_rmse': (-baseline_cv['test_rmse']).tolist(),
        'best_parameters': search.best_params_, 'search_iterations': args.iterations,
        'seed': 42, 'n_train': len(train), 'n_test': len(test),
        'data_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
        'versions': {'python': platform.python_version(), 'sklearn': sklearn.__version__,
                     'xgboost': xgboost.__version__, 'numpy': np.__version__,
                     'pandas': pd.__version__, 'joblib': joblib.__version__},
        'holdout_metrics': rows,
        'rmse_improvement_bootstrap_95_percent_interval': np.quantile(deltas, [0.025, 0.975]).tolist(),
        'limitations': [
            'Original holdout was already used in earlier model development; this is not external validation.',
            'Unweighted participant split; NHANES survey design and household/PSU clustering are not modeled.',
            'Cross-sectional CAP prediction is not a diagnosis or a causal prediction of dietary response.',
            'Eligible complete exams include ages 12 through 80 (age may be top-coded).',
            'Bootstrap interval conditions on the fitted models and does not include tuning uncertainty.',
        ],
    }
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(comparison.to_string(index=False), flush=True)
    print(f'Selected: {selected_name}; baseline CV RMSE={baseline_score:.3f}; tuned={tuned_score:.3f}', flush=True)
    print(f'Saved model and evaluation to {out}', flush=True)


if __name__ == '__main__':
    main()
