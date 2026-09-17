"""Evaluate supplied-model ideas with a fixed original holdout and CV ablations.

Run from the project root: python -m src.train_engineered_model
"""
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from src.cap_features import CAPFeatures, RAW_FEATURES, RAW_NUMERIC, ENGINEERED
from src.train_final_model import ROOT, TARGET, load_cohort, make_pipeline, metrics
from src.improve_cap_model import summarize_uncertainty

SUPPLIED_PARAMS = dict(n_estimators=1200, learning_rate=0.02, max_depth=4,
    min_child_weight=8, subsample=0.8, colsample_bytree=0.6, reg_lambda=3., reg_alpha=0.5)


def build_model(params, engineered=True, indicators=False):
    pipeline = make_pipeline(params, numeric=RAW_NUMERIC + (ENGINEERED if engineered else []),
                             add_indicator=indicators)
    if engineered:
        return Pipeline([('features', CAPFeatures()), ('regressor', pipeline)])
    return pipeline


def main():
    out = ROOT / 'data/results/engineered'
    out.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'data/processed/nhanes_merged.csv'
    data = pd.read_csv(source).set_index('SEQN')
    cohort = load_cohort(source)
    data = data.loc[cohort.SEQN].reset_index()
    previous_dir = ROOT / 'data/results/second_pass'
    # Preserve exact ordered training/test records so OOF comparisons are paired.
    old_oof = pd.read_csv(previous_dir / 'oof_predictions.csv')
    old_test = pd.read_csv(previous_dir / 'test_predictions.csv')
    indexed = data.set_index('SEQN')
    train = indexed.loc[old_oof.SEQN]
    test = indexed.loc[old_test.SEQN]
    assert set(train.index).isdisjoint(test.index) and len(train) + len(test) == len(data)
    X, y = train[RAW_FEATURES], train[TARGET]
    np.testing.assert_allclose(y, old_oof.Actual_CAP)
    cv = list(KFold(n_splits=5, shuffle=True, random_state=42).split(X))
    candidates = {
        'supplied_parameters_raw_labs': build_model(SUPPLIED_PARAMS, engineered=False),
        'supplied_parameters_with_ratios': build_model(SUPPLIED_PARAMS),
        'ratios_600_trees': build_model({**SUPPLIED_PARAMS, 'n_estimators': 600}),
        'ratios_800_trees_regularized': build_model({**SUPPLIED_PARAMS, 'n_estimators': 800,
                                                  'min_child_weight': 15, 'reg_lambda': 10}),
        'ratios_600_trees_missing_flags': build_model({**SUPPLIED_PARAMS, 'n_estimators': 600}, indicators=True),
    }
    previous = joblib.load(previous_dir / 'patient_state_model.joblib')
    previous_selection = json.loads((previous_dir / 'selection.json').read_text())
    previous_name = 'Previous expanded model'
    oof = {previous_name: old_oof[previous_selection['selected_model']].to_numpy()}
    rows = [{'Model': previous_name, **metrics(y, oof[previous_name])}]
    for name, candidate in candidates.items():
        prediction = cross_val_predict(candidate, X, y, cv=cv, n_jobs=1)
        oof[name] = prediction
        rows.append({'Model': name, **metrics(y, prediction)})
        pd.DataFrame(rows).sort_values('RMSE').to_csv(out / 'cv_comparison.csv', index=False)
        print(f'{name}: OOF RMSE={rows[-1]["RMSE"]:.3f}; R2={rows[-1]["R2"]:.3f}', flush=True)
    chosen = min(rows, key=lambda row: row['RMSE'])['Model']
    model = clone(previous if chosen == previous_name else candidates[chosen])
    input_features = list(previous.feature_names_in_) if chosen == previous_name else RAW_FEATURES
    selection = {
        'selected_model': chosen, 'features': input_features, 'target': TARGET, 'units': 'dB/m',
        'selection_criterion': 'lowest pooled five-fold training OOF RMSE; original fixed holdout',
        'n_train': len(train), 'n_test': len(test), 'seed': 42,
        'supplied_parameters': SUPPLIED_PARAMS,
        'data_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'versions': previous_selection['versions'],
        'limitations': previous_selection['limitations'] + [
            'LBXSTR and LBXSGL are assay-specific biochemistry inputs; they are not interchangeable with reference assays.',
            'HSI_partial omits diabetes; FLI_proxy uses non-reference triglycerides. Neither is a validated clinical score here.',
            'Reported CAP regression performance does not validate steatosis grades or dietary intervention effects.',
        ],
    }
    (out / 'selection.json').write_text(json.dumps(selection, indent=2))
    model.fit(train[input_features], y)
    joblib.dump(model, out / 'patient_state_model.joblib')
    # Also fit the supplied settings for a direct comparison on our original split.
    supplied = clone(candidates['supplied_parameters_with_ratios']).fit(X, y)
    first = joblib.load(ROOT / 'data/results/final/patient_state_model.joblib')
    predictions = pd.DataFrame({'SEQN': test.index, 'Actual_CAP': test[TARGET].to_numpy()})
    comparison = []
    for name, estimator in [('First-pass final', first), ('Previous final', previous),
                            ('Supplied settings, original split', supplied), ('Second-pass winner', model)]:
        prediction = estimator.predict(test[list(estimator.feature_names_in_)])
        predictions[name] = prediction
        comparison.append({'Model': name, **metrics(test[TARGET], prediction)})
    predictions.to_csv(out / 'test_predictions.csv', index=False)
    pd.DataFrame(comparison).to_csv(out / 'model_comparison.csv', index=False)
    pd.DataFrame({'SEQN': train.index, 'Actual_CAP': y.to_numpy(), **oof}).to_csv(out / 'oof_predictions.csv', index=False)
    pd.read_csv(previous_dir / 'split_manifest.csv').to_csv(out / 'split_manifest.csv', index=False)
    train[input_features].isna().mean().rename('Missing_fraction').to_csv(out / 'training_missingness.csv')
    summary = summarize_uncertainty(out)
    pred = predictions['Second-pass winner'].to_numpy()
    diagnostic = pd.DataFrame({'Actual_CAP': test[TARGET].to_numpy(), 'Prediction': pred})
    diagnostic['Absolute_error'] = abs(diagnostic.Actual_CAP - diagnostic.Prediction)
    diagnostic['CAP_decile'] = pd.qcut(diagnostic.Actual_CAP, 10, duplicates='drop')
    diagnostic.groupby('CAP_decile', observed=True).agg(N=('Actual_CAP', 'size'), MAE=('Absolute_error', 'mean'))\
        .to_csv(out / 'errors_by_cap_decile.csv')
    subgroup = []
    for field in ['RIAGENDR', 'RIDRETH3']:
        for value in sorted(test[field].dropna().unique()):
            mask = test[field].eq(value).to_numpy()
            subgroup.append({'Group': field, 'Value': value, 'N': int(mask.sum()), **metrics(test[TARGET].to_numpy()[mask], pred[mask])})
    pd.DataFrame(subgroup).to_csv(out / 'subgroup_metrics.csv', index=False)
    restored = joblib.load(out / 'patient_state_model.joblib')
    np.testing.assert_allclose(restored.predict(test[input_features]), pred)
    print(pd.DataFrame(comparison).to_string(index=False), flush=True)
    print(f'Selected {chosen}; uncertainty: {summary}', flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
