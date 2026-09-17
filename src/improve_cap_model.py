"""Second-pass experiments; rank by development CV before holdout evaluation.

Run as: python -m src.improve_cap_model
Outputs are separate from the original 14-input model.
"""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from threadpoolctl import threadpool_limits

from src.train_final_model import ROOT, NUMERIC, CATEGORICAL, TARGET, load_cohort, make_pipeline, metrics

# Explicit allowlist: no elastography results/quality variables, IDs, weights,
# duplicated unit conversions, or secondary glucose/triglyceride assays.
EXTRA = ['BMXHT', 'BMXWT', 'LBXSGTSI', 'LBXSUA', 'LBXTR']
LABS = ['LBXSAL', 'LBXSAPSI', 'LBXSCR', 'LBXSTB', 'LBXSTP',
        'LBXSGB', 'LBXSIR', 'LBXSBU', 'LBXSLDSI']


def summarize_uncertainty(out):
    predictions = pd.read_csv(out / 'test_predictions.csv')
    actual = predictions['Actual_CAP'].to_numpy()
    previous_error = (actual - predictions['Previous final'].to_numpy()) ** 2
    new_error = (actual - predictions['Second-pass winner'].to_numpy()) ** 2
    rng = np.random.default_rng(42)
    differences = []
    for _ in range(2000):
        sample = rng.integers(0, len(actual), size=len(actual))
        differences.append(float(np.sqrt(previous_error[sample].mean()) - np.sqrt(new_error[sample].mean())))
    result = {
        'rmse_improvement': float(np.sqrt(previous_error.mean()) - np.sqrt(new_error.mean())),
        'paired_bootstrap_95_percent_interval': np.quantile(differences, [0.025, 0.975]).tolist(),
        'resamples': 2000, 'seed': 42,
        'interpretation': 'Conditional on fitted models; ignores tuning uncertainty, repeated holdout use, and survey clustering.',
    }
    (out / 'uncertainty.json').write_text(json.dumps(result, indent=2))
    return result


def candidates(best_params):
    result = {}
    for label, numeric in [('original', NUMERIC), ('extended', NUMERIC + EXTRA),
                           ('expanded', NUMERIC + EXTRA + LABS)]:
        for depth, trees in [(3, 600), (4, 350), (4, 700)]:
            params = {**best_params, 'max_depth': depth, 'n_estimators': trees}
            result[f'{label}_xgb_d{depth}_n{trees}'] = make_pipeline(params, numeric=numeric)
        result[f'{label}_xgb_missing_flags'] = make_pipeline(best_params, numeric=numeric, add_indicator=True)
        for leaves in [7, 15]:
            result[f'{label}_hist_leaves{leaves}'] = make_pipeline(numeric=numeric,
                estimator=HistGradientBoostingRegressor(max_iter=250, learning_rate=0.04,
                    max_leaf_nodes=leaves, l2_regularization=20, min_samples_leaf=40,
                    early_stopping=False, random_state=42))
        for alpha in [10, 100]:
            pipeline = make_pipeline(numeric=numeric, estimator=Ridge(alpha=alpha))
            categorical = pipeline.named_steps['preprocess'].transformers[1]
            pipeline.set_params(preprocess=ColumnTransformer([
                ('numeric', Pipeline([
                    ('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
                    ('splines', SplineTransformer(n_knots=5, degree=3, knots='quantile', include_bias=False)),
                    ('scale', StandardScaler()),
                ]), numeric), categorical]))
            result[f'{label}_spline_ridge{alpha}'] = pipeline
    return result


def main():
    out = ROOT / 'data/results/second_pass'
    out.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'data/processed/nhanes_merged.csv'
    data = pd.read_csv(source)
    cohort = load_cohort(source)
    data = data.set_index('SEQN').loc[cohort.SEQN].reset_index()
    manifest = pd.read_csv(ROOT / 'data/results/final/split_manifest.csv')
    train, test = train_test_split(np.arange(len(data)), test_size=0.2, random_state=42)
    assert set(data.iloc[test].SEQN) == set(manifest.loc[manifest.partition.eq('test'), 'SEQN'])
    all_features = NUMERIC + EXTRA + LABS + CATEGORICAL
    X, y = data[all_features], data[TARGET]
    metadata = json.loads((ROOT / 'data/results/final/metadata.json').read_text())
    params = {k.removeprefix('model__'): v for k, v in metadata['best_parameters'].items()}
    models = candidates(params)
    # Every pipeline is fitted on its declared raw inputs, preserving inference contracts.
    features = {name: list(model.named_steps['preprocess'].transformers[0][2]) + CATEGORICAL
                for name, model in models.items()}
    cv = list(KFold(n_splits=5, shuffle=True, random_state=42).split(train))
    predictions, rows = {}, []
    for name, model in models.items():
        prediction = cross_val_predict(model, X.iloc[train][features[name]], y.iloc[train], cv=cv, n_jobs=1)
        predictions[name] = prediction
        row = {'Model': name, 'Inputs': len(features[name]), **metrics(y.iloc[train], prediction)}
        rows.append(row)
        print(f'{name}: OOF RMSE={row["RMSE"]:.3f}, R2={row["R2"]:.3f}', flush=True)
        pd.DataFrame(rows).sort_values('RMSE').to_csv(out / 'cv_comparison.csv', index=False)
    # Fixed equal weighting of the two strongest candidates, also selected only by CV.
    ranked = sorted(rows, key=lambda row: row['RMSE'])
    top = [row['Model'] for row in ranked[:2]]
    blend = np.mean([predictions[name] for name in top], axis=0)
    blend_metrics = metrics(y.iloc[train], blend)
    blend_name = 'equal_blend_top_two'
    if blend_metrics['RMSE'] < ranked[0]['RMSE']:
        selected_name = blend_name
        selected = VotingRegressor([(f'model{i}', clone(models[name])) for i, name in enumerate(top)])
        chosen_features = [f for f in all_features if any(f in features[name] for name in top)]
        predictions[blend_name] = blend
    else:
        selected_name = ranked[0]['Model']
        selected = clone(models[selected_name])
        chosen_features = features[selected_name]
    rows.append({'Model': blend_name, 'Inputs': len(set(sum([features[n] for n in top], []))), **blend_metrics})
    pd.DataFrame(rows).sort_values('RMSE').to_csv(out / 'cv_comparison.csv', index=False)
    # Freeze selection and schema before evaluation, with a separate original-input winner.
    original_name = min((r for r in rows if r['Model'].startswith('original_')), key=lambda r: r['RMSE'])['Model']
    original_model = clone(models[original_name])
    selection = {'selected_model': selected_name, 'features': chosen_features, 'blend_members': top,
        'original_input_winner': original_name, 'selection': 'minimum pooled out-of-fold training RMSE',
        'n_candidates': len(models), 'seed': 42, 'n_train': len(train), 'n_test': len(test),
        'data_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'versions': metadata['versions'],
        'limitations': metadata['limitations'] + [
            'Repeated development on the same cohort adds selection optimism; external validation remains necessary.',
            'Expanded models require additional measurements; they are not drop-in replacements for the 14-input model.',
            'LBXTR is a fasting subsample measurement with substantial missingness; use only appropriate reference-assay inputs.',
        ]}
    (out / 'selection.json').write_text(json.dumps(selection, indent=2))
    selected.fit(X.iloc[train][chosen_features], y.iloc[train])
    original_model.fit(X.iloc[train][features[original_name]], y.iloc[train])
    joblib.dump(selected, out / 'patient_state_model.joblib')
    joblib.dump(original_model, out / 'original_input_model.joblib')
    previous = joblib.load(ROOT / 'data/results/final/patient_state_model.joblib')
    comparison, test_predictions = [], pd.DataFrame({'SEQN': data.iloc[test].SEQN, 'Actual_CAP': y.iloc[test]})
    for name, model in [('Previous final', previous), ('Original-input winner', original_model),
                        ('Second-pass winner', selected)]:
        prediction = model.predict(X.iloc[test][list(model.feature_names_in_)])
        test_predictions[name] = prediction
        comparison.append({'Model': name, **metrics(y.iloc[test], prediction)})
    test_predictions.to_csv(out / 'test_predictions.csv', index=False)
    summarize_uncertainty(out)
    pd.DataFrame(comparison).to_csv(out / 'model_comparison.csv', index=False)
    pd.DataFrame({'SEQN': data.iloc[train].SEQN, 'Actual_CAP': y.iloc[train], **predictions}).to_csv(out / 'oof_predictions.csv', index=False)
    data.iloc[train][all_features].isna().mean().rename('Missing_fraction').to_csv(out / 'training_missingness.csv')
    manifest.to_csv(out / 'split_manifest.csv', index=False)
    restored = joblib.load(out / 'patient_state_model.joblib')
    np.testing.assert_allclose(restored.predict(X.iloc[test][chosen_features]), test_predictions['Second-pass winner'])
    print(f'Selected: {selected_name}', flush=True)
    print(pd.DataFrame(comparison).to_string(index=False), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
