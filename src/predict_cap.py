"""Batch inference with the saved preprocessing, CAP regressor and steatosis classifiers."""
import argparse
from pathlib import Path
import sys

# Serialized feature transformers live in the importable src package.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = ROOT / 'data/results/combined/patient_state_model.joblib'
# The earlier 14-input model trained on 2017-2020 only; still loadable with --model.
LEGACY_MODEL_FILE = ROOT / 'data/results/final/patient_state_model.joblib'


def _validated_inputs(patients, model):
    required = list(model.feature_names_in_)
    missing = sorted(set(required) - set(patients.columns))
    if missing:
        raise ValueError(f'Missing required patient columns: {missing}')
    X = patients.loc[:, required].apply(pd.to_numeric, errors='raise')
    if X.empty or np.isinf(X.to_numpy(dtype=float)).any():
        raise ValueError('Supply at least one patient with finite values or missing values (NaN).')
    for column, allowed in [('RIAGENDR', [1, 2]), ('RIDRETH3', [1, 2, 3, 4, 6, 7])]:
        if not X[column].dropna().isin(allowed).all():
            raise ValueError(f'Unsupported NHANES codes in {column}; expected {allowed}')
    return X


def predict_patients(patients, model_file=MODEL_FILE):
    model = joblib.load(model_file)
    X = _validated_inputs(patients, model)
    return pd.Series(model.predict(X), index=patients.index, name='Predicted_CAP')


def predict_steatosis(patients, classifier_file):
    """Calibrated probability that CAP reaches the classifier's cutoff."""
    classifier = joblib.load(classifier_file)
    X = _validated_inputs(patients, classifier)
    return pd.Series(classifier.predict_proba(X), index=patients.index,
                     name=f'P_CAP_ge_{classifier.cutoff:g}')


def steatosis_classifiers(model_file):
    """Steatosis classifier files saved next to a regressor, if any."""
    return sorted(Path(model_file).parent.glob('steatosis_*.joblib'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--model', type=Path, default=MODEL_FILE)
    parser.add_argument('--steatosis', action='store_true',
                        help='Add calibrated steatosis probabilities from classifiers next to --model.')
    args = parser.parse_args()
    patients = pd.read_csv(args.input)
    result = patients[['SEQN']].copy() if 'SEQN' in patients else pd.DataFrame(index=patients.index)
    result['Predicted_CAP'] = predict_patients(patients, args.model)
    if args.steatosis:
        classifiers = steatosis_classifiers(args.model)
        if not classifiers:
            parser.error(f'No steatosis_*.joblib classifiers next to {args.model}')
        for classifier_file in classifiers:
            probability = predict_steatosis(patients, classifier_file)
            result[probability.name] = probability
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f'Saved {len(result)} CAP predictions to {args.output}')


if __name__ == '__main__':
    main()
