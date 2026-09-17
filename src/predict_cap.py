"""Batch inference with the saved preprocessing and CAP regressor."""
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
MODEL_FILE = ROOT / 'data/results/final/patient_state_model.joblib'


def predict_patients(patients, model_file=MODEL_FILE):
    model = joblib.load(model_file)
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
    return pd.Series(model.predict(X), index=patients.index, name='Predicted_CAP')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--model', type=Path, default=MODEL_FILE)
    args = parser.parse_args()
    patients = pd.read_csv(args.input)
    result = patients[['SEQN']].copy() if 'SEQN' in patients else pd.DataFrame(index=patients.index)
    result['Predicted_CAP'] = predict_patients(patients, args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f'Saved {len(result)} CAP predictions to {args.output}')


if __name__ == '__main__':
    main()
