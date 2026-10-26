"""Month 0 of the virtual cohort: the real NHANES test patients inside the model's scope.

The test partition comes from the split manifest saved when the Layer 2 model was trained, and
the scope rule (ages 12-59, reference CAP with IQR < 40 dB/m) matches
src.train_combined_model.in_scope. The rule is restated here rather than imported so that the
simulator does not load the training module; test_simulation.py checks the two agree.
"""
import pandas as pd

from src.simulation.parameters import ROOT

DATA_FILE = ROOT / 'data' / 'processed' / 'nhanes_combined.csv'
MANIFEST_FILE = ROOT / 'data' / 'results' / 'combined' / 'split_manifest.csv'
SCOPE_MAX_AGE = 60
RELIABLE_IQR = 40


def in_scope(data):
    return ((data.RIDAGEYR < SCOPE_MAX_AGE) & (data.LUXCPIQR < RELIABLE_IQR)).to_numpy()


def month_zero(data_file=DATA_FILE, manifest_file=MANIFEST_FILE):
    """In-scope test patients with their measured values, one row per patient."""
    data = pd.read_csv(data_file)
    manifest = pd.read_csv(manifest_file)
    test_ids = set(manifest.loc[manifest.partition.eq('test'), 'SEQN'])
    test = data[data.SEQN.isin(test_ids)].reset_index(drop=True)
    if len(test) != len(test_ids):
        raise ValueError('Split manifest and dataset disagree on the test participants.')
    return test[in_scope(test)].reset_index(drop=True)
