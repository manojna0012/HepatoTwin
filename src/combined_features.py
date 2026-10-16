"""Input contract and feature transforms for the combined 2017-2023 CAP model.

Extends the 36-input CAPFeatures contract with NHANES measurements and
questionnaire items collected the same way in both FibroScan cycles
(2017-March 2020 and August 2021-August 2023). Inputs use raw NHANES coding:
"refused" and "don't know" answers are turned into missing values here, so
callers can pass NHANES files unchanged. Derived scores are research proxies,
not validated clinical indices.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from src.cap_features import CAPFeatures, ENGINEERED, RAW_FEATURES as CORE_FEATURES

BODY = ['BMXHIP', 'BMXARMC', 'BMXARML', 'BMXLEG']
DEMOGRAPHIC = ['INDFMPIR', 'DMDEDUC2', 'DMDBORN4']
FASTING = ['LBXIN', 'LBXGLU']
CBC = ['LBXWBCSI', 'LBXLYPCT', 'LBXNEPCT', 'LBXMOPCT', 'LBXHGB', 'LBXHCT',
       'LBXMCVSI', 'LBXRDW', 'LBXPLTSI', 'LBXMPSI']
INFLAMMATION = ['LBXHSCRP', 'LBXFER']
BLOOD_PRESSURE = ['BPXOSY1', 'BPXOSY2', 'BPXOSY3', 'BPXODI1', 'BPXODI2', 'BPXODI3', 'BPXOPLS1']
QUESTIONNAIRE = [
    'DIQ010', 'DIQ160', 'DIQ050', 'DIQ070',  # diabetes, prediabetes, insulin, diabetes pills
    'BPQ020', 'BPQ080',                      # told high blood pressure, high cholesterol
    'WHQ070', 'WHD050',                      # tried to lose weight, weight 1 year ago (lb)
    'HUQ010', 'SLD012', 'SMQ020', 'SMQ040',  # general health, sleep hours, smoking
    'ALQ111', 'ALQ121', 'ALQ130', 'ALQ151', 'PAD680',  # alcohol, sedentary minutes
]
EXTRA_FEATURES = BODY + DEMOGRAPHIC + FASTING + CBC + INFLAMMATION + BLOOD_PRESSURE + QUESTIONNAIRE
COMBINED_FEATURES = CORE_FEATURES + EXTRA_FEATURES
DERIVED = ['WHR', 'HOMA_IR', 'TyG', 'mean_SBP', 'mean_DBP', 'weight_change_1y']

# NHANES "refused" / "don't know" codes. Single-digit items use 7 and 9; items
# whose valid answers include 7 or 9 (counts, hours, frequency scales) use wider codes.
WIDE_MISSING_CODES = {
    'ALQ121': {77, 99}, 'ALQ130': {777, 999}, 'PAD680': {7777, 9999},
    'WHD050': {7777, 9999}, 'DMDBORN4': {77, 99}, 'SLD012': set(),
}
CODED_COLUMNS = QUESTIONNAIRE + ['DMDEDUC2', 'DMDBORN4']


def clean_missing_codes(frame):
    frame = frame.copy()
    for column in CODED_COLUMNS:
        codes = WIDE_MISSING_CODES.get(column, {7, 9})
        frame[column] = frame[column].where(~frame[column].isin(codes))
    return frame


class CombinedCAPFeatures(TransformerMixin, BaseEstimator):
    """Validate the combined raw-input contract and add derived features."""

    def fit(self, X, y=None):
        self.transform(X)
        self.feature_names_in_ = np.asarray(COMBINED_FEATURES, dtype=object)
        self.n_features_in_ = len(COMBINED_FEATURES)
        return self

    def transform(self, X):
        missing = sorted(set(COMBINED_FEATURES) - set(X.columns))
        if missing:
            raise ValueError(f'Missing required patient columns: {missing}')
        core = CAPFeatures().transform(X)
        extra = X.loc[:, EXTRA_FEATURES].apply(pd.to_numeric, errors='raise')
        if np.isinf(extra.to_numpy(dtype=float)).any():
            raise ValueError('Infinite raw inputs are not supported.')
        d = pd.concat([core, clean_missing_codes(extra)], axis=1)

        def positive(series):
            return series.where(series > 0)

        d['WHR'] = d.BMXWAIST / positive(d.BMXHIP)
        d['HOMA_IR'] = d.LBXIN * d.LBXGLU / 405
        # Fasting glucose when measured, otherwise the standard biochemistry glucose.
        glucose = d.LBXGLU.fillna(d.LBXSGL)
        d['TyG'] = np.log(positive(d.LBXSTR) * positive(glucose) / 2)
        d['mean_SBP'] = d[['BPXOSY1', 'BPXOSY2', 'BPXOSY3']].mean(axis=1)
        d['mean_DBP'] = d[['BPXODI1', 'BPXODI2', 'BPXODI3']].mean(axis=1)
        # Self-reported weight is in pounds; BMXWT is measured kilograms.
        d['weight_change_1y'] = d.BMXWT - d.WHD050 * 0.4536
        return d.replace([np.inf, -np.inf], np.nan)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'feature_names_in_')
        return np.asarray(CORE_FEATURES + ENGINEERED + EXTRA_FEATURES + DERIVED, dtype=object)
