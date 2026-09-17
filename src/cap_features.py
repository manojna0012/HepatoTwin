"""Deterministic patient feature transforms adapted from the supplied model.

Index-like features are research proxies, not validated clinical scores.
"""
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from src.train_final_model import NUMERIC, CATEGORICAL

EXTRA_NUMERIC = [
    'BMXWT', 'BMXHT', 'LBXSGTSI', 'LBXSTR', 'LBXSGL', 'LBXSUA', 'LBXSAL',
    'LBXSTP', 'LBXSGB', 'LBXSAPSI', 'LBXSTB', 'LBXSCR', 'LBXSBU', 'LBXSCA',
    'LBXSPH', 'LBXSKSI', 'LBXSNASI', 'LBXSCLSI', 'LBXSC3SI', 'LBXSIR', 'LBXSLDSI', 'LBXSCK',
]
RAW_NUMERIC = NUMERIC + EXTRA_NUMERIC
RAW_FEATURES = RAW_NUMERIC + CATEGORICAL
ENGINEERED = ['WHtR', 'AST_ALT', 'TG_HDL', 'TC_HDL', 'HSI_partial', 'FLI_proxy',
              'energy_fraction_protein', 'energy_fraction_carb', 'energy_fraction_fat', 'kcal_per_kg']


class CAPFeatures(TransformerMixin, BaseEstimator):
    """Keep a stable raw-input contract and turn undefined ratios into missing values."""
    def fit(self, X, y=None):
        self.transform(X)
        self.feature_names_in_ = np.asarray(RAW_FEATURES, dtype=object)
        self.n_features_in_ = len(RAW_FEATURES)
        return self

    def transform(self, X):
        missing = sorted(set(RAW_FEATURES) - set(X.columns))
        if missing:
            raise ValueError(f'Missing required patient columns: {missing}')
        d = X.loc[:, RAW_FEATURES].apply(pd.to_numeric, errors='raise').copy()
        if np.isinf(d.to_numpy(dtype=float)).any():
            raise ValueError('Infinite raw inputs are not supported.')

        def ratio(numerator, denominator):
            return numerator / denominator.where(denominator > 0)

        d['WHtR'] = ratio(d.BMXWAIST, d.BMXHT)
        d['AST_ALT'] = ratio(d.LBXSASSI, d.LBXSATSI)
        d['TG_HDL'] = ratio(d.LBXSTR, d.LBDHDD)
        d['TC_HDL'] = ratio(d.LBXTC, d.LBDHDD)
        # Diabetes is not available in this input contract; do not call this full HSI.
        female = d.RIAGENDR.eq(2).astype(float).where(d.RIAGENDR.notna())
        d['HSI_partial'] = 8 * ratio(d.LBXSATSI, d.LBXSASSI) + d.BMXBMI + 2 * female
        # Uses the biochemistry triglyceride assay, not a reference fasting assay.
        z = (0.953 * np.log(d.LBXSTR.where(d.LBXSTR > 0)) + 0.139 * d.BMXBMI
             + 0.718 * np.log(d.LBXSGTSI.where(d.LBXSGTSI > 0)) + 0.053 * d.BMXWAIST - 15.745)
        d['FLI_proxy'] = 100 * expit(z)
        for nutrient, factor, name in [('DR1TPROT', 4, 'protein'), ('DR1TCARB', 4, 'carb'), ('DR1TTFAT', 9, 'fat')]:
            d[f'energy_fraction_{name}'] = ratio(factor * d[nutrient], d.DR1TKCAL)
        d['kcal_per_kg'] = ratio(d.DR1TKCAL.where(d.DR1TKCAL > 0), d.BMXWT)
        return d.replace([np.inf, -np.inf], np.nan)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'feature_names_in_')
        return np.asarray(RAW_FEATURES + ENGINEERED, dtype=object)
