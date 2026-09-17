"""Checks for preprocessing isolation and the saved inference contract."""
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.train_final_model import FEATURES, NUMERIC, make_pipeline, load_cohort
from src.predict_cap import predict_patients
from src.cap_features import CAPFeatures, RAW_FEATURES
from src.train_engineered_model import build_model
from src.demo import load_patient


class CAPModelTests(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame({name: [1., 2., np.nan, 4., 5., 6.] for name in NUMERIC})
        self.X['RIAGENDR'] = [1., 2., 1., 2., 1., 2.]
        self.X['RIDRETH3'] = [1., 2., 3., 4., 6., 7.]
        self.y = np.array([210., 220., 230., 240., 250., 260.])

    def test_training_medians_and_roundtrip(self):
        model = make_pipeline({'n_estimators': 5, 'max_depth': 2}).fit(self.X, self.y)
        numeric = model.named_steps['preprocess'].named_transformers_['numeric']
        np.testing.assert_allclose(numeric.statistics_, np.full(len(NUMERIC), 4.))
        patient = self.X.iloc[[0]].copy()
        patient['BMXBMI'] = np.nan
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'model.joblib'
            joblib.dump(model, path)
            expected = model.predict(patient)
            result = predict_patients(patient[FEATURES[::-1]], path)
            np.testing.assert_allclose(expected, result)
            np.testing.assert_allclose(numeric.statistics_, np.full(len(NUMERIC), 4.))
            with self.assertRaisesRegex(ValueError, 'Missing required'):
                predict_patients(patient.drop(columns=['BMXBMI']), path)
            patient['RIAGENDR'] = 9
            with self.assertRaisesRegex(ValueError, 'Unsupported'):
                predict_patients(patient, path)

    def test_duplicate_participants_rejected(self):
        cohort = self.X.copy()
        cohort['SEQN'] = [1, 1, 2, 3, 4, 5]
        cohort['LUAXSTAT'] = 1
        cohort['LUXCAPM'] = self.y
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'data.csv'
            cohort.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'unique'):
                load_cohort(path)

    def test_expanded_schema_requires_additional_measurement(self):
        X = self.X.assign(LBXSGTSI=[10., 20., 30., np.nan, 50., 60.])
        model = make_pipeline({'n_estimators': 5, 'max_depth': 2},
                              numeric=NUMERIC + ['LBXSGTSI'], add_indicator=True).fit(X, self.y)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'expanded.joblib'
            joblib.dump(model, path)
            with self.assertRaisesRegex(ValueError, 'LBXSGTSI'):
                predict_patients(self.X, path)
            patient = X.iloc[[0]].copy()
            patient['LBXSGTSI'] = np.nan
            np.testing.assert_allclose(predict_patients(patient, path), model.predict(patient))

    def test_engineered_ratios_and_zero_denominators(self):
        X = pd.DataFrame({name: [10., 20.] for name in RAW_FEATURES})
        X['RIAGENDR'] = [1, 2]
        X['RIDRETH3'] = [1, 2]
        X['BMXWAIST'] = 100.
        X['BMXHT'] = [200., 0.]
        X['DR1TKCAL'] = [2000., 0.]
        X['DR1TPROT'] = 100.
        transformer = CAPFeatures().fit(X)
        result = transformer.transform(X)
        self.assertAlmostEqual(result.WHtR.iloc[0], 0.5)
        self.assertAlmostEqual(result.energy_fraction_protein.iloc[0], 0.2)
        self.assertTrue(pd.isna(result.WHtR.iloc[1]))
        self.assertTrue(pd.isna(result.energy_fraction_protein.iloc[1]))
        self.assertFalse(np.isinf(result.to_numpy()).any())
        pd.testing.assert_frame_equal(result.iloc[[0]], transformer.transform(X.iloc[[0]]))
        model = build_model({'n_estimators': 5, 'max_depth': 2}).fit(X, [200, 300])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'engineered.joblib'
            joblib.dump(model, path)
            np.testing.assert_allclose(predict_patients(X, path), model.predict(X))
            with self.assertRaisesRegex(ValueError, 'BMXHT'):
                predict_patients(X.drop(columns='BMXHT'), path)

    def test_demo_rejects_training_participants(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pd.DataFrame({'SEQN': [1, 2], 'partition': ['train', 'test']}).to_csv(root / 'split_manifest.csv', index=False)
            pd.DataFrame({'SEQN': [1, 2], 'LUAXSTAT': [1, 1], 'LUXCAPM': [200, 300]}).to_csv(root / 'data.csv', index=False)
            patient = load_patient(root / 'model.joblib', data_file=root / 'data.csv')
            self.assertEqual(patient.SEQN, 2)
            with self.assertRaisesRegex(ValueError, 'held-out'):
                load_patient(root / 'model.joblib', seqn=1, data_file=root / 'data.csv')


if __name__ == '__main__':
    unittest.main()
