"""Checks for the combined-model input contract, features and saved artifacts."""
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.combined_features import COMBINED_FEATURES, CombinedCAPFeatures
from src.predict_cap import predict_patients, predict_steatosis
from src.steatosis_model import CalibratedSteatosisClassifier
from src.train_combined_model import classifier_pipeline, exam_weights, regressor_pipeline


def patients(n=8):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({name: rng.uniform(1, 100, n) for name in COMBINED_FEATURES})
    X['RIAGENDR'] = [1., 2.] * (n // 2)
    X['RIDRETH3'] = [1., 2., 3., 4., 6., 7., 1., 2.][:n]
    return X


class CombinedModelTests(unittest.TestCase):
    def test_missing_codes_depend_on_question(self):
        X = patients(2)
        X[['SLD012', 'ALQ121', 'ALQ130', 'DIQ010', 'DMDBORN4']] = [[7., 7., 9., 9., 99.], [9., 99., 999., 1., 1.]]
        result = CombinedCAPFeatures().fit(X).transform(X)
        self.assertEqual(result.SLD012.tolist(), [7., 9.])      # sleep hours are valid answers
        self.assertEqual(result.ALQ121.iloc[0], 7.)              # frequency category 7 is valid
        self.assertTrue(pd.isna(result.ALQ121.iloc[1]))          # 99 = don't know
        self.assertEqual(result.ALQ130.iloc[0], 9.)              # 9 drinks is valid
        self.assertTrue(pd.isna(result.ALQ130.iloc[1]))
        self.assertTrue(pd.isna(result.DIQ010.iloc[0]))          # 9 = don't know
        self.assertTrue(pd.isna(result.DMDBORN4.iloc[0]))

    def test_derived_features(self):
        X = patients(2)
        X[['BMXWAIST', 'BMXHIP', 'LBXIN', 'LBXGLU', 'LBXSTR', 'LBXSGL', 'BMXWT', 'WHD050']] = [
            [90., 100., 10., 81., 150., 100., 80., 180.], [90., 0., 10., np.nan, 150., 100., 80., np.nan]]
        result = CombinedCAPFeatures().fit(X).transform(X)
        self.assertAlmostEqual(result.WHR.iloc[0], 0.9)
        self.assertTrue(pd.isna(result.WHR.iloc[1]))              # zero hip circumference
        self.assertAlmostEqual(result.HOMA_IR.iloc[0], 2.0)
        self.assertAlmostEqual(result.TyG.iloc[0], np.log(150 * 81 / 2))
        self.assertAlmostEqual(result.TyG.iloc[1], np.log(150 * 100 / 2))  # falls back to LBXSGL
        self.assertAlmostEqual(result.weight_change_1y.iloc[0], 80 - 180 * 0.4536)
        self.assertEqual(list(result.columns), list(CombinedCAPFeatures().fit(X).get_feature_names_out()))

    def test_missing_column_rejected(self):
        with self.assertRaisesRegex(ValueError, 'LBXIN'):
            CombinedCAPFeatures().fit(patients().drop(columns='LBXIN'))

    def test_exam_weights(self):
        iqr = np.array([20., 40., 80., 400., np.nan])
        np.testing.assert_allclose(exam_weights(iqr, 'quality_weighted'), [1., 1., 0.5, 0.25, 1.])
        np.testing.assert_allclose(exam_weights(iqr, 'reliable_only'), [1., 0., 0., 0., 0.])
        np.testing.assert_allclose(exam_weights(iqr, 'all_exams'), np.ones(5))

    def test_saved_models_roundtrip(self):
        X, y = patients(), np.linspace(200, 340, 8)
        model = regressor_pipeline('base').set_params(model__n_estimators=5).fit(X, y)
        label = (y >= 260).astype(int)
        pipeline = classifier_pipeline().set_params(model__n_estimators=5).fit(X, label)
        calibrator = IsotonicRegression(out_of_bounds='clip').fit(pipeline.predict_proba(X)[:, 1], label)
        classifier = CalibratedSteatosisClassifier(pipeline, calibrator, 260, 0.5)
        with tempfile.TemporaryDirectory() as temp:
            model_path, classifier_path = Path(temp) / 'model.joblib', Path(temp) / 'clf.joblib'
            joblib.dump(model, model_path)
            joblib.dump(classifier, classifier_path)
            shuffled = X[COMBINED_FEATURES[::-1]]
            np.testing.assert_allclose(predict_patients(shuffled, model_path), model.predict(X))
            probability = predict_steatosis(shuffled, classifier_path)
            np.testing.assert_allclose(probability, classifier.predict_proba(X))
            self.assertTrue(((probability >= 0) & (probability <= 1)).all())
            with self.assertRaisesRegex(ValueError, 'Missing required'):
                predict_patients(X.drop(columns='BMXHIP'), model_path)


if __name__ == '__main__':
    unittest.main()
