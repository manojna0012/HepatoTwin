"""Calibrated steatosis probability model saved by train_combined_model.py."""
import numpy as np


class CalibratedSteatosisClassifier:
    """Fitted feature/classifier pipeline plus an isotonic probability calibrator.

    Probabilities estimate P(CAP >= cutoff) in the training population. They are
    research estimates, not validated clinical risks.
    """

    def __init__(self, pipeline, calibrator, cutoff, threshold):
        self.pipeline = pipeline
        self.calibrator = calibrator
        self.cutoff = cutoff
        self.threshold = threshold

    @property
    def feature_names_in_(self):
        return self.pipeline.feature_names_in_

    def predict_proba(self, X):
        """Return calibrated P(CAP >= cutoff) as a 1-D array."""
        return np.clip(self.calibrator.predict(self.pipeline.predict_proba(X)[:, 1]), 0, 1)

    def predict(self, X):
        """Return 1 where the calibrated probability reaches the F1-selected threshold."""
        return (self.predict_proba(X) >= self.threshold).astype(int)
