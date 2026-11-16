from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class PDModel:
    """Fitted preprocessor + estimator + calibrator + decision threshold, in one object."""

    name: str
    preprocessor: Any
    estimator: Any
    calibrator: Any
    threshold: float

    def predict_raw(self, X) -> np.ndarray:
        """Uncalibrated score (what SHAP explains)."""
        return self.estimator.predict_proba(self.preprocessor.transform(X))[:, 1]

    def predict(self, X) -> np.ndarray:
        """Calibrated probability of default."""
        return self.calibrator.predict(self.predict_raw(X))

    def decide(self, X) -> np.ndarray:
        """True = flagged as likely defaulter."""
        return self.predict(X) >= self.threshold
