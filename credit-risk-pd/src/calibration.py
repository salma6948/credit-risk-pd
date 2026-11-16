"""Probability calibration, fitted on the validation set only."""
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold

_EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, float), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


class Identity:
    name = "none"

    def fit(self, p, y):
        return self

    def predict(self, p):
        return np.clip(np.asarray(p, float), _EPS, 1 - _EPS)


class Platt:
    name = "sigmoid"

    def fit(self, p, y):
        self.lr_ = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p).reshape(-1, 1), y)
        return self

    def predict(self, p):
        return self.lr_.predict_proba(_logit(p).reshape(-1, 1))[:, 1]


class Isotonic:
    name = "isotonic"

    def fit(self, p, y):
        self.iso_ = IsotonicRegression(y_min=_EPS, y_max=1 - _EPS, out_of_bounds="clip").fit(p, y)
        return self

    def predict(self, p):
        return self.iso_.predict(np.asarray(p, float))


def select_and_fit(raw_val, y_val, seed, n_splits=5):
    """Pick the calibrator (none / sigmoid / isotonic) by out-of-fold log-loss *inside the
    validation set*, then refit the winner on the whole validation set.
    Returns (fitted_calibrator, comparison_rows)."""
    raw_val = np.asarray(raw_val, float)
    y_val = np.asarray(y_val)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows, best = [], None
    for cls in (Identity, Platt, Isotonic):
        oof = np.zeros_like(raw_val)
        for tr, te in cv.split(raw_val, y_val):
            oof[te] = cls().fit(raw_val[tr], y_val[tr]).predict(raw_val[te])
        row = {
            "calibrator": cls.name,
            "oof_log_loss": float(log_loss(y_val, oof)),
            "oof_brier": float(brier_score_loss(y_val, oof)),
        }
        rows.append(row)
        if best is None or row["oof_log_loss"] < best[0]:
            best = (row["oof_log_loss"], cls)
    return best[1]().fit(raw_val, y_val), rows
