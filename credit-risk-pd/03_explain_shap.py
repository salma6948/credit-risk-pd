"""Step 3 - explain the model with SHAP.

Uses the frozen model from step 2 and the VALIDATION set (the test set stays closed).
SHAP explains the model's raw score (log-odds), before calibration; calibration is a
monotone transformation, so the ranking of drivers is unchanged.

Findings from this step are for understanding and documentation only: do not go back and
select/drop features because of them, that would turn the validation set into training data.

Usage:  python 03_explain_shap.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src import config as C
from src.data import load_split, split_xy

N_EXPLAIN = 2000


def main() -> None:
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    model = joblib.load(C.MODEL_DIR / "pd_model.joblib")
    Xva, _ = split_xy(load_split("val"))
    sample = Xva.sample(n=min(N_EXPLAIN, len(Xva)), random_state=C.SEED)
    Z = model.preprocessor.transform(sample)

    if model.name == "lightgbm":
        sv = shap.TreeExplainer(model.estimator)(Z)
    else:  # logistic regression: background taken from TRAIN
        Xtr, _ = split_xy(load_split("train"))
        bg = model.preprocessor.transform(Xtr.sample(n=1000, random_state=C.SEED))
        sv = shap.LinearExplainer(model.estimator, bg)(Z)
    if sv.values.ndim == 3:  # some versions return one slice per class
        sv = sv[:, :, 1]

    mean_abs = pd.Series(np.abs(sv.values).mean(axis=0), index=Z.columns).sort_values(ascending=False)
    mean_abs.rename("mean_abs_shap").to_csv(C.REPORT_DIR / "shap_importance.csv")
    print("Top drivers (mean |SHAP|, log-odds):")
    print(mean_abs.head(10).round(4).to_string())

    plt.figure()
    shap.plots.beeswarm(sv, max_display=15, show=False)
    plt.title("SHAP - effect of each feature on the default score (validation sample)")
    plt.tight_layout()
    plt.savefig(C.FIG_DIR / "shap_beeswarm.png", dpi=150)
    plt.close()

    plt.figure()
    shap.plots.bar(sv, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(C.FIG_DIR / "shap_importance.png", dpi=150)
    plt.close()

    # One local explanation: the client whose calibrated PD is closest to the decision threshold
    pd_hat = model.calibrator.predict(model.estimator.predict_proba(Z)[:, 1])
    i = int(np.argmin(np.abs(pd_hat - model.threshold)))
    plt.figure()
    shap.plots.waterfall(sv[i], max_display=12, show=False)
    plt.title(f"Borderline client (PD={pd_hat[i]:.2f}, threshold={model.threshold:.3f})")
    plt.tight_layout()
    plt.savefig(C.FIG_DIR / "shap_waterfall_borderline_client.png", dpi=150)
    plt.close()
    print("Saved figures to reports/figures/ and reports/shap_importance.csv")


if __name__ == "__main__":
    main()
