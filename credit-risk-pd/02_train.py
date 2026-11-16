"""Step 2 - baseline + stronger model, calibration, decision threshold.

Only train.csv and val.csv are read here. The test set is never opened.

  * Preprocessing is fitted on TRAIN only (inside a Pipeline).
  * Hyper-parameters and early stopping are chosen on VALIDATION only.
  * The calibrator is fitted on VALIDATION predictions of the frozen model.
  * The decision threshold is chosen on VALIDATION by minimising expected cost.

Usage:  python 02_train.py
"""
import itertools
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression

from src import config as C
from src.calibration import select_and_fit
from src.data import load_split, split_xy
from src.features import build_preprocessor
from src.metrics import confusion_at, score_summary
from src.model import PDModel

# `eval_set` works in every LightGBM version; recent ones only warn that it was renamed.
warnings.filterwarnings("ignore", message=".*eval_set.*")


def main() -> None:
    C.MODEL_DIR.mkdir(exist_ok=True)
    C.REPORT_DIR.mkdir(exist_ok=True)
    Xtr, ytr = split_xy(load_split("train"))
    Xva, yva = split_xy(load_split("val"))
    print(f"train: {len(Xtr)} rows | val: {len(Xva)} rows | test: not loaded")

    candidates = []  # (val_log_loss, name, preprocessor, estimator, params, val_scores)

    # ---- Baseline: logistic regression ------------------------------------------
    pre_lin = build_preprocessor(linear=True).fit(Xtr)         # fitted on train only
    Ztr, Zva = pre_lin.transform(Xtr), pre_lin.transform(Xva)
    for c in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
        est = LogisticRegression(C=c, max_iter=5000).fit(Ztr, ytr)
        p = est.predict_proba(Zva)[:, 1]
        s = score_summary(yva, p)
        candidates.append((s["log_loss"], "logistic_regression", pre_lin, est, {"C": c}, s))

    # ---- Stronger model: LightGBM, early stopping on validation ------------------
    pre_tree = build_preprocessor(linear=False).fit(Xtr)
    Ttr, Tva = pre_tree.transform(Xtr), pre_tree.transform(Xva)
    grid = itertools.product((7, 15, 31), (50, 100, 200), (1.0, 10.0))
    for num_leaves, min_child, reg_lambda in grid:
        est = lgb.LGBMClassifier(
            objective="binary", learning_rate=0.03, n_estimators=3000,
            num_leaves=num_leaves, min_child_samples=min_child, reg_lambda=reg_lambda,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            random_state=C.SEED, deterministic=True, force_row_wise=True, verbose=-1,
        )
        est.fit(Ttr, ytr, eval_set=[(Tva, yva)], eval_metric="binary_logloss",
                callbacks=[lgb.early_stopping(100, verbose=False)])
        p = est.predict_proba(Tva)[:, 1]
        s = score_summary(yva, p)
        params = {"num_leaves": num_leaves, "min_child_samples": min_child,
                  "reg_lambda": reg_lambda, "best_iteration": int(est.best_iteration_)}
        candidates.append((s["log_loss"], "lightgbm", pre_tree, est, params, s))

    # ---- Model selection on validation log-loss (a proper scoring rule) ----------
    candidates.sort(key=lambda t: t[0])
    best_per_family = {}
    for cand in candidates:
        best_per_family.setdefault(cand[1], cand)
    _, name, pre, est, params, val_scores = candidates[0]
    print("\nBest per family (validation):")
    for fam, (_, _, _, _, prm, s) in best_per_family.items():
        print(f"  {fam:20s} AUC={s['auc']:.4f}  logloss={s['log_loss']:.4f}  brier={s['brier']:.4f}  {prm}")
    print(f"=> selected: {name} {params}")

    # ---- Calibration fitted on validation predictions ---------------------------
    raw_val = est.predict_proba(pre.transform(Xva))[:, 1]
    calibrator, calib_rows = select_and_fit(raw_val, yva, seed=C.SEED)
    p_val = calibrator.predict(raw_val)
    print("\nCalibration candidates (out-of-fold on validation):")
    for r in calib_rows:
        print(f"  {r['calibrator']:9s} logloss={r['oof_log_loss']:.4f} brier={r['oof_brier']:.4f}")
    print(f"=> calibrator: {calibrator.name}")

    # ---- Decision threshold on validation, by expected cost ---------------------
    grid_thr = np.round(np.arange(0.02, 0.90, 0.005), 3)
    costs = [confusion_at(yva, p_val, t)["expected_cost_per_client"] for t in grid_thr]
    thr = float(grid_thr[int(np.argmin(costs))])
    theoretical = C.COST_FP / (C.COST_FP + C.COST_FN)
    print(f"\nThreshold (min expected cost on validation): {thr:.3f} "
          f"[theoretical optimum for perfectly calibrated PDs: {theoretical:.3f}]")
    print("Validation confusion at threshold:", {k: round(v, 4) if isinstance(v, float) else v
                                                for k, v in confusion_at(yva, p_val, thr).items()})

    model = PDModel(name=name, preprocessor=pre, estimator=est, calibrator=calibrator, threshold=thr)
    joblib.dump(model, C.MODEL_DIR / "pd_model.joblib")

    report = {
        "selected": {"family": name, "params": params, "calibrator": calibrator.name, "threshold": thr},
        "validation_scores_raw": val_scores,
        "validation_scores_calibrated": score_summary(yva, p_val),
        "validation_at_threshold": confusion_at(yva, p_val, thr),
        "cost_assumptions": {"cost_fn": C.COST_FN, "cost_fp": C.COST_FP, "theoretical_threshold": theoretical},
        "calibration_comparison": calib_rows,
        "all_candidates": [
            {"family": f, "params": prm, **{k: s[k] for k in ("auc", "log_loss", "brier")}}
            for _, f, _, _, prm, s in candidates
        ],
    }
    (C.REPORT_DIR / "validation_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nSaved {C.MODEL_DIR / 'pd_model.joblib'} and reports/validation_report.json")


if __name__ == "__main__":
    main()
