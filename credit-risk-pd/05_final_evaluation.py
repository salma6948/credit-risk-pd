"""Step 5 - final evaluation on the TEST set. Run it ONCE, at the very end.

The script refuses to run a second time (it writes reports/test_evaluation.json) unless
--force is given: re-running after tweaking the model would turn the test set into a
second validation set.

Reports: discrimination (AUC/Gini/KS/PR-AUC), calibration (Brier, log-loss, ECE, reliability
diagram), the business decision at the threshold fixed on validation, and a fairness audit
on SEX / AGE / MARRIAGE / EDUCATION (SEX is not a model feature, but is audited anyway:
removing a column does not remove its proxies).

Usage:  python 05_final_evaluation.py [--force]
        python 05_final_evaluation.py --smoke-test-on-val   (checks the code on the validation
                                                              set, leaves the test set untouched)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve

from src import config as C
from src.data import load_split, split_xy
from src.features import regroup_codes
from src.metrics import (bootstrap_ci, confusion_at, fairness_table, reliability_table,
                         score_summary)



def age_band(age: pd.Series) -> pd.Series:
    return pd.cut(age, [0, 29, 39, 49, 200], labels=["<30", "30-39", "40-49", "50+"]).astype(str)


def main(force: bool, smoke: bool) -> None:
    split = "val" if smoke else "test"
    tag = "_smoketest_on_val" if smoke else ""
    lock = C.REPORT_DIR / f"test_evaluation{tag}.json"
    if lock.exists() and not force and not smoke:
        sys.exit(f"{lock.name} already exists: the test set has been used. Use --force to override "
                 "(and say so in your write-up).")
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    model = joblib.load(C.MODEL_DIR / "pd_model.joblib")
    test = load_split(split)
    if smoke:
        print("*** SMOKE TEST on the VALIDATION set: numbers are optimistic, test set untouched ***")
    X, y = split_xy(test)
    raw = model.predict_raw(X)
    p = model.predict(X)
    thr = model.threshold

    # ---- Discrimination & calibration -------------------------------------------
    cal = score_summary(y, p)
    raw_s = score_summary(y, raw)
    ci = {
        "auc": bootstrap_ci(y, p, roc_auc_score, seed=C.SEED),
        "brier": bootstrap_ci(y, p, brier_score_loss, seed=C.SEED),
        "log_loss": bootstrap_ci(y, p, log_loss, seed=C.SEED),
    }
    decision = confusion_at(y, p, thr)
    print(f"Model: {model.name} | calibrator: {model.calibrator.name} | threshold {thr:.3f}")
    print(f"AUC {cal['auc']:.4f} (95% CI {ci['auc'][0]:.4f}-{ci['auc'][1]:.4f}) | Gini {cal['gini']:.4f} "
          f"| KS {cal['ks']:.4f} | PR-AUC {cal['pr_auc']:.4f}")
    print(f"Brier {cal['brier']:.4f} (95% CI {ci['brier'][0]:.4f}-{ci['brier'][1]:.4f}) "
          f"| log-loss {cal['log_loss']:.4f} | ECE {cal['ece']:.4f}")
    print(f"mean predicted PD {cal['mean_pred']:.4f} vs observed default rate {cal['observed_rate']:.4f}")
    print("Decision at validation threshold:", {k: round(v, 4) if isinstance(v, float) else v
                                                for k, v in decision.items()})

    # ---- Figures --------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ax = axes[0]
    for label, scores in (("raw score", raw), (f"calibrated ({model.calibrator.name})", p)):
        t = reliability_table(y, scores)
        ax.plot(t["mean_pred"], t["obs_rate"], marker="o", label=label)
    top = max(reliability_table(y, p)["obs_rate"].max(), reliability_table(y, p)["mean_pred"].max()) * 1.05
    ax.plot([0, top], [0, top], "k--", lw=1, label="perfect")
    ax.set(xlabel="mean predicted PD", ylabel="observed default rate", title="Reliability (test, decile bins)")
    ax.legend(fontsize=8)
    fpr, tpr, _ = roc_curve(y, p)
    axes[1].plot(fpr, tpr)
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].set(xlabel="false positive rate", ylabel="true positive rate",
                title=f"ROC (AUC={cal['auc']:.3f})")
    axes[2].hist(p[y == 0], bins=40, alpha=0.6, density=True, label="no default")
    axes[2].hist(p[y == 1], bins=40, alpha=0.6, density=True, label="default")
    axes[2].axvline(thr, c="k", ls="--", lw=1)
    axes[2].set(xlabel="calibrated PD", title="Score distribution")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / f"test_performance{tag}.png", dpi=150)
    plt.close(fig)

    # ---- Fairness audit ---------------------------------------------------------------
    clean = regroup_codes(X)
    groups = {
        "SEX": X["SEX"].map({1: "male", 2: "female"}),
        "AGE": age_band(X["AGE"]),
        "MARRIAGE": clean["MARRIAGE"].map({1: "married", 2: "single", 3: "other"}),
        "EDUCATION": clean["EDUCATION"].map({1: "graduate school", 2: "university",
                                             3: "high school", 4: "other"}),
    }
    fair = fairness_table(groups, y, p, thr)
    fair.to_csv(C.REPORT_DIR / f"fairness_test{tag}.csv", index=False)
    show = ["attribute", "level", "n", "n_default", "observed_rate", "mean_pred", "flag_rate", "tpr", "fpr",
            "approval_ratio_vs_ref", "tpr_gap_vs_ref", "fpr_gap_vs_ref", "small_sample"]
    print("\nFairness audit (flag = predicted default = refusal; reference = largest level):")
    print(fair[show].round(3).to_string(index=False))
    worst = fair[~fair["reference"] & ~fair["small_sample"]].sort_values("approval_ratio_vs_ref").head(1)
    if len(worst):
        r = worst.iloc[0]
        print(f"\nLowest approval ratio vs reference (non-small groups): {r['attribute']}={r['level']} "
              f"-> {r['approval_ratio_vs_ref']:.2f} (the '4/5 rule' of thumb flags values below 0.80; "
              "indicative only, not a legal test).")

    # ---- Persist & lock -----------------------------------------------------------------
    lock.write_text(json.dumps({
        "model": model.name,
        "calibrator": model.calibrator.name,
        "threshold": thr,
        "test_rows": len(test),
        "calibrated": cal,
        "raw_score": raw_s,
        "bootstrap_95ci": ci,
        "decision_at_threshold": decision,
    }, indent=2))
    print(f"\nSaved reports/{lock.name}, reports/fairness_test{tag}.csv, reports/figures/test_performance{tag}.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke-test-on-val", action="store_true",
                    help="run the whole script on the validation set (never touches test)")
    a = ap.parse_args()
    main(a.force, a.smoke_test_on_val)
