"""Scoring, calibration, drift (PSI) and fairness metrics."""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from src import config as C

_EPS = 1e-6


# --- Discrimination & calibration ---------------------------------------------
def ks_statistic(y, p) -> float:
    y, p = np.asarray(y), np.asarray(p)
    return float(ks_2samp(p[y == 1], p[y == 0]).statistic)


def reliability_table(y, p, n_bins=10) -> pd.DataFrame:
    """Equal-frequency bins (about the same number of clients in each)."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), n_bins, labels=False)
    return (
        df.groupby("bin")
        .agg(n=("y", "size"), mean_pred=("p", "mean"), obs_rate=("y", "mean"))
        .reset_index()
    )


def expected_calibration_error(y, p, n_bins=10) -> float:
    t = reliability_table(y, p, n_bins)
    return float((t["n"] * (t["mean_pred"] - t["obs_rate"]).abs()).sum() / t["n"].sum())


def score_summary(y, p) -> dict:
    y = np.asarray(y)
    p = np.clip(np.asarray(p, float), _EPS, 1 - _EPS)
    auc = roc_auc_score(y, p)
    return {
        "auc": float(auc),
        "gini": float(2 * auc - 1),
        "pr_auc": float(average_precision_score(y, p)),
        "ks": ks_statistic(y, p),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p)),
        "ece": expected_calibration_error(y, p),
        "mean_pred": float(p.mean()),
        "observed_rate": float(y.mean()),
    }


def bootstrap_ci(y, p, fn, n_boot=1000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    idx = np.arange(len(y))
    vals = []
    for _ in range(n_boot):
        s = rng.choice(idx, size=len(idx), replace=True)
        if y[s].min() == y[s].max():
            continue
        vals.append(fn(y[s], p[s]))
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def confusion_at(y, p, thr) -> dict:
    y = np.asarray(y)
    flag = np.asarray(p) >= thr
    tp = int((flag & (y == 1)).sum())
    fp = int((flag & (y == 0)).sum())
    fn = int((~flag & (y == 1)).sum())
    tn = int((~flag & (y == 0)).sum())
    return {
        "threshold": float(thr),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "flag_rate": float(flag.mean()),
        "recall": tp / max(tp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "fpr": fp / max(fp + tn, 1),
        "expected_cost_per_client": (C.COST_FN * fn + C.COST_FP * fp) / len(y),
    }


# --- Population Stability Index ------------------------------------------------
def _psi_from_counts(expected, actual) -> float:
    e = np.clip(expected / expected.sum(), 1e-4, None)
    a = np.clip(actual / actual.sum(), 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_numeric(reference, current, n_bins=10) -> float:
    """Bin edges are defined on the REFERENCE sample only (quantiles), then applied to both."""
    reference, current = np.asarray(reference, float), np.asarray(current, float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, n_bins + 1)[1:-1]))
    bins = np.concatenate(([-np.inf], edges, [np.inf]))
    e, _ = np.histogram(reference, bins=bins)
    a, _ = np.histogram(current, bins=bins)
    return _psi_from_counts(e.astype(float), a.astype(float))


def psi_status(reference, current, low=-2, high=3) -> float:
    """Repayment-status codes: clipped to [-2, 3] so that '3+' is one category."""
    r = pd.Series(reference).clip(low, high)
    c = pd.Series(current).clip(low, high)
    cats = list(range(low, high + 1))
    e = r.value_counts().reindex(cats, fill_value=0).to_numpy(float)
    a = c.value_counts().reindex(cats, fill_value=0).to_numpy(float)
    return _psi_from_counts(e, a)


def psi_label(value: float) -> str:
    if value < C.PSI_MODERATE:
        return "stable"
    return "moderate" if value < C.PSI_SIGNIFICANT else "significant"


# --- Fairness ------------------------------------------------------------------
def fairness_table(groups: dict, y, p, thr, min_n=200, min_pos=30) -> pd.DataFrame:
    """One row per (attribute, level). `groups` maps attribute name -> Series of labels
    aligned with y. Positive class = default; 'flag' = predicted default (loan refused).
    Disparities are measured against the largest level of each attribute."""
    y = np.asarray(y)
    p = np.asarray(p)
    flag = p >= thr
    rows = []
    for attr, labels in groups.items():
        labels = np.asarray(labels)
        levels = pd.unique(labels)
        ref_level = max(levels, key=lambda lv: (labels == lv).sum())
        block = []
        for lv in levels:
            m = labels == lv
            pos, neg = (m & (y == 1)).sum(), (m & (y == 0)).sum()
            block.append({
                "attribute": attr,
                "level": lv,
                "n": int(m.sum()),
                "n_default": int(pos),
                "observed_rate": float(y[m].mean()),
                "mean_pred": float(p[m].mean()),
                "calibration_gap": float(p[m].mean() - y[m].mean()),
                "flag_rate": float(flag[m].mean()),
                "tpr": float(flag[m & (y == 1)].mean()) if pos else np.nan,
                "fpr": float(flag[m & (y == 0)].mean()) if neg else np.nan,
                "precision": float(y[m & flag].mean()) if (m & flag).any() else np.nan,
                "small_sample": bool(m.sum() < min_n or pos < min_pos),
                "reference": lv == ref_level,
            })
        ref = next(r for r in block if r["reference"])
        for r in block:
            r["approval_ratio_vs_ref"] = (1 - r["flag_rate"]) / max(1 - ref["flag_rate"], 1e-9)
            r["tpr_gap_vs_ref"] = r["tpr"] - ref["tpr"]
            r["fpr_gap_vs_ref"] = r["fpr"] - ref["fpr"]
        rows.extend(block)
    return pd.DataFrame(rows)
