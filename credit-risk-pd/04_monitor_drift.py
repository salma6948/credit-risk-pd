"""Step 4 - drift monitoring with the Population Stability Index (PSI).

The dataset has no application date, but it carries six monthly statements per client
(April-September 2005). We therefore measure *month-to-month drift on the same clients*:

  reference = TRAIN set, April 2005 (bin edges are defined on this sample only)
  monitored = VALIDATION set, each later month (May ... September 2005)

This is a monitoring DEMONSTRATION, not a test on new cohorts of customers: say so in the README.
As a noise floor we also compute PSI(train vs validation) for the same month: values of
that size are pure sampling noise.  Conventional reading: <0.10 stable, 0.10-0.25 moderate,
>0.25 significant.  The test set is not used.

Usage:  python 04_monitor_drift.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import config as C
from src.data import load_split
from src.metrics import psi_label, psi_numeric, psi_status

FAMILIES = {
    "Repayment status (PAY_x)": (C.PAY_STATUS_COLS, psi_status),
    "Bill amount (BILL_AMTx)": (C.BILL_COLS, psi_numeric),
    "Payment amount (PAY_AMTx)": (C.PAY_AMT_COLS, psi_numeric),
}
REF_MONTH = 6  # April 2005, the earliest statement


def main() -> None:
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    train, val = load_split("train"), load_split("val")
    rows = []
    for family, (cols, fn) in FAMILIES.items():
        ref_col = cols[REF_MONTH - 1]
        for k in range(REF_MONTH, 0, -1):  # April -> September
            col = cols[k - 1]
            rows.append({
                "family": family,
                "month": C.MONTH_LABEL[k],
                "column": col,
                "psi_vs_april": 0.0 if k == REF_MONTH else fn(train[ref_col], val[col]),
                "noise_floor_same_month": fn(train[col], val[col]),
            })
    out = pd.DataFrame(rows)
    out["reading"] = out["psi_vs_april"].map(psi_label)

    # Data-quality guard: a huge PSI can come from a CODING change rather than real drift.
    # In this file, status code 1 ("1 month late") is almost absent from PAY_2..PAY_6 but
    # frequent in PAY_0, so PAY_0 looks drifted when it is not comparable.
    share1 = {col: float((train[col] == 1).mean()) for col in C.PAY_STATUS_COLS}
    ref_share1 = share1[C.PAY_STATUS_COLS[REF_MONTH - 1]]
    out["share_code_1_train"] = out["column"].map(share1)
    out["coding_alert"] = (
        out["family"].str.startswith("Repayment")
        & ((out["share_code_1_train"] - ref_share1).abs() > 0.05)
    )
    out.to_csv(C.REPORT_DIR / "psi_monthly.csv", index=False)
    print(out.round(4).to_string(index=False))
    flagged = out.loc[out["coding_alert"], "column"].tolist()
    if flagged:
        print(f"\nWARNING - coding discontinuity in {flagged}: share of status code 1 is "
              f"{share1[flagged[0]]:.1%} there vs {ref_share1:.1%} in the reference month. "
              "Do NOT read this PSI as real drift; see README (Known data issues).")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for family, g in out.groupby("family", sort=False):
        ax.plot(g["month"], g["psi_vs_april"], marker="o", label=family)
    ax.axhline(C.PSI_MODERATE, ls="--", c="grey", lw=1)
    ax.axhline(C.PSI_SIGNIFICANT, ls="--", c="red", lw=1)
    ax.text(0.02, C.PSI_MODERATE + 0.005, "0.10 moderate", fontsize=8, color="grey")
    ax.text(0.02, C.PSI_SIGNIFICANT + 0.005, "0.25 significant", fontsize=8, color="red")
    for _, r in out[out["coding_alert"]].iterrows():
        ax.annotate("coding\nchange", (r["month"], r["psi_vs_april"]), textcoords="offset points",
                    xytext=(-38, -8), fontsize=8, color="darkred")
    ax.set_ylabel("PSI vs April 2005 (reference = train)")
    ax.set_title("Month-to-month drift, same clients (validation set)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "psi_monthly.png", dpi=150)
    print("Saved reports/psi_monthly.csv and reports/figures/psi_monthly.png")


if __name__ == "__main__":
    main()
