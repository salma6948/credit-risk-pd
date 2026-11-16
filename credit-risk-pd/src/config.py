"""Single source of truth for paths, column groups and modelling choices."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "default of credit card clients.xls"
SPLIT_DIR = ROOT / "data" / "splits"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"

SEED = 42
ID_COL = "ID"
TARGET = "default payment next month"

# 60 / 20 / 20 split. VAL_SIZE is a fraction of the *full* dataset.
TEST_SIZE = 0.20
VAL_SIZE = 0.20

# --- Columns -----------------------------------------------------------------
# Month index k = 1..6 runs from September 2005 (k=1) back to April 2005 (k=6).
# The repayment-status column for k=1 is called PAY_0 in the file (there is no PAY_1).
MONTH_LABEL = {1: "2005-09", 2: "2005-08", 3: "2005-07", 4: "2005-06", 5: "2005-05", 6: "2005-04"}
PAY_STATUS_COLS = ["PAY_0"] + [f"PAY_{k}" for k in range(2, 7)]
BILL_COLS = [f"BILL_AMT{k}" for k in range(1, 7)]
PAY_AMT_COLS = [f"PAY_AMT{k}" for k in range(1, 7)]
DERIVED_COLS = ["UTILISATION_MEAN", "MAX_DELAY", "N_LATE_MONTHS", "PAYMENT_RATIO"]

# --- Modelling choices -------------------------------------------------------
# SEX is deliberately NOT a model feature (protected attribute), but it is kept in
# the raw frames so that script 05 can audit the model for disparities on it.
MODEL_NUMERIC = ["LIMIT_BAL", "AGE"] + PAY_STATUS_COLS + BILL_COLS + PAY_AMT_COLS + DERIVED_COLS
MODEL_CATEGORICAL = ["EDUCATION", "MARRIAGE"]
AUDIT_ATTRIBUTES = ["SEX", "AGE", "MARRIAGE", "EDUCATION"]

# Rare / undocumented codes, regrouped with a rule fixed in advance (target never used).
EDUCATION_REGROUP = {0: 4, 5: 4, 6: 4}   # -> "others"
MARRIAGE_REGROUP = {0: 3}                # -> "others"

# Business costs used ONLY to choose the decision threshold on the validation set.
# These are assumptions, not facts: a missed defaulter is assumed 5x as costly as
# wrongly flagging a good customer. Change them and re-run 02_train.py.
COST_FN = 5.0   # defaulter not flagged
COST_FP = 1.0   # good customer flagged

# Population Stability Index conventions
PSI_MODERATE = 0.10
PSI_SIGNIFICANT = 0.25
