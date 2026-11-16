"""Feature engineering.

Everything that *learns* something (scaling, one-hot categories) lives inside a
scikit-learn Pipeline and is therefore fitted on the training set only.
Everything stateless (regrouping codes, ratios, log transform) is a fixed rule that
does not depend on any dataset, so applying it before or after the split is equivalent.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src import config as C


def regroup_codes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["EDUCATION"] = df["EDUCATION"].replace(C.EDUCATION_REGROUP)
    df["MARRIAGE"] = df["MARRIAGE"].replace(C.MARRIAGE_REGROUP)
    return df


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise features: each client's value depends on that client only."""
    df = df.copy()
    df["UTILISATION_MEAN"] = df[C.BILL_COLS].mean(axis=1) / df["LIMIT_BAL"].clip(lower=1)
    df["MAX_DELAY"] = df[C.PAY_STATUS_COLS].max(axis=1)
    df["N_LATE_MONTHS"] = (df[C.PAY_STATUS_COLS] > 0).sum(axis=1)
    billed = df[C.BILL_COLS].clip(lower=0).sum(axis=1)
    paid = df[C.PAY_AMT_COLS].sum(axis=1)
    df["PAYMENT_RATIO"] = (paid / billed.clip(lower=1)).clip(upper=5)
    return df


def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


def prepare(df: pd.DataFrame, log_amounts: bool) -> pd.DataFrame:
    df = add_derived(regroup_codes(df))
    if log_amounts:
        cols = ["LIMIT_BAL"] + C.BILL_COLS + C.PAY_AMT_COLS
        df[cols] = signed_log(df[cols])
    return df


def build_preprocessor(linear: bool) -> Pipeline:
    """linear=True : log-scaled amounts + standardisation (for logistic regression).
    linear=False: raw amounts, no scaling (trees; keeps SHAP plots readable).
    Columns not listed (ID-like, SEX...) are dropped by the ColumnTransformer."""
    numeric_step = StandardScaler() if linear else "passthrough"
    columns = ColumnTransformer(
        [
            ("num", numeric_step, C.MODEL_NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), C.MODEL_CATEGORICAL),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    columns.set_output(transform="pandas")
    return Pipeline(
        [
            ("prepare", FunctionTransformer(prepare, kw_args={"log_amounts": linear}, validate=False)),
            ("columns", columns),
        ]
    )
