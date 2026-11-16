"""Guards against data leakage. Run after 01_make_splits.py:  pytest -q"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src import config as C
from src.data import load_split, split_xy
from src.features import build_preprocessor, prepare


def test_splits_are_disjoint_and_complete():
    ids = {n: set(load_split(n)[C.ID_COL]) for n in ("train", "val", "test")}
    assert not (ids["train"] & ids["val"]) and not (ids["train"] & ids["test"]) and not (ids["val"] & ids["test"])


def test_no_duplicate_clients_across_splits():
    key = [c for c in load_split("train").columns if c != C.ID_COL]
    frames = [load_split(n)[key].assign(split=n) for n in ("train", "val", "test")]
    allrows = pd.concat(frames)
    dup = allrows.duplicated(subset=key, keep=False)
    assert not dup.any(), "identical clients found (possibly across splits)"


def test_preprocessor_statistics_come_from_train_only():
    Xtr, _ = split_xy(load_split("train"))
    Xva, _ = split_xy(load_split("val"))
    pre = build_preprocessor(linear=True).fit(Xtr)
    scaler = pre.named_steps["columns"].named_transformers_["num"]
    prepared_train = prepare(Xtr, log_amounts=True)[C.MODEL_NUMERIC]
    np.testing.assert_allclose(scaler.mean_, prepared_train.mean().to_numpy(), rtol=1e-9)
    prepared_val = prepare(Xva, log_amounts=True)[C.MODEL_NUMERIC]
    assert not np.allclose(scaler.mean_, prepared_val.mean().to_numpy(), rtol=1e-9)


def test_stateless_features_do_not_depend_on_other_rows():
    X, _ = split_xy(load_split("train"))
    full = prepare(X, log_amounts=False)
    part = prepare(X.iloc[:50], log_amounts=False)
    pd.testing.assert_frame_equal(full.iloc[:50], part)


def test_test_set_is_only_read_by_final_evaluation():
    for script in ("02_train.py", "03_explain_shap.py", "04_monitor_drift.py"):
        code = (ROOT / script).read_text()
        assert not re.search(r"load_split\(\s*['\"]test['\"]", code), f"{script} reads the test set"
