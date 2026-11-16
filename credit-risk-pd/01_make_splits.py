"""Step 1 - minimal cleaning, then 60/20/20 stratified split.

Order matters: duplicates are removed BEFORE splitting (otherwise two copies of the same
client could land in train and test), and nothing else is done to the data here. All other
preprocessing is learned later, on the training set only.

Usage:  python 01_make_splits.py [--force]
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

from src import config as C
from src.data import load_raw


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(force: bool) -> None:
    C.SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    existing = [C.SPLIT_DIR / f"{n}.csv" for n in ("train", "val", "test")]
    if any(p.exists() for p in existing) and not force:
        sys.exit(
            "Splits already exist. Re-splitting after looking at results would let test "
            "information leak into your choices. Use --force only if you start over."
        )

    df = load_raw()
    assert df[C.ID_COL].is_unique, "ID must be unique"
    assert df.isna().sum().sum() == 0, "unexpected missing values"
    assert set(df[C.TARGET].unique()) == {0, 1}, "target must be binary"
    n_raw = len(df)

    # 1) Remove exact duplicates (all columns identical except ID). Keep the first one.
    key = [c for c in df.columns if c != C.ID_COL]
    dup_mask = df.duplicated(subset=key, keep="first")
    removed = df[dup_mask]
    df = df[~dup_mask].reset_index(drop=True)
    removed.to_csv(C.SPLIT_DIR / "removed_duplicates.csv", index=False)

    # 2) Stratified 60/20/20 split on the target, fixed seed.
    trainval, test = train_test_split(
        df, test_size=C.TEST_SIZE, stratify=df[C.TARGET], random_state=C.SEED
    )
    train, val = train_test_split(
        trainval,
        test_size=C.VAL_SIZE / (1 - C.TEST_SIZE),
        stratify=trainval[C.TARGET],
        random_state=C.SEED,
    )
    ids = [set(s[C.ID_COL]) for s in (train, val, test)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2]), "overlap!"
    assert sum(len(s) for s in (train, val, test)) == len(df)

    manifest = {
        "rows_in_raw_file": n_raw,
        "exact_duplicates_removed": int(dup_mask.sum()),
        "rows_after_cleaning": len(df),
        "seed": C.SEED,
        "sklearn_version": sklearn.__version__,
        "splits": {},
    }
    for name, part in (("train", train), ("val", val), ("test", test)):
        path = C.SPLIT_DIR / f"{name}.csv"
        part.sort_values(C.ID_COL).to_csv(path, index=False)
        manifest["splits"][name] = {
            "rows": len(part),
            "defaults": int(part[C.TARGET].sum()),
            "default_rate": round(float(part[C.TARGET].mean()), 4),
            "sha256": sha256(path),
        }
    (C.SPLIT_DIR / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="overwrite existing splits")
    main(ap.parse_args().force)
