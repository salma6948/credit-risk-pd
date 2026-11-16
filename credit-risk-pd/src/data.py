"""Loading helpers. No statistic is learned from the data here."""
import pandas as pd

from src import config as C


def load_raw(path=None) -> pd.DataFrame:
    """Read the UCI 'default of credit card clients' workbook.

    The first sheet row holds generic labels (X1..X23); the real header is row 2.
    """
    path = path or C.RAW_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download 'default of credit card clients' from the UCI "
            "repository and put the .xls file in data/raw/."
        )
    df = pd.read_excel(path, header=1)
    expected = {C.ID_COL, C.TARGET, "SEX", "EDUCATION", "MARRIAGE", "AGE", "LIMIT_BAL"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Unexpected file layout, missing columns: {sorted(missing)}")
    return df


def load_split(name: str) -> pd.DataFrame:
    if name not in {"train", "val", "test"}:
        raise ValueError(name)
    return pd.read_csv(C.SPLIT_DIR / f"{name}.csv")


def split_xy(df: pd.DataFrame):
    """Features keep every column except ID and target (SEX included, for auditing);
    the model's ColumnTransformer decides which ones it actually uses."""
    y = df[C.TARGET].astype(int)
    X = df.drop(columns=[C.ID_COL, C.TARGET])
    return X, y
