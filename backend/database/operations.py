import pandas as pd
from pathlib import Path
from typing import Optional


class DataError(Exception):
    pass


class DataFormatError(Exception):
    pass


DATA_DIR = Path("data/processed")
DATA_FILE = DATA_DIR / "data_2025.csv"
INITIAL_DATA_FILE = DATA_DIR / "initial_data.csv"
STANDARDS_FILE = DATA_DIR / "standards.csv"


def ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_current_data_filename() -> str:
    ensure_data_dir()
    return str(DATA_FILE)


async def load_data() -> pd.DataFrame:
    ensure_data_dir()
    if not DATA_FILE.exists():
        return pd.DataFrame(columns=["date", "avg_pullups"])
    return pd.read_csv(DATA_FILE)


async def save_data(df: pd.DataFrame):
    ensure_data_dir()
    df.to_csv(DATA_FILE, index=False)


async def load_initial_data() -> pd.DataFrame:
    ensure_data_dir()
    if not INITIAL_DATA_FILE.exists():
        return pd.DataFrame(columns=["date", "avg_pullups"])
    return pd.read_csv(INITIAL_DATA_FILE)


async def reset_data():
    ensure_data_dir()
    if DATA_FILE.exists():
        DATA_FILE.unlink()


def load_original_standards() -> pd.DataFrame:
    ensure_data_dir()
    if not STANDARDS_FILE.exists():
        return pd.DataFrame(columns=["weight_category", "rank", "max_pullups"])
    return pd.read_csv(STANDARDS_FILE)
