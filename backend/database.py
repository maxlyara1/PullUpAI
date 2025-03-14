# backend/database.py
import pandas as pd
import os
from datetime import datetime
import logging
import aiofiles  # Добавляем aiofiles
import aiofiles.os  # Добавляем aiofiles.os

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

DATA_DIR = "data"
DATA_2025_FILENAME = "data_2025.csv"
INITIAL_DATA_FILENAME = "initial_data.csv"
ORIGINAL_STANDARDS_FILENAME = "original_standards.csv"


class DataError(Exception):
    pass


class DataNotFoundError(DataError):
    pass


class DataEmptyError(DataError):
    pass


class DataSaveError(DataError):
    pass


class DataFormatError(DataError):  # Custom exception for format errors
    pass


async def load_data():
    """Loads data from CSV, validating the date format."""
    filepath = os.path.join(DATA_DIR, DATA_2025_FILENAME)
    try:
        if not await aiofiles.os.path.exists(filepath):  # Используем aiofiles.os
            raise DataNotFoundError(f"File not found: {filepath}")

        async with aiofiles.open(
            filepath, mode="r", encoding="utf-8"
        ) as f:  # Используем aiofiles
            content = await f.read()

        # Используем StringIO, чтобы передать содержимое файла в pandas
        from io import StringIO

        df = pd.read_csv(StringIO(content))

        if df.empty:
            raise DataEmptyError(f"File is empty: {filepath}")

        # Validate date format.
        try:
            pd.to_datetime(df["date"], format="%d.%m.%Y")
        except ValueError as e:
            raise DataFormatError(
                f"Invalid date format in {filepath}.  Expected DD.MM.YYYY. Error: {e}"
            ) from e

        return df

    except (DataError, pd.errors.ParserError) as e:
        logger.error(f"Error loading data: {e}")
        raise


async def save_data(df):
    """Saves data to CSV."""
    filepath = os.path.join(DATA_DIR, DATA_2025_FILENAME)
    try:
        #  Сохраняем в строку, а затем асинхронно записываем в файл
        async with aiofiles.open(filepath, mode="w", encoding="utf-8") as f:
            await f.write(df.to_csv(index=False))

    except Exception as e:
        logger.exception(f"Error saving data: {e}")
        raise DataSaveError(f"Error saving data to {filepath}: {e}") from e


async def load_initial_data():
    """Loads initial data, validating the date format."""
    filepath = os.path.join(DATA_DIR, INITIAL_DATA_FILENAME)
    try:
        if not await aiofiles.os.path.exists(filepath):  # Используем aiofiles.os
            raise DataNotFoundError(f"File not found: {filepath}")

        async with aiofiles.open(
            filepath, mode="r", encoding="utf-8"
        ) as f:  # Используем aiofiles
            content = await f.read()

        from io import StringIO

        df = pd.read_csv(StringIO(content))

        if df.empty:
            raise DataEmptyError(f"File is empty: {filepath}")

        # Validate Date format
        try:
            pd.to_datetime(df["date"], format="%d.%m.%Y")
        except ValueError as e:
            raise DataFormatError(
                f"Invalid date format in {filepath}. Expected DD.MM.YYYY. Error: {e}"
            ) from e

        return df

    except (DataError, pd.errors.ParserError) as e:
        logger.error(f"Error loading initial data: {e}")
        raise


async def reset_data():
    """Resets the 2025 data."""
    filepath = os.path.join(DATA_DIR, DATA_2025_FILENAME)
    df = pd.DataFrame(columns=["date", "avg_pullups"])
    await save_data(df)  # Используем await, так как save_data теперь асинхронная


def get_current_data_filename():
    """Returns the current data filename."""
    return os.path.join(DATA_DIR, DATA_2025_FILENAME)


def load_original_standards():
    """Loads original standards from CSV."""
    filepath = os.path.join(DATA_DIR, ORIGINAL_STANDARDS_FILENAME)
    if not os.path.exists(filepath):
        return pd.DataFrame()  # Return empty DataFrame if file not found
    try:
        df = pd.read_csv(filepath, sep=",")
        return df
    except pd.errors.EmptyDataError:
        return pd.DataFrame()  # Return empty DataFrame if file is empty
    except Exception as e:
        logger.exception(f"Error loading original standards: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error
