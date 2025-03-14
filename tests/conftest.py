import pytest
import pandas as pd
from pathlib import Path
from backend.database.operations import DATA_FILE, ensure_data_dir


@pytest.fixture(autouse=True)
def setup_test_data():
    ensure_data_dir()
    # Создаем тестовые данные
    test_data = pd.DataFrame({"date": ["2024-03-13"], "avg_pullups": [10.0]})
    test_data.to_csv(DATA_FILE, index=False)
    yield
    # Очищаем тестовые данные после тестов
    if DATA_FILE.exists():
        DATA_FILE.unlink()
