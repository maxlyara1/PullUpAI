import pandas as pd
from datetime import datetime
from backend.database.operations import (
    load_data,
    save_data,
    reset_data,
    DataFormatError,
)
from backend.models.training import TrainingData


async def add_training_data(
    date: str, avg_pullups: float, total_pullups: int | None = None
):
    """Добавить новые данные о тренировке в базу данных."""
    df = await load_data()
    try:
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        formatted_date = date_obj.strftime("%d.%m.%Y")
    except ValueError as exc:
        raise DataFormatError("Неверный формат даты. Используйте YYYY-MM-DD.") from exc

    new_row = pd.DataFrame(
        {
            "date": [formatted_date],
            "avg_pullups": [avg_pullups],
            "total_pullups": [total_pullups],
        }
    )
    df = pd.concat([df, new_row], ignore_index=True)
    await save_data(df)


async def delete_training_data(idx: int):
    """Удалить данные о тренировке по индексу из базы данных."""
    df = await load_data()
    try:
        df.drop(idx, inplace=True)
        df.reset_index(drop=True, inplace=True)
    except IndexError:
        pass
    await save_data(df)


async def reset_training_data():
    """Сбросить все данные о тренировках в базе данных."""
    await reset_data()


async def update_training_data(idx: int, data: TrainingData):
    """Обновить данные о тренировке по индексу в базе данных."""
    current_data = await load_data()

    if idx < 0 or idx >= len(current_data):
        raise IndexError("Индекс за пределами диапазона")

    try:
        date_obj = datetime.strptime(data.date, "%Y-%m-%d")
        formatted_date = date_obj.strftime("%d.%m.%Y")
    except ValueError as exc:
        raise DataFormatError("Неверный формат даты. Используйте YYYY-MM-DD.") from exc

    current_data.loc[idx, "date"] = formatted_date
    current_data.loc[idx, "avg_pullups"] = data.avg_pullups
    current_data.loc[idx, "total_pullups"] = data.total_pullups

    await save_data(current_data)


async def get_all_training_data():
    """Получить все данные о тренировках из базы данных."""
    df = await load_data()
    return {
        "date": pd.to_datetime(df["date"], format="%d.%m.%Y")
        .dt.strftime("%Y-%m-%d")
        .tolist(),
        "avg_pullups": df["avg_pullups"].tolist(),
        "total_pullups": df["total_pullups"].tolist(),
    }
