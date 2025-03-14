from typing import List, Callable, Any

from fastapi import APIRouter, HTTPException

from backend.models.training import TrainingData
from backend.services.training import (
    add_training_data,
    delete_training_data,
    reset_training_data,
    update_training_data,
    get_all_training_data,
)

router = APIRouter()


def handle_exceptions(func: Callable) -> Callable:
    """Декоратор для обработки исключений в эндпоинтах."""

    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    return wrapper


@router.get("/training", response_model=dict)
@handle_exceptions
async def get_training_data():
    """Получить все данные о тренировках."""
    return await get_all_training_data()


@router.post("/training")
@handle_exceptions
async def create_training_data(data: TrainingData):
    """Добавить новые данные о тренировке."""
    await add_training_data(data.date, data.avg_pullups, data.total_pullups)
    return {"message": "Данные успешно добавлены"}


@router.delete("/training/{idx}")
@handle_exceptions
async def remove_training_data(idx: int):
    """Удалить данные о тренировке по индексу."""
    await delete_training_data(idx)
    return {"message": "Данные успешно удалены"}


@router.put("/training/{idx}")
@handle_exceptions
async def modify_training_data(idx: int, data: TrainingData):
    """Обновить данные о тренировке по индексу."""
    await update_training_data(idx, data)
    return {"message": "Данные успешно обновлены"}


@router.delete("/training/reset")
@handle_exceptions
async def clear_training_data():
    """Сбросить все данные о тренировках."""
    await reset_training_data()
    return {"message": "Все данные успешно сброшены"}
