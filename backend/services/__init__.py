"""
Модуль сервисов для работы с данными о тренировках и прогнозирования.

Содержит функции для управления данными о тренировках подтягиваний,
а также создания прогнозов на основе накопленных данных.
"""

from backend.services.prediction import get_prediction_data
from backend.services.training import (
    add_training_data,
    delete_training_data,
    reset_training_data,
    update_training_data,
    get_all_training_data,
)

__all__ = [
    "get_prediction_data",
    "add_training_data",
    "delete_training_data",
    "reset_training_data",
    "update_training_data",
    "get_all_training_data",
]
