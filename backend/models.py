# backend/models.py
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime
from typing import List, Optional, Dict, Any  #  Добавили Any


class TrainingData(BaseModel):
    date: str = Field(..., example="2025-01-14")
    avg_pullups: Optional[float] = Field(
        None, ge=0, example=5.2
    )  # Сделали необязательным
    total_pullups: Optional[int] = Field(
        None, ge=0, example=26
    )  # Добавили, сделали необязательным

    @field_validator("date")
    @classmethod
    def parse_date(cls, value):
        try:
            datetime.strptime(value, "%Y-%m-%d")  # Проверяем формат даты
            return value
        except ValueError:
            raise ValueError("Invalid date format. Use YYYY-MM-DD.")

    @model_validator(mode="after")
    def check_pullups(self):
        if self.avg_pullups is None and self.total_pullups is None:
            raise ValueError(
                "Either 'avg_pullups' or 'total_pullups' must be provided."
            )
        if self.avg_pullups is not None and self.total_pullups is not None:
            raise ValueError(
                "Only one of 'avg_pullups' or 'total_pullups' should be provided."
            )
        return self

    class Config:
        json_schema_extra = {  # Для доков
            "example": {"date": "2025-01-14", "avg_pullups": 5.2}
        }


class PredictionData(BaseModel):
    data_2025: List[TrainingData]
    chart1: str  #  Оставляем str, так как теперь это JSON строка
    chart2: str  #  Оставляем str
    mae_2021: str
    r2_2021: str
    mae_2025: str
    r2_2025: str
    forecast_improvement: str
    selected_degree: int
    achievement_dates: dict
    pullup_standards: dict
    forecast_days: int
