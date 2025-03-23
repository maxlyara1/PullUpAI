from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional


class TrainingData(BaseModel):
    date: str
    avg_pullups: Optional[float] = None
    total_pullups: Optional[int] = None

    @model_validator(mode="after")
    def validate_pullups(self):
        avg = self.avg_pullups
        total = self.total_pullups

        if avg is None and total is None:
            raise ValueError("Необходимо указать либо среднее, либо сумму подтягиваний")

        if total is not None and avg is None:
            self.avg_pullups = total / 5

        if avg is not None and total is None:
            self.total_pullups = round(avg * 5)

        return self
