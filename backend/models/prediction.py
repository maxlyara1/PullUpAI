"""
Модели для данных прогноза.
"""

from pydantic import BaseModel
from typing import List, Dict, Optional, Union


class PredictionData(BaseModel):
    """Модель данных для прогноза."""

    data_2025: List[dict]
    chart1: str
    chart2: str
    mae_2021: float
    r2_2021: float
    mae_2025: Optional[float]
    r2_2025: Optional[float]
    forecast_improvement: Optional[float]
    selected_degree: int
    achievement_dates: Dict[str, Union[int, str]]
    pullup_standards: Dict[str, int]
    forecast_days: int
    initial_year: int
    predict_year: int
