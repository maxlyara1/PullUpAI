"""
Модуль для прогнозирования результатов подтягиваний на основе исторических данных.

Включает функции для подготовки данных, обучения моделей и создания прогнозов 
с использованием методов машинного обучения.
"""
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.pipeline import Pipeline
from datetime import datetime, timedelta
import json
import logging
import joblib
from backend.database.operations import (
    load_data,
    load_initial_data,
    load_original_standards,
    DataError,
)
from backend.models.training import TrainingData
from functools import lru_cache

# Настройка логирования
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

# Константы для настройки модели
FORECAST_INTERVAL = 4
MIN_PULLUPS = 0
MAX_PULLUPS = 100
MAX_CACHE_SIZE = 50

# Названия колонок в данных
COL_DATE = "date"
COL_DAYS = "days"
COL_AVG_PULLUPS = "avg_pullups"
COL_LAG1 = "lag_avg_pullups_1"
COL_LAG2 = "lag_avg_pullups_2"
COL_GROWTH = "pullups_growth"

FEATURE_COLUMNS = [
    COL_DAYS,
    COL_LAG1,
    COL_LAG2,
    COL_GROWTH,
]

LOG_COLUMNS = [COL_DATE, COL_DAYS, COL_AVG_PULLUPS] + FEATURE_COLUMNS

IS_CACHE_ENABLED = False


class ModelCache:
    """
    Кэш для хранения обученных моделей.
    
    Предотвращает повторное обучение моделей для одинаковых входных данных.
    Автоматически удаляет наименее используемые модели при превышении максимального размера.
    """
    def __init__(self, max_size: int = MAX_CACHE_SIZE):
        """
        Инициализирует кэш моделей.
        
        Args:
            max_size: Максимальное количество моделей в кэше
        """
        self.cache = {}
        self.max_size = max_size
        self.access_times = {}

    def get(self, key: str) -> Optional[Pipeline]:
        """
        Получить модель из кэша по ключу.
        
        Args:
            key: Ключ для поиска модели
            
        Returns:
            Найденная модель или None, если модель не найдена
        """
        if not IS_CACHE_ENABLED:
            return None
            
        if key in self.cache:
            self._update_access_time(key)
            return self.cache[key]
        return None

    def set(self, key: str, model: Pipeline) -> None:
        """
        Добавить модель в кэш.
        
        Args:
            key: Ключ для сохранения модели
            model: Модель для сохранения
        """
        if not IS_CACHE_ENABLED:
            return
            
        if len(self.cache) >= self.max_size:
            self._remove_least_used()
        self.cache[key] = model
        self._update_access_time(key)

    def _update_access_time(self, key: str) -> None:
        """Обновить время последнего доступа к модели."""
        self.access_times[key] = datetime.now()

    def _remove_least_used(self) -> None:
        """Удалить наименее используемую модель из кэша."""
        if not self.cache:
            return
        oldest_key = min(self.access_times.items(), key=lambda x: x[1])[0]
        del self.cache[oldest_key]
        del self.access_times[oldest_key]
        
    def clear(self) -> None:
        """Очистить кэш моделей."""
        self.cache.clear()
        self.access_times.clear()


MODEL_CACHE = ModelCache()

# Загрузка предобученных моделей
try:
    k1_model = joblib.load("backend/k1_poly_model.pkl")
    poly = joblib.load("backend/k1_poly_features.pkl")
except FileNotFoundError:
    k1_model = None
    poly = None
    logger.warning("Модель k1_poly не найдена. Будет использовано базовое значение K1.")


def validate_training_data(df: pd.DataFrame) -> None:
    """
    Проверить корректность данных о тренировках.
    
    Args:
        df: DataFrame с данными о тренировках
        
    Raises:
        ValueError: В случае некорректных данных
    """
    if df[COL_AVG_PULLUPS].min() < MIN_PULLUPS:
        raise ValueError(f"Количество подтягиваний не может быть меньше {MIN_PULLUPS}")
    if df[COL_AVG_PULLUPS].max() > MAX_PULLUPS:
        raise ValueError(f"Количество подтягиваний не может быть больше {MAX_PULLUPS}")

    try:
        pd.to_datetime(df[COL_DATE])
    except:
        raise ValueError("Некорректный формат даты")


def prepare_features(df: pd.DataFrame, sort_by_column: str = COL_DATE) -> pd.DataFrame:
    if df.empty:
        return df

    validate_training_data(df)

    df = df.sort_values(by=sort_by_column)

    df[COL_LAG1] = df[COL_AVG_PULLUPS].shift(1)
    df[COL_LAG2] = df[COL_AVG_PULLUPS].shift(2)

    df["rolling_mean_3"] = df[COL_AVG_PULLUPS].rolling(window=3, min_periods=1).mean()

    df["growth_rate"] = df[COL_AVG_PULLUPS].pct_change()

    df["lag1_growth"] = df[COL_LAG1] * df["growth_rate"]

    if len(df) <= 2:
        current_avg = df[COL_AVG_PULLUPS].iloc[-1]

        df[COL_LAG1] = df[COL_LAG1].fillna(current_avg)
        df[COL_LAG2] = df[COL_LAG2].fillna(current_avg)

        df["growth_rate"] = df["growth_rate"].fillna(0)
        df["lag1_growth"] = df["lag1_growth"].fillna(0)
    else:
        df[COL_LAG1] = df[COL_LAG1].ffill().bfill()
        df[COL_LAG2] = df[COL_LAG2].ffill().bfill()

        df["growth_rate"] = df["growth_rate"].interpolate(
            method="linear", limit_direction="both"
        )
        df["lag1_growth"] = df["lag1_growth"].interpolate(
            method="linear", limit_direction="both"
        )

    df[COL_GROWTH] = df[COL_AVG_PULLUPS].diff()
    df[COL_GROWTH] = df[COL_GROWTH].interpolate(method="linear", limit_direction="both")

    for col in [COL_LAG1, COL_LAG2, "growth_rate", "lag1_growth", COL_GROWTH]:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    return df


def _get_cached_model_key(df_initial_json: str) -> str:
    timestamp = datetime.now().timestamp()
    return f"{df_initial_json}_{timestamp}"


def calculate_k1(max_pullups: int) -> float:
    """Рассчитать коэффициент K1 на основе максимального количества подтягиваний."""
    if k1_model is None:
        return 0.6
    min_pullups = 1
    max_pullups_limit = 50
    max_pullups = np.clip(max_pullups, min_pullups, max_pullups_limit)
    return k1_model.predict(poly.transform(np.array([[max_pullups]])))[0]


def max_from_avg(avg_pullups: float) -> int:
    """Определить максимальное количество подтягиваний по среднему значению."""
    if k1_model is None:
        return int(avg_pullups / 0.6)
    best_max_pullups = 1
    min_diff = float("inf")
    for max_p in range(1, 51):
        k1_val = calculate_k1(max_p)
        predicted_avg = k1_val * max_p
        diff = abs(predicted_avg - avg_pullups)
        if diff < min_diff:
            min_diff = diff
            best_max_pullups = max_p
    return best_max_pullups


def convert_to_json_serializable(obj: any) -> any:
    """Преобразовать объект в JSON-сериализуемый формат."""
    if isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, datetime):
        return obj.isoformat()
    else:
        return obj


WEIGHT_CATEGORY_MAPPING = {
    "до 52": 52,
    "до 56": 56,
    "до 60": 60,
    "до 67,5": 67.5,
    "до 75": 75,
    "до 82,5": 82.5,
    "до 90": 90,
    "до 100": 100,
    "до 110": 110,
    "до 125": 125,
    "до 140": 140,
    "140 +": 140,
}


def get_body_weight(weight_category: str) -> float:
    """Получить вес тела для категории."""
    return WEIGHT_CATEGORY_MAPPING.get(weight_category, 75)


def calculate_k2(weight_category: str, extra_weight: float = 24.0) -> float:
    """Рассчитать коэффициент K2 на основе весовой категории."""
    body_weight = get_body_weight(weight_category)
    return 1 + (extra_weight / body_weight)


def calculate_pullup_standards(
    original_standards_df: pd.DataFrame, weight_category: str
) -> dict:
    """Рассчитать стандарты подтягиваний для весовой категории."""
    filtered_df = original_standards_df[
        original_standards_df["weight_category"] == weight_category
    ]
    adapted_standards = {}
    for _, row in filtered_df.iterrows():
        rank = row["rank"]
        adapted_standards[rank] = row["max_pullups"]
    return adapted_standards


def process_dates(dates: pd.Series, start_date: datetime) -> list:
    """Обработать даты для регрессии."""
    return [(date - start_date).days for date in dates]


async def _load_all_data(weight_category: str) -> tuple:
    """Загрузить все необходимые данные."""
    try:
        df_2025_real = await load_data()
        df_initial = await load_initial_data()
        original_standards_df = load_original_standards()
        pullup_standards = calculate_pullup_standards(
            original_standards_df, weight_category
        )
        return df_2025_real, df_initial, pullup_standards
    except DataError as e:
        logger.error(f"Failed to load data: {e}")
        raise


def _prepare_data_for_regression(
    df_initial: pd.DataFrame, df_2025_real: pd.DataFrame
) -> tuple:
    """Подготовить данные для регрессии."""
    # Подготовка данных 2025
    df_2025_real[COL_DATE] = pd.to_datetime(df_2025_real[COL_DATE], format="%d.%m.%Y")

    # Если у пользователя уже есть данные, используем первую дату из этих данных как начало отсчета
    # Если данных нет, используем сегодняшний день как стартовую точку для прогноза
    if len(df_2025_real) > 0:
        start_date_2025 = df_2025_real[COL_DATE].iloc[0]
    else:
        start_date_2025 = datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0
        )

    predict_year = start_date_2025.year
    df_2025_real[COL_DAYS] = process_dates(df_2025_real[COL_DATE], start_date_2025)

    # Подготовка обучающих данных
    df_initial[COL_DATE] = pd.to_datetime(df_initial[COL_DATE], format="%d.%m.%Y")
    if len(df_initial) > 0:
        start_date_initial = df_initial[COL_DATE].iloc[0]
    else:
        start_date_initial = datetime(
            2021, 7, 12
        )  # Используем стандартную дату, если нет начальных данных

    df_initial[COL_DAYS] = process_dates(df_initial[COL_DATE], start_date_initial)
    initial_year = start_date_initial.year

    # Создаем признаки с помощью универсальной функции
    df_initial = prepare_features(df_initial)

    return df_initial, start_date_2025, initial_year, predict_year


def _build_regression_model(initial_data_processed: pd.DataFrame) -> Pipeline:
    """Построить модель регрессии с улучшенной стабильностью."""
    
    # Принудительно очищаем кэш моделей
    MODEL_CACHE.clear()

    # Сначала выполним сглаживание данных с помощью скользящего среднего
    # Уменьшаем окно сглаживания на этапе подготовки
    smoothed_data = initial_data_processed.copy()
    if len(smoothed_data) > 3:
        smoothed_data[COL_AVG_PULLUPS] = smoothed_data[COL_AVG_PULLUPS].rolling(
            window=min(2, len(smoothed_data)), 
            min_periods=1,
            center=True).mean()
    
    X_initial = smoothed_data[FEATURE_COLUMNS].to_numpy()
    y_initial = smoothed_data[COL_AVG_PULLUPS].to_numpy()

    # Используем более узкий диапазон значений alpha для более стабильной регуляризации
    # Уменьшаем степень регуляризации для более гибкой модели
    alphas = np.logspace(-2, 2, 20)  # Логарифмическая шкала от 0.01 до 100
    
    # Создаем улучшенный пайплайн с предобработкой и RidgeCV
    model = Pipeline(
        [
            ("scaler", StandardScaler()),  # Масштабирование признаков
            (
                "ridge_cv",
                RidgeCV(
                    alphas=alphas,
                    cv=min(5, len(smoothed_data)) if len(smoothed_data) > 3 else None,  # Используем CV только если достаточно данных
                    scoring='neg_mean_squared_error'  # Метрика для оценки качества
                ),
            ),
        ]
    )

    # Обучаем модель
    model.fit(X_initial, y_initial)

    # Получаем оптимальное значение alpha
    best_alpha = model.named_steps["ridge_cv"].alpha_

    # Создаем финальную модель с оптимальным alpha
    final_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=best_alpha)) 
        ]
    )
    final_model.fit(X_initial, y_initial)

    return final_model


def _generate_prediction_days(
    last_day_2025: int, forecast_days: int
) -> np.ndarray:
    """Генерирует массив дней для прогноза, начиная с последнего дня данных 2025 года.
    
    Args:
        last_day_2025: Последний день данных 2025 года
        forecast_days: Количество дней для прогноза
        
    Returns:
        Массив дней для прогноза
    """
    # Генерируем дни с шагом 1, чтобы получить непрерывный прогноз
    return np.arange(last_day_2025 + 1, last_day_2025 + forecast_days + 1, 1)


def _get_historical_data(
    df_2025_real: pd.DataFrame, df_initial: pd.DataFrame
) -> pd.DataFrame:
    """Получает исторические данные для прогноза."""
    # ИЗМЕНЕНО: Возвращаем только данные 2025 года, если их достаточно
    if len(df_2025_real) >= 2:
        historical_data = df_2025_real.copy()
    else:
        # Если данных 2025 года недостаточно, используем только исходные данные
        historical_data = df_initial.copy()

    # Сортируем данные по дням
    if not historical_data.empty and COL_DAYS in historical_data.columns:
        historical_data = historical_data.sort_values(by=COL_DAYS)
        
    return historical_data


def _prepare_initial_features(prepared_data: pd.DataFrame) -> tuple:
    """Подготавливает начальные значения признаков для прогноза."""
    # Получаем последние значения для начала прогноза
    last_values = (
        prepared_data.iloc[-1]
        if not prepared_data.empty
        else pd.Series(
            {
                COL_AVG_PULLUPS: 0,
                COL_LAG1: 0,
                COL_LAG2: 0,
                COL_GROWTH: 0,
            }
        )
    )

    last_pullups = [last_values[COL_LAG2], last_values[COL_LAG1]]
    last_growth = last_values[COL_GROWTH]

    return last_pullups, last_growth


def _predict_next_value(
    model: Pipeline,
    day: int,
    last_pullups: list,
    last_growth: float,
) -> tuple:
    """Предсказывает следующее значение и обновляет признаки."""
    # Создаем признаки в том же порядке, что и в FEATURE_COLUMNS
    features = [
        day,  # COL_DAYS
        last_pullups[-1],  # COL_LAG1
        last_pullups[-2],  # COL_LAG2
        last_growth,  # COL_GROWTH
    ]

    input_features = np.array([features])

    # Предсказание через пайплайн (включает масштабирование)
    predicted_avg = model.predict(input_features)[0]

    # Ограничиваем минимальное значение прогноза
    predicted_avg = max(predicted_avg, 1.0)

    # Обновляем значения для следующего шага
    new_last_pullups = last_pullups[1:] + [predicted_avg]
    new_growth = predicted_avg - last_pullups[-1]

    return predicted_avg, features, new_last_pullups, new_growth


def _make_predictions(
    model: Pipeline,
    historical_data: pd.DataFrame,
    predict_days: np.ndarray,
) -> np.ndarray:
    """Выполняет прогнозирование на основе исторических данных с улучшенной стабильностью."""
    
    # Создаем признаки на основе исторических данных
    prepared_data = prepare_features(historical_data)
    
    # Уменьшаем сглаживание исторических данных перед прогнозом
    if len(prepared_data) > 3:
        prepared_data[COL_AVG_PULLUPS] = prepared_data[COL_AVG_PULLUPS].rolling(
            window=min(2, len(prepared_data)), 
            min_periods=1, 
            center=True).mean()

    # Получаем начальные значения признаков
    last_pullups, last_growth = _prepare_initial_features(prepared_data)
    
    # Выполняем прогноз
    predicted_pullups = []
    inference_features = []
    
    # Увеличиваем максимальный прирост между последовательными предсказаниями
    max_growth_per_day = 0.35  # Увеличиваем максимальный прирост в день (было 0.25)
    
    prev_prediction = last_pullups[-1]  # Последнее известное значение

    for i, day in enumerate(predict_days.tolist()):
        predicted_avg, features, last_pullups, last_growth = _predict_next_value(
            model,
            day,
            last_pullups,
            last_growth,
        )
        
        # Применяем более мягкое сглаживание для стабилизации прогноза
        # Ограничиваем изменение предсказания относительно предыдущего
        days_since_last = 1 if i == 0 else predict_days[i] - predict_days[i-1]
        max_change = max_growth_per_day * days_since_last
        
        # Ограничиваем изменение предсказания с большим допуском
        if predicted_avg - prev_prediction > max_change:
            predicted_avg = prev_prediction + max_change
        elif prev_prediction - predicted_avg > max_change and predicted_avg < prev_prediction:
            predicted_avg = prev_prediction - max_change
            
        # Сохраняем сглаженное предсказание
        predicted_pullups.append(predicted_avg)
        inference_features.append(features)
        
        # Обновляем last_pullups и last_growth с учетом сглаженных значений
        last_pullups = last_pullups[1:] + [predicted_avg]
        last_growth = predicted_avg - prev_prediction
        
        # Обновляем предыдущее предсказание
        prev_prediction = predicted_avg

    # Уменьшаем финальное сглаживание для всего ряда предсказаний
    if len(predicted_pullups) > 5:
        smoothed_predictions = pd.Series(predicted_pullups).rolling(
            window=min(2, len(predicted_pullups)),  # Уменьшаем окно сглаживания до 2 (было 3)
            min_periods=1,
            center=True
        ).mean().to_numpy()
        
        # Гарантируем монотонность роста только если исходные данные показывают стабильный рост
        if last_pullups[0] < last_pullups[-1] and last_pullups[-1] - last_pullups[0] > 1.5:  # Снижаем порог с 2 до 1.5
            for i in range(1, len(smoothed_predictions)):
                if smoothed_predictions[i] < smoothed_predictions[i-1]:
                    # Уменьшаем силу коррекции монотонности еще больше
                    smoothed_predictions[i] = 0.9 * smoothed_predictions[i] + 0.1 * smoothed_predictions[i-1]  # Было 0.8/0.2
        
        return smoothed_predictions
    
    return np.array(predicted_pullups)


def _forecast(
    model: Optional[Pipeline],
    df_2025_real: pd.DataFrame,
    df_initial: pd.DataFrame,
    forecast_days: int,
) -> tuple:
    """Выполняет полный процесс прогнозирования."""
    # Создаем копии данных, чтобы избежать проблем с кэшированием pandas
    df_2025_real = df_2025_real.copy(deep=True)
    df_initial = df_initial.copy(deep=True)
    
    # Всегда используем исторические данные для обучения модели
    model_source = "historical"  # Всегда используем исторические данные
    
    # Проверяем, передана ли модель или нужно создать новую
    if model is None:
        logger.info("Модель не предоставлена, создаем новую модель регрессии на исторических данных")
        
        # ИЗМЕНЕНО: Всегда строим модель на основе исторических данных,
        # даже если есть достаточно данных 2025 года
        if len(df_initial) > 0:
            model = _build_regression_model(df_initial)
            logger.info("Модель регрессии построена на основе исторических данных")
        else:
            logger.warning("Исторические данные отсутствуют, невозможно построить модель.")
            # Возвращаем пустые массивы в случае ошибки
            return np.array([]), np.array([]), model_source
    
    # Генерируем дни для прогноза
    last_day_2025 = 0
    if not df_2025_real.empty:
        # Если есть данные 2025, берем последний день как точку отсчета для прогноза
        if COL_DAYS not in df_2025_real.columns:
            # Если колонка COL_DAYS отсутствует, добавляем ее
            if not pd.api.types.is_datetime64_any_dtype(df_2025_real[COL_DATE]):
                df_2025_real[COL_DATE] = pd.to_datetime(df_2025_real[COL_DATE], format="%d.%m.%Y")
            
            start_date = df_2025_real[COL_DATE].iloc[0]
            df_2025_real[COL_DAYS] = [(date - start_date).days for date in df_2025_real[COL_DATE]]
            logger.info(f"Добавлен признак дней для данных 2025 года")
        
        last_day_2025 = df_2025_real[COL_DAYS].max()
        logger.info(f"Последний день данных 2025: {last_day_2025}")
    
    # Используем обновленную функцию для генерации дней прогноза
    predict_days = _generate_prediction_days(last_day_2025, forecast_days)
    logger.info(f"Созданы дни для прогноза от {predict_days[0]} до {predict_days[-1]}, всего {len(predict_days)} дней")

    # КРИТИЧЕСКИЙ МОМЕНТ: Подготавливаем только данные 2025 года для прогнозирования
    # Мы должны использовать только текущие данные пользователя для начальной точки прогноза
    if len(df_2025_real) >= 2:
        logger.info(f"Подготавливаем данные 2025 года ({len(df_2025_real)} записей) для прогнозирования")
        
        # Преобразуем данные 2025, чтобы они соответствовали формату для прогнозирования
        if not pd.api.types.is_datetime64_any_dtype(df_2025_real[COL_DATE]):
            df_2025_real[COL_DATE] = pd.to_datetime(df_2025_real[COL_DATE], format="%d.%m.%Y")
            
        # Убедимся, что у нас есть признак дней
        if COL_DAYS not in df_2025_real.columns:
            start_date = df_2025_real[COL_DATE].iloc[0]
            df_2025_real[COL_DAYS] = [(date - start_date).days for date in df_2025_real[COL_DATE]]
            logger.info(f"Добавлен признак дней для данных 2025 года")
        
        # Подготавливаем признаки для прогнозирования
        historical_data = prepare_features(df_2025_real)
        logger.info(f"Данные 2025 года подготовлены: {len(historical_data)} записей")
        
        # Логирование последней точки данных 2025 года - это будет началом прогноза
        if not historical_data.empty:
            last_row = historical_data.iloc[-1]
            logger.info(f"Последняя точка 2025 года: день={last_row[COL_DAYS]}, подтягивания={last_row[COL_AVG_PULLUPS]}")
    else:
        # Если данных 2025 года недостаточно, используем только начальные данные для прогноза
        logger.info(f"Недостаточно данных 2025 года, используем только начальные данные")
        historical_data = df_initial.copy()
        # Добавляем обработку признаков для исторических данных
        historical_data = prepare_features(historical_data)
        logger.info(f"Подготовлены только исторические данные: {len(historical_data)} записей")

    logger.info(f"historical_data: {historical_data}")
    # Выполняем прогнозирование с обновленным порядком аргументов
    predicted_pullups = _make_predictions(model, historical_data, predict_days)
    logger.info(f"Выполнен прогноз на {len(predicted_pullups)} точек")
    
    return predict_days, predicted_pullups, model_source


def _calculate_achievement_dates(
    predict_days: np.ndarray,
    predicted_avg_pullups: np.ndarray,
    pullup_standards: dict,
    weight_category: str,
    actual_data: pd.DataFrame = None,  # Добавляем параметр для фактических данных
) -> dict:
    """Рассчитать даты достижения стандартов."""
    achievement_dates = {}

    # Вычисляем k2 один раз
    k2 = calculate_k2(weight_category)

    # Предварительно вычисляем все максимальные значения
    max_pullups_no_weight_predicted = np.array(
        [max_from_avg(avg) for avg in predicted_avg_pullups]
    )
    max_pullups_with_weight_predicted = np.round(
        max_pullups_no_weight_predicted / k2
    ).astype(int)

    # Проверяем фактические достижения
    actual_max_achieved = 0
    if actual_data is not None and not actual_data.empty:
        # Вычисляем максимальное значение один раз
        actual_max_values = np.array(
            [max_from_avg(avg) for avg in actual_data[COL_AVG_PULLUPS]]
        )
        actual_max_with_weight = np.round(actual_max_values / k2).astype(int)
        actual_max_achieved = (
            np.max(actual_max_with_weight) if len(actual_max_with_weight) > 0 else 0
        )

    # Предварительно вычисляем параметры для экстраполяции
    if len(max_pullups_with_weight_predicted) > 1:
        last_value = max_pullups_with_weight_predicted[-1]
        first_value = max_pullups_with_weight_predicted[0]
        days_total = predict_days[-1] - predict_days[0]
        progress_per_day = (
            (last_value - first_value) / days_total if days_total > 0 else 0
        )
    else:
        progress_per_day = 0

    # Обрабатываем каждый разряд
    for rank, max_pullups_with_weight_standard in pullup_standards.items():
        # Если разряд уже достигнут по фактическим данным
        if actual_max_achieved >= max_pullups_with_weight_standard:
            achievement_dates[rank] = 0  # 0 означает "уже достигнуто"
            continue

        # Ищем индекс первого дня, когда достигается стандарт
        # Используем numpy для более эффективного поиска
        indices = np.where(
            max_pullups_with_weight_predicted >= max_pullups_with_weight_standard
        )[0]

        if len(indices) > 0:
            # Найден день достижения стандарта
            first_index = indices[0]
            # Используем абсолютное значение дня для предотвращения зависимости от forecast_days
            achievement_dates[rank] = int(predict_days[first_index])
        elif progress_per_day > 0:
            # Экстраполируем, если есть положительный прогресс
            days_to_target = (
                max_pullups_with_weight_standard - last_value
            ) / progress_per_day
            # Используем абсолютное значение при экстраполяции
            # Это значение не должно зависеть от forecast_days
            achievement_dates[rank] = int(predict_days[-1] + days_to_target)
        else:
            # Стандарт не будет достигнут в прогнозируемом периоде
            achievement_dates[rank] = None

    # Логируем рассчитанные даты достижения для отладки
    logger.debug(f"Рассчитанные даты достижения разрядов: {achievement_dates}")
    
    return achievement_dates


def _create_chart2_data(
    predict_days: np.ndarray,
    predicted_avg_pullups: np.ndarray,
    df_2025_real: pd.DataFrame,
    pullup_standards: dict,
    achievement_dates: dict,
    start_date_2025: datetime,
    predict_year: int,
    forecast_days: int,
    weight_category: str,
) -> dict:
    """Создать данные для второго графика."""

    # Проверяем наличие данных пользователя
    if df_2025_real.empty or len(df_2025_real) == 1:
        # Если данных нет или есть только одно наблюдение, возвращаем специальное сообщение вместо графика
        return {
            "data": [],
            "standards": [],
            "title": "Загрузите данные для построения прогноза",
            "xAxisLabel": "Дата",
            "yAxisLabel": "Среднее количество подтягиваний",
            "noUserData": True,
            "message": "Необходимо загрузить данные о ваших тренировках для построения персонального прогноза. Для построения прогноза требуется минимум два наблюдения.",
            "model_source": "historical",  # Добавляем информацию о модели
        }

    # Подготовка данных для графика
    max_pullups_no_weight_predicted = [
        max_from_avg(avg) for avg in predicted_avg_pullups
    ]
    k2 = calculate_k2(weight_category)
    max_pullups_with_weight_predicted = [
        round(max_p / k2) for max_p in max_pullups_no_weight_predicted
    ]

    # Создаем основные данные графика
    chart_data = []

    # Сначала добавляем фактические данные
    if not df_2025_real.empty:
        for i, row in df_2025_real.iterrows():
            date = row[COL_DATE].strftime("%Y-%m-%d")
            chart_data.append(
                {
                    "date": date,
                    "day": row[COL_DAYS],
                    "actual": row[COL_AVG_PULLUPS],
                    "average": None,
                    "maximum": None,
                    "withWeight": None,
                }
            )

    # Добавляем прогнозные данные
    added_prediction_points = 0
    for i, day in enumerate(predict_days.tolist()):
        date = (start_date_2025 + timedelta(days=int(day))).strftime("%Y-%m-%d")
        # Проверяем, нет ли уже точки с такой датой
        existing_point = next(
            (point for point in chart_data if point["date"] == date), None
        )
        if existing_point:
            # Если точка существует, добавляем к ней прогнозные значения
            existing_point["average"] = round(predicted_avg_pullups[i], 1)
            existing_point["maximum"] = max_pullups_no_weight_predicted[i]
            existing_point["withWeight"] = max_pullups_with_weight_predicted[i]
            added_prediction_points += 1
        else:
            # Если точек нет, создаем новую
            chart_data.append(
                {
                    "date": date,
                    "day": day,
                    "actual": None,
                    "average": round(predicted_avg_pullups[i], 1),
                    "maximum": max_pullups_no_weight_predicted[i],
                    "withWeight": max_pullups_with_weight_predicted[i],
                }
            )
            added_prediction_points += 1
    
    # Сортируем данные по дням
    chart_data.sort(key=lambda x: x["day"])

    # Подготовка данных о нормативах
    standards_data = []
    for rank, pullups in pullup_standards.items():
        achievement_day = achievement_dates.get(rank)
        if (
            achievement_day is not None
            and achievement_day != "Не достигнуто в прогнозе"
        ):
            # Для уже достигнутых разрядов (achievement_day == 0)
            if achievement_day == 0:
                achievement_date = start_date_2025.strftime("%Y-%m-%d")
            else:
                achievement_date = (
                    start_date_2025 + timedelta(days=achievement_day)
                ).strftime("%Y-%m-%d")

            standards_data.append(
                {
                    "rank": rank,
                    "value": pullups,
                    "achievementDate": achievement_date,
                }
            )

    # Определяем заголовок на основе наличия фактических данных
    title = (
        f"Прогноз и фактические тренировки в {predict_year} году"
        if not df_2025_real.empty
        else f"Прогноз тренировок на {predict_year} год"
    )

    return {
        "data": chart_data,
        "standards": standards_data,
        "title": title,
        "xAxisLabel": "Дата",
        "yAxisLabel": "Среднее количество подтягиваний",
        "noUserData": False,
        "model_source": "historical",  # Явно указываем, что модель историческая
    }


async def get_prediction_data(
    weight_category: str = "до 75", forecast_days: int = 90
) -> dict:
    """Получить данные для прогноза."""
    try:
        logger.info(f"=== НАЧАЛО ПРОГНОЗИРОВАНИЯ ===")
        logger.info(f"Запрос прогноза для весовой категории: {weight_category}, на {forecast_days} дней")
        
        # Очищаем кэш моделей перед каждым запросом,
        # чтобы гарантировать использование свежих данных
        MODEL_CACHE.clear()
        logger.info(f"Кэш моделей очищен")
        
        # Всегда загружаем свежие данные из файлов
        logger.info(f"Загрузка данных из файлов...")
        (
            df_2025_real,
            df_initial,
            pullup_standards,
        ) = await _load_all_data(weight_category)
        
        logger.info(f"Данные загружены: {len(df_2025_real)} записей 2025, {len(df_initial)} записей начальных")
        
        # Принудительно сбрасываем кэш для используемых данных
        df_2025_real = df_2025_real.copy(deep=True)
        df_initial = df_initial.copy(deep=True)
        logger.info(f"Данные скопированы для предотвращения проблем с кэшированием")

        # Подготовим данные для таблицы независимо от проверки на количество наблюдений
        data_2025_prepared = []
        for i in range(len(df_2025_real)):
            data_2025_prepared.append(
                TrainingData(
                    date=pd.to_datetime(df_2025_real[COL_DATE].iloc[i]).strftime("%Y-%m-%d"),
                    avg_pullups=df_2025_real[COL_AVG_PULLUPS].iloc[i],
                )
            )

        if df_2025_real.empty or len(df_2025_real) == 1:
            # Если данных пользователя нет или есть только одно наблюдение, возвращаем сообщение без выполнения прогноза
            logger.info(
                "Данные пользователя отсутствуют или недостаточны для построения прогноза."
            )

            # Текущая дата для сообщения
            current_date = datetime.now().replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            predict_year = current_date.year

            # Создаем сообщение для графика без выполнения прогноза
            chart2_data = {
                "data": [],
                "standards": [],
                "title": "Загрузите данные для построения прогноза",
                "xAxisLabel": "Дата",
                "yAxisLabel": "Среднее количество подтягиваний",
                "noUserData": True,
                "message": "Необходимо загрузить данные о ваших тренировках для построения персонального прогноза. Для построения прогноза требуется минимум два наблюдения.",
                "model_source": "historical",  # Явно указываем источник модели
            }

            # Сохраняем метку времени для предотвращения кэширования
            timestamp = datetime.now().timestamp()
            
            # Формируем результат
            result = {
                "data_2025": data_2025_prepared,
                "chart2": json.dumps(convert_to_json_serializable(chart2_data)),
                "achievement_dates": {},
                "pullup_standards": pullup_standards,
                "forecast_days": forecast_days,
                "initial_year": None,
                "predict_year": predict_year,
                "progress_type": "Недостаточно данных",
                "growth_per_day": 0.0,
                "timestamp": timestamp,  # Добавляем временную метку для предотвращения кэширования
                "model_source": "historical",  # Явно указываем, что модель должна быть основана на исторических данных
            }
            
            # Логируем результаты для отладки
            logger.info(f"Прогноз завершен успешно. Timestamp: {timestamp}")
            logger.info(f"Отправка результата с {len(data_2025_prepared)} записями данных")
            if chart2_data and "data" in chart2_data:
                logger.info(f"График содержит {len(chart2_data['data'])} точек данных")
            logger.info(f"=== КОНЕЦ ПРОГНОЗИРОВАНИЯ ===")
            
            return result

        # Подготавливаем данные и получаем необходимую информацию
        df_initial, start_date_2025, initial_year, predict_year = _prepare_data_for_regression(df_initial, df_2025_real)
        
        # ИЗМЕНЕНИЕ: Не строим модель здесь, а передаем пустую модель в _forecast
        # Там уже будет принято решение на основе каких данных строить модель
        # это гарантирует, что если есть данные 2025, они будут использованы для модели
        empty_model = None  # Пустая модель для передачи в _forecast
        
        # Вызываем _forecast - он сам решит, на каких данных построить модель
        predict_days, predicted_avg_pullups, model_source = _forecast(
            empty_model, df_2025_real, df_initial, forecast_days
        )
        
        achievement_dates = _calculate_achievement_dates(
            predict_days,
            predicted_avg_pullups,
            pullup_standards,
            weight_category,
            df_2025_real,
        )
        chart2_data = _create_chart2_data(
            predict_days,
            predicted_avg_pullups,
            df_2025_real,
            pullup_standards,
            achievement_dates,
            start_date_2025,
            predict_year,
            forecast_days,
            weight_category,
        )

        # Определяем тип прогресса на основе прироста
        progress_type = "Умеренный"
        growth_per_day = 0.0

        if len(predict_days) > 1:
            days_diff = predict_days[-1] - predict_days[0]
            if days_diff > 0:
                avg_diff = predicted_avg_pullups[-1] - predicted_avg_pullups[0]
                growth_per_day = avg_diff / days_diff
                
                if growth_per_day > 0.20:
                    progress_type = "Быстрый"
                elif growth_per_day > 0.10:
                    progress_type = "Умеренный"
                elif growth_per_day > 0.05:
                    progress_type = "Медленный"
                else:
                    progress_type = "Очень медленный"

        # Сохраняем метку времени для предотвращения кэширования
        timestamp = datetime.now().timestamp()
        
        # Формируем результат
        result = {
            "data_2025": data_2025_prepared,
            "chart2": json.dumps(convert_to_json_serializable(chart2_data)),
            "achievement_dates": achievement_dates,
            "pullup_standards": pullup_standards,
            "forecast_days": forecast_days,
            "initial_year": initial_year,
            "predict_year": predict_year,
            "progress_type": progress_type,
            "growth_per_day": growth_per_day,
            "timestamp": timestamp,  # Добавляем временную метку для предотвращения кэширования
            "model_source": model_source,  # Передаем информацию об источнике модели
        }
        
        # Логируем результаты для отладки
        logger.info(f"Прогноз завершен успешно. Timestamp: {timestamp}")
        logger.info(f"Отправка результата с {len(data_2025_prepared)} записями данных")
        if chart2_data and "data" in chart2_data:
            logger.info(f"График содержит {len(chart2_data['data'])} точек данных")
        logger.info(f"Источник модели: {model_source}")  # Логируем источник модели
        logger.info(f"=== КОНЕЦ ПРОГНОЗИРОВАНИЯ ===")
        
        return result

    except DataError as e:
        logger.error(f"Error preparing prediction data: {e}")
        raise
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        raise
