import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.pipeline import Pipeline
from datetime import datetime, timedelta
import json
import logging
import joblib
from backend.database.operations import (
    load_data,
    save_data,
    load_initial_data,
    reset_data,
    load_original_standards,
    DataError,
    DataFormatError,
)
from backend.models.training import TrainingData
from functools import lru_cache

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

# Константы
FORECAST_INTERVAL = 4

# Названия колонок
COL_DATE = "date"
COL_DAYS = "days"
COL_AVG_PULLUPS = "avg_pullups"
COL_LAG1 = "lag_avg_pullups_1"
COL_LAG2 = "lag_avg_pullups_2"
COL_GROWTH = "pullups_growth"

# Список признаков для обучения и предсказания
FEATURE_COLUMNS = [
    COL_DAYS,
    COL_LAG1,
    COL_LAG2,
    COL_GROWTH,
]

# Список колонок для отображения в логах
LOG_COLUMNS = [COL_DATE, COL_DAYS, COL_AVG_PULLUPS] + FEATURE_COLUMNS

# Кэш для моделей регрессии
MODEL_CACHE = {}

try:
    k1_model = joblib.load("backend/k1_poly_model.pkl")
    poly = joblib.load("backend/k1_poly_features.pkl")
except FileNotFoundError:
    k1_model = None
    poly = None
    print("Модель k1_poly не найдена. Будет использовано базовое значение K1.")


def prepare_features(df: pd.DataFrame, sort_by_column: str = COL_DATE) -> pd.DataFrame:
    """
    Создает признаки для модели: лаги и метрики роста.
    
    Args:
        df: DataFrame с данными
        sort_by_column: колонка для сортировки данных
        
    Returns:
        DataFrame с добавленными признаками
    """
    if df.empty:
        return df
    
    # Сортируем данные
    df = df.sort_values(by=sort_by_column)
    
    # Создаем лаги значений подтягиваний
    df[COL_LAG1] = df[COL_AVG_PULLUPS].shift(1)
    df[COL_LAG2] = df[COL_AVG_PULLUPS].shift(2)
    
    # Проверяем, есть ли только одна строка данных или слишком мало данных
    if len(df) <= 2:
        # Заполняем лаги значением текущего среднего количества подтягиваний,
        # чтобы избежать NaN значений при коротких последовательностях
        current_avg = df[COL_AVG_PULLUPS].iloc[-1]
        df[COL_LAG1] = df[COL_LAG1].fillna(current_avg)
        df[COL_LAG2] = df[COL_LAG2].fillna(current_avg)
    
    # Заполняем оставшиеся пропуски в лагах: сначала последними доступными значениями (ffill),
    # потом, если остались пропуски в начале, заполняем следующими (bfill)
    df[COL_LAG1] = df[COL_LAG1].ffill().bfill()
    df[COL_LAG2] = df[COL_LAG2].ffill().bfill()
    
    # Абсолютное изменение: текущее значение - предыдущее
    df[COL_GROWTH] = df[COL_AVG_PULLUPS].diff().fillna(0)
    
    return df


@lru_cache(maxsize=32)
def _get_cached_model_key(df_initial_json: str) -> str:
    """Получить ключ для кэша модели."""
    return f"{df_initial_json}"


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
        logger.info(f"Используется первая дата из данных пользователя как начальная точка прогноза: {start_date_2025.strftime('%d.%m.%Y')}")
    else:
        start_date_2025 = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        logger.info(f"Данные пользователя отсутствуют. Используется текущая дата как начальная точка прогноза: {start_date_2025.strftime('%d.%m.%Y')}")
    
    predict_year = start_date_2025.year
    df_2025_real[COL_DAYS] = process_dates(df_2025_real[COL_DATE], start_date_2025)

    # Подготовка обучающих данных
    df_initial[COL_DATE] = pd.to_datetime(df_initial[COL_DATE], format="%d.%m.%Y")
    if len(df_initial) > 0:
        start_date_initial = df_initial[COL_DATE].iloc[0]
        logger.info(f"Найдены начальные данные с первой датой: {start_date_initial.strftime('%d.%m.%Y')}")
    else:
        start_date_initial = datetime(2021, 7, 12) # Используем стандартную дату, если нет начальных данных
        logger.info(f"Начальные данные отсутствуют. Используется стандартная дата: {start_date_initial.strftime('%d.%m.%Y')}")
    
    df_initial[COL_DAYS] = process_dates(df_initial[COL_DATE], start_date_initial)
    initial_year = start_date_initial.year

    # Создаем признаки с помощью универсальной функции
    df_initial = prepare_features(df_initial)

    logger.info(f"Подготовлены данные для регрессии. Количество записей: {len(df_initial)}")

    return df_initial, start_date_2025, initial_year, predict_year


def _build_regression_model(initial_data_processed: pd.DataFrame) -> tuple:
    """Построить модель регрессии."""
    # Создаем ключ для кэша на основе данных и степени полинома
    df_json = initial_data_processed.to_json()
    cache_key = _get_cached_model_key(df_json)

    # Проверяем, есть ли модель в кэше
    if cache_key in MODEL_CACHE:
        logger.info("Используется закешированная модель регрессии")
        return MODEL_CACHE[cache_key]

    logger.info("Создается новая модель регрессии")
    
    X_initial = initial_data_processed[FEATURE_COLUMNS].to_numpy()
    y_initial = initial_data_processed[COL_AVG_PULLUPS].to_numpy()
    
    # Создаем пайплайн с предобработкой и RidgeCV для подбора оптимального alpha
    alphas = np.arange(0.01, 1000.01, 0.1)  # От 0.01 до 1000 с шагом 0.1
    model = Pipeline([
        ('scaler', StandardScaler()),  # Стандартизация признаков
        ('ridge_cv', RidgeCV(alphas=alphas, store_cv_values=True))  # RidgeCV с кросс-валидацией
    ])
    
    # Обучаем модель
    model.fit(X_initial, y_initial)
    
    # Получаем оптимальное значение alpha
    best_alpha = model.named_steps['ridge_cv'].alpha_
    logger.info(f"Найдено оптимальное значение alpha: {best_alpha:.3f}")
    
    # Создаем финальную модель с оптимальным alpha
    final_model = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=best_alpha))
    ])
    final_model.fit(X_initial, y_initial)
    
    logger.info("Модель регрессии обучена успешно")

    # Сохраняем модель в кэш
    MODEL_CACHE[cache_key] = final_model
    return final_model


def _generate_prediction_days(forecast_days: int) -> np.ndarray:
    """Генерирует массив дней для прогноза."""
    return np.arange(0, forecast_days, FORECAST_INTERVAL)


def _get_historical_data(df_2025_real: pd.DataFrame, df_initial: pd.DataFrame) -> pd.DataFrame:
    """Получает исторические данные для прогноза."""
    historical_data = pd.DataFrame()
    
    if not df_2025_real.empty:
        # Используем реальные данные 2025 года
        historical_data = df_2025_real.copy()

        
    # Сортируем данные по дням
    if not historical_data.empty:
        historical_data = historical_data.sort_values(by="days")
        
    return historical_data


def _prepare_initial_features(prepared_data: pd.DataFrame) -> tuple:
    """Подготавливает начальные значения признаков для прогноза."""
    # Получаем последние значения для начала прогноза
    last_values = prepared_data.iloc[-1] if not prepared_data.empty else pd.Series({
        COL_AVG_PULLUPS: 0,
        COL_LAG1: 0,
        COL_LAG2: 0,
        COL_GROWTH: 0,
    })
    
    current_avg = last_values[COL_AVG_PULLUPS]
    last_pullups = [
        last_values[COL_LAG2], 
        last_values[COL_LAG1]
    ]
    last_growth = last_values[COL_GROWTH]
    
    return current_avg, last_pullups, last_growth


def _predict_next_value(
    model: Pipeline,
    day: int,
    last_pullups: list,
    last_growth: float,
) -> tuple:
    """Предсказывает следующее значение и обновляет признаки."""
    # Создаем признаки в том же порядке, что и в FEATURE_COLUMNS
    features = [
        day,                # COL_DAYS
        last_pullups[-1],   # COL_LAG1
        last_pullups[-2],   # COL_LAG2
        last_growth,        # COL_GROWTH
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


def _determine_progress_factor(df_real: pd.DataFrame) -> tuple:
    """
    Анализирует имеющиеся данные пользователя и определяет коэффициент
    коррекции прогноза в зависимости от темпа прогресса.
    
    Args:
        df_real: DataFrame с реальными данными пользователя
        
    Returns:
        tuple: (коэффициент коррекции, описание прогресса, рост в день)
    """
    # Если нет данных или всего одна запись - используем стандартный прогноз
    if df_real.empty or len(df_real) < 2:
        logger.info("Недостаточно данных для анализа темпа прогресса. Используется стандартный прогноз.")
        return 1.0, "Недостаточно данных", 0.0
    
    # Сортируем данные по дате
    df_sorted = df_real.sort_values(by=COL_DATE)
    
    # Вычисляем скорость прогресса (среднее изменение за день)
    first_avg = df_sorted[COL_AVG_PULLUPS].iloc[0]
    last_avg = df_sorted[COL_AVG_PULLUPS].iloc[-1]
    
    # Вычисляем количество дней между первой и последней тренировкой
    first_date = pd.to_datetime(df_sorted[COL_DATE].iloc[0])
    last_date = pd.to_datetime(df_sorted[COL_DATE].iloc[-1])
    days_passed = max((last_date - first_date).days, 1)  # Избегаем деления на ноль
    
    # Считаем среднедневной прирост
    growth_per_day = (last_avg - first_avg) / days_passed
    
    # Определяем коэффициент коррекции на основе скорости прогресса
    # Эти пороговые значения можно настроить на основе анализа данных
    if growth_per_day > 0.07:  # Быстрый прогресс (как у автора)
        factor = 1.0
        progress_type = "Быстрый"
        logger.info(f"Обнаружен быстрый темп прогресса ({growth_per_day:.3f}/день). Прогноз без коррекции.")
    elif growth_per_day > 0.05:  # Средний прогресс
        factor = 1.25
        progress_type = "Средний"
        logger.info(f"Обнаружен средний темп прогресса ({growth_per_day:.3f}/день). Прогноз скорректирован (x{factor}).")
    elif growth_per_day > 0.03:  # Медленный прогресс
        factor = 1.5
        progress_type = "Медленный"
        logger.info(f"Обнаружен медленный темп прогресса ({growth_per_day:.3f}/день). Прогноз скорректирован (x{factor}).")
    else:  # Очень медленный прогресс
        factor = 2.0
        progress_type = "Очень медленный"
        logger.info(f"Обнаружен очень медленный темп прогресса ({growth_per_day:.3f}/день). Прогноз скорректирован (x{factor}).")
    
    return factor, progress_type, round(growth_per_day, 3)


def _make_predictions(
    model: Pipeline,
    predict_days: np.ndarray,
    historical_data: pd.DataFrame,
    progress_factor: float = 1.0
) -> tuple:
    """Выполняет прогнозирование."""
    # Создаем признаки на основе исторических данных
    prepared_data = prepare_features(historical_data)
    
    logger.info(f"Начало прогнозирования на {len(predict_days)} дней с коэффициентом коррекции {progress_factor}")
    
    # Получаем начальные значения признаков
    current_avg, last_pullups, last_growth = _prepare_initial_features(prepared_data)
    
    # Выполняем прогноз
    predicted_pullups = []
    inference_features = []
    
    # Базовое значение для первого прогноза
    base_value = current_avg if not prepared_data.empty else 1.0
    
    # Прогрессия времени для корректировки
    time_correction = np.zeros(len(predict_days))
    
    for i, day in enumerate(predict_days):
        predicted_avg, features, last_pullups, last_growth = _predict_next_value(
            model,
            day,
            last_pullups,
            last_growth,
        )
        
        # Применяем коррекцию на основе прогресса пользователя
        # Чем дальше в будущее, тем сильнее коррекция
        if i > 0:  # Не корректируем первое значение
            # Рассчитываем временную коррекцию (усиливается со временем)
            time_factor = min(1.0, i / (len(predict_days) * 0.5))
            
            # Применяем корректирующий множитель
            correction = 1.0 + ((progress_factor - 1.0) * time_factor)
            
            # Разница между текущим прогнозом и предыдущим значением
            progress_diff = predicted_avg - predicted_pullups[-1]
            
            # Корректируем только прирост, а не все значение
            corrected_diff = progress_diff / correction
            
            # Финальное скорректированное значение
            predicted_avg = predicted_pullups[-1] + corrected_diff
        
        predicted_pullups.append(predicted_avg)
        inference_features.append(features)
    
    logger.info(f"Прогноз выполнен. Количество точек прогноза: {len(predicted_pullups)}")
    return np.array(predicted_pullups)


def _forecast(
    model: Pipeline,
    df_2025_real: pd.DataFrame,
    df_initial: pd.DataFrame,
    forecast_days: int,
) -> tuple:
    """Выполняет полный процесс прогнозирования."""
    # Генерируем дни для прогноза
    predict_days = _generate_prediction_days(forecast_days)
    
    # Получаем исторические данные
    historical_data = _get_historical_data(df_2025_real, df_initial)
    
    # Определяем коэффициент коррекции на основе темпа прогресса пользователя
    progress_factor, progress_type, growth_per_day = _determine_progress_factor(df_2025_real)
    
    # Выполняем прогнозирование с учетом коэффициента коррекции
    predicted_pullups = _make_predictions(model, predict_days, historical_data, progress_factor)
    
    return predict_days, predicted_pullups, progress_type, growth_per_day


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
    max_pullups_no_weight_predicted = np.array([max_from_avg(avg) for avg in predicted_avg_pullups])
    max_pullups_with_weight_predicted = np.round(max_pullups_no_weight_predicted / k2).astype(int)
    
    # Проверяем фактические достижения
    actual_max_achieved = 0
    if actual_data is not None and not actual_data.empty:
        # Вычисляем максимальное значение один раз
        actual_max_values = np.array([max_from_avg(avg) for avg in actual_data[COL_AVG_PULLUPS]])
        actual_max_with_weight = np.round(actual_max_values / k2).astype(int)
        actual_max_achieved = np.max(actual_max_with_weight) if len(actual_max_with_weight) > 0 else 0
    
    # Предварительно вычисляем параметры для экстраполяции
    if len(max_pullups_with_weight_predicted) > 1:
        last_value = max_pullups_with_weight_predicted[-1]
        first_value = max_pullups_with_weight_predicted[0]
        days_total = predict_days[-1] - predict_days[0]
        progress_per_day = (last_value - first_value) / days_total if days_total > 0 else 0
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
        indices = np.where(max_pullups_with_weight_predicted >= max_pullups_with_weight_standard)[0]
        
        if len(indices) > 0:
            # Найден день достижения стандарта
            first_index = indices[0]
            achievement_dates[rank] = int(predict_days[first_index])
        elif progress_per_day > 0:
            # Экстраполируем, если есть положительный прогресс
            days_to_target = (max_pullups_with_weight_standard - last_value) / progress_per_day
            achievement_dates[rank] = int(predict_days[-1] + days_to_target)
        else:
            # Стандарт не будет достигнут в прогнозируемом периоде
            achievement_dates[rank] = None
    
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
    if df_2025_real.empty:
        # Если данных нет, возвращаем специальное сообщение вместо графика
        return {
            "data": [],
            "standards": [],
            "title": "Загрузите данные для построения прогноза",
            "xAxisLabel": "Дата",
            "yAxisLabel": "Среднее количество подтягиваний",
            "noUserData": True,
            "message": "Необходимо загрузить данные о ваших тренировках для построения персонального прогноза. Без данных прогноз невозможен."
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
            chart_data.append({
                "date": date,
                "day": row[COL_DAYS],
                "actual": row[COL_AVG_PULLUPS],
                "average": None,
                "maximum": None,
                "withWeight": None
            })
            
    # Добавляем прогнозные данные
    for i, day in enumerate(predict_days.tolist()):
        date = (start_date_2025 + timedelta(days=int(day))).strftime("%Y-%m-%d")
        # Проверяем, нет ли уже точки с такой датой
        existing_point = next((point for point in chart_data if point["date"] == date), None)
        if existing_point:
            # Если точка существует, добавляем к ней прогнозные значения
            existing_point["average"] = round(predicted_avg_pullups[i], 1)
            existing_point["maximum"] = max_pullups_no_weight_predicted[i]
            existing_point["withWeight"] = max_pullups_with_weight_predicted[i]
        else:
            # Если точки нет, создаем новую
            chart_data.append({
                "date": date,
                "day": day,
                "actual": None,
                "average": round(predicted_avg_pullups[i], 1),
                "maximum": max_pullups_no_weight_predicted[i],
                "withWeight": max_pullups_with_weight_predicted[i]
            })

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
    }


async def get_prediction_data(
    weight_category: str = "до 75", forecast_days: int = 90
) -> dict:
    """Получить данные для прогноза."""
    try:
        (
            df_2025_real,
            df_initial,
            pullup_standards,
        ) = await _load_all_data(weight_category)
        
        # Если данных пользователя нет, возвращаем сообщение без выполнения прогноза
        if df_2025_real.empty:
            logger.info("Данные пользователя отсутствуют. Прогноз не может быть построен.")
            
            # Текущая дата для сообщения
            current_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            predict_year = current_date.year
            
            # Создаем сообщение для графика без выполнения прогноза
            chart2_data = {
                "data": [],
                "standards": [],
                "title": "Загрузите данные для построения прогноза",
                "xAxisLabel": "Дата",
                "yAxisLabel": "Среднее количество подтягиваний",
                "noUserData": True,
                "message": "Необходимо загрузить данные о ваших тренировках для построения персонального прогноза. Без данных прогноз невозможен."
            }
            
            return {
                "data_2025": [],
                "chart2": json.dumps(convert_to_json_serializable(chart2_data)),
                "achievement_dates": {},
                "pullup_standards": pullup_standards,
                "forecast_days": forecast_days,
                "initial_year": None,
                "predict_year": predict_year,
                "progress_type": "Недостаточно данных",
                "growth_per_day": 0.0,
            }
        
        # Если данные есть, выполняем стандартный прогноз
        (
            df_initial,
            start_date_2025,
            initial_year,
            predict_year,
        ) = _prepare_data_for_regression(df_initial, df_2025_real)
        model = _build_regression_model(df_initial)
        predict_days, predicted_avg_pullups, progress_type, growth_per_day = _forecast(
            model, df_2025_real, df_initial, forecast_days
        )
        achievement_dates = _calculate_achievement_dates(
            predict_days,
            predicted_avg_pullups,
            pullup_standards,
            weight_category,
            df_2025_real,  # Передаем фактические данные
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

        data_2025_prepared = []
        for i in range(len(df_2025_real)):
            data_2025_prepared.append(
                TrainingData(
                    date=df_2025_real[COL_DATE].iloc[i].strftime("%Y-%m-%d"),
                    avg_pullups=df_2025_real[COL_AVG_PULLUPS].iloc[i],
                )
            )

        return {
            "data_2025": data_2025_prepared,
            "chart2": json.dumps(convert_to_json_serializable(chart2_data)),
            "achievement_dates": achievement_dates,
            "pullup_standards": pullup_standards,
            "forecast_days": forecast_days,
            "initial_year": initial_year,
            "predict_year": predict_year,
            "progress_type": progress_type,
            "growth_per_day": growth_per_day,
        }

    except DataError as e:
        logger.error(f"Error preparing prediction data: {e}")
        raise
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        raise
