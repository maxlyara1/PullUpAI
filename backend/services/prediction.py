import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
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
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

FORECAST_INTERVAL = 4

# Кэш для моделей регрессии
MODEL_CACHE = {}

try:
    k1_model = joblib.load("backend/k1_poly_model.pkl")
    poly = joblib.load("backend/k1_poly_features.pkl")
except FileNotFoundError:
    k1_model = None
    poly = None
    print("Модель k1_poly не найдена. Будет использовано базовое значение K1.")


@lru_cache(maxsize=32)
def _get_cached_model_key(df_initial_json: str, degree: int) -> str:
    """Получить ключ для кэша модели."""
    return f"{df_initial_json}_{degree}"


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
    if not df_2025_real.empty:
        df_2025_real["date"] = pd.to_datetime(df_2025_real["date"], format="%d.%m.%Y")
        start_date_2025 = df_2025_real["date"].iloc[0]
        predict_year = start_date_2025.year
        df_2025_real["days"] = process_dates(df_2025_real["date"], start_date_2025)
    else:
        start_date_2025 = datetime(datetime.now().year + 1, 1, 14)
        predict_year = start_date_2025.year
        df_2025_real["days"] = []

    if not df_initial.empty:
        df_initial["date"] = pd.to_datetime(df_initial["date"], format="%d.%m.%Y")
        start_date_initial = df_initial["date"].iloc[0]
        df_initial["days"] = process_dates(df_initial["date"], start_date_initial)

        df_initial = df_initial.sort_values(by="date")
        df_initial["lag_avg_pullups_1"] = df_initial["avg_pullups"].shift(1)
        df_initial["lag_avg_pullups_2"] = df_initial["avg_pullups"].shift(2)
        df_initial["lag_avg_pullups_3"] = df_initial["avg_pullups"].shift(3)
        df_initial["pullups_growth"] = df_initial["avg_pullups"].diff()
        df_initial = df_initial.fillna(0)

    else:
        start_date_initial = datetime(2021, 7, 12)
        df_initial["days"] = []
        df_initial["lag_avg_pullups_1"] = 0
        df_initial["lag_avg_pullups_2"] = 0
        df_initial["lag_avg_pullups_3"] = 0
        df_initial["pullups_growth"] = 0

    initial_year = start_date_initial.year

    return df_initial, start_date_2025, initial_year, predict_year


def _build_regression_model(initial_data_processed: pd.DataFrame, degree: int) -> tuple:
    """Построить модель регрессии."""
    # Создаем ключ для кэша на основе данных и степени полинома
    df_json = initial_data_processed.to_json()
    cache_key = _get_cached_model_key(df_json, degree)

    # Проверяем, есть ли модель в кэше
    if cache_key in MODEL_CACHE:
        logger.info("Using cached regression model")
        return MODEL_CACHE[cache_key]

    logger.info("Building new regression model")
    X_initial = initial_data_processed[
        [
            "days",  # Используем дни
            "lag_avg_pullups_1",
            "lag_avg_pullups_2",
            "lag_avg_pullups_3",
            "pullups_growth",
        ]
    ].to_numpy()
    y_initial = initial_data_processed["avg_pullups"].to_numpy()
    poly = PolynomialFeatures(degree=degree)
    X_poly_initial = poly.fit_transform(X_initial)
    model = LinearRegression()
    model.fit(X_poly_initial, y_initial)

    # Сохраняем модель в кэш
    MODEL_CACHE[cache_key] = (model, poly, X_poly_initial)
    return model, poly, X_poly_initial


def _generate_forecast(
    model: LinearRegression,
    poly: PolynomialFeatures,
    df_2025_real: pd.DataFrame,
    df_initial: pd.DataFrame,
    start_date_2025: datetime,
    forecast_days: int,
) -> tuple:
    """Сгенерировать прогноз."""
    if not df_2025_real.empty:
        # Находим последнюю дату с фактическими данными
        last_real_day = df_2025_real["days"].max()
        # Генерируем дни для прогноза, начиная с последней фактической даты
        predict_days = np.arange(
            last_real_day, last_real_day + forecast_days, FORECAST_INTERVAL
        )
    else:
        predict_days = np.arange(0, forecast_days, FORECAST_INTERVAL)

    historical_data_for_forecast = pd.DataFrame()

    if not df_2025_real.empty:
        historical_data_for_forecast = df_2025_real.copy()
    elif not df_initial.empty:
        historical_data_for_forecast = df_initial.copy()
    else:
        historical_data_for_forecast["avg_pullups"] = [0] * 3
        historical_data_for_forecast["days"] = [-3, -2, -1]

    # Add lag features to historical_data_for_forecast for inference
    historical_data_for_forecast = historical_data_for_forecast.sort_values(by="days")

    # Создаем скользящее окно для более плавных лагов
    window_size = 3
    pullups_series = historical_data_for_forecast["avg_pullups"]

    # Используем скользящее среднее для сглаживания
    smoothed_pullups = pullups_series.rolling(
        window=window_size, min_periods=1, center=True
    ).mean()

    # Заполняем лаги с использованием сглаженных данных
    historical_data_for_forecast["lag_avg_pullups_1"] = smoothed_pullups.shift(1)
    historical_data_for_forecast["lag_avg_pullups_2"] = smoothed_pullups.shift(2)
    historical_data_for_forecast["lag_avg_pullups_3"] = smoothed_pullups.shift(3)

    # Рассчитываем рост как разницу между сглаженными значениями
    historical_data_for_forecast["pullups_growth"] = smoothed_pullups.diff()

    # Заполняем пропущенные значения последним известным значением
    last_known_value = smoothed_pullups.iloc[-1]
    historical_data_for_forecast = historical_data_for_forecast.fillna(
        method="ffill"
    ).fillna(last_known_value)

    predicted_pullups = []
    last_pullups = smoothed_pullups.tolist()[-3:]
    last_lags = historical_data_for_forecast[
        ["lag_avg_pullups_1", "lag_avg_pullups_2", "lag_avg_pullups_3"]
    ].values.tolist()[-3:]
    last_growth = historical_data_for_forecast["pullups_growth"].tolist()[-1]

    # Используем скользящее окно для прогноза
    window_predictions = []
    for day in predict_days:
        input_features = np.array(
            [
                [
                    day,
                    last_lags[-1][0] if last_lags else last_known_value,
                    last_lags[-1][1] if len(last_lags) > 1 else last_known_value,
                    last_lags[-1][2] if len(last_lags) > 2 else last_known_value,
                    last_growth,
                ]
            ]
        )

        input_features_poly = poly.transform(input_features)
        predicted_avg = model.predict(input_features_poly)[0]

        # Добавляем предсказание в окно
        window_predictions.append(predicted_avg)
        if len(window_predictions) > window_size:
            window_predictions.pop(0)

        # Используем среднее по окну для более плавного прогноза
        smoothed_prediction = np.mean(window_predictions)
        predicted_pullups.append(smoothed_prediction)

        # Обновляем значения для следующего прогноза
        last_pullups = last_pullups[1:] + [smoothed_prediction]
        last_lags = last_lags[1:] + [
            [last_pullups[-1], last_pullups[-2], last_pullups[-3]]
        ]
        last_growth = (
            smoothed_prediction - last_pullups[-2] if len(last_pullups) > 1 else 0
        )

    # Если есть фактические данные, добавляем их в начало прогноза
    if not df_2025_real.empty:
        predict_days = np.concatenate([df_2025_real["days"].values, predict_days])
        predicted_pullups = np.concatenate(
            [df_2025_real["avg_pullups"].values, predicted_pullups]
        )

    return predict_days, predicted_pullups


def _calculate_achievement_dates(
    predict_days: np.ndarray,
    predicted_avg_pullups: np.ndarray,
    pullup_standards: dict,
    weight_category: str,
    actual_data: pd.DataFrame = None,  # Добавляем параметр для фактических данных
) -> dict:
    """Рассчитать даты достижения стандартов."""
    achievement_dates = {}
    k2 = calculate_k2(weight_category)
    max_pullups_no_weight_predicted = [
        max_from_avg(avg) for avg in predicted_avg_pullups
    ]
    max_pullups_with_weight_predicted = [
        round(max_p / k2) for max_p in max_pullups_no_weight_predicted
    ]

    # Проверяем фактические достижения
    actual_max_achieved = 0
    if actual_data is not None and not actual_data.empty:
        actual_max_achieved = max(
            round(max_from_avg(avg) / k2) for avg in actual_data["avg_pullups"]
        )

    for rank, max_pullups_with_weight_standard in pullup_standards.items():
        # Если разряд уже достигнут по фактическим данным
        if actual_max_achieved >= max_pullups_with_weight_standard:
            achievement_dates[rank] = 0  # 0 означает "уже достигнуто"
            continue

        try:
            day_achieved = next(
                day
                for day, predicted_max_with_weight in zip(
                    predict_days, max_pullups_with_weight_predicted
                )
                if predicted_max_with_weight >= max_pullups_with_weight_standard
            )
            achievement_dates[rank] = int(day_achieved)
        except StopIteration:
            # Если не нашли точку пересечения, экстраполируем
            if len(max_pullups_with_weight_predicted) > 1:
                last_value = max_pullups_with_weight_predicted[-1]
                first_value = max_pullups_with_weight_predicted[0]
                days_total = predict_days[-1] - predict_days[0]
                progress_per_day = (last_value - first_value) / days_total
                if progress_per_day > 0:
                    days_to_target = (
                        max_pullups_with_weight_standard - last_value
                    ) / progress_per_day
                    achievement_dates[rank] = int(predict_days[-1] + days_to_target)
                else:
                    achievement_dates[rank] = None
            else:
                achievement_dates[rank] = None

    return achievement_dates


def _create_chart1_data(
    initial_data_processed: pd.DataFrame,
    model: LinearRegression,
    X_poly_initial: np.ndarray,
    initial_year: int,
) -> dict:
    """Создать данные для первого графика."""
    regression_line = model.predict(X_poly_initial).tolist()

    chart_data = []
    for i, day in enumerate(initial_data_processed["days"].tolist()):
        chart_data.append(
            {
                "day": day,
                "actual": initial_data_processed["avg_pullups"].tolist()[i],
                "regression": regression_line[i],
            }
        )

    return {
        "data": chart_data,
        "title": f"Динамика подтягиваний в {initial_year} году",
        "xAxisLabel": "Дни с начала наблюдений",
        "yAxisLabel": "Среднее количество подтягиваний",
    }


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
    for i, day in enumerate(predict_days.tolist()):
        date = (start_date_2025 + timedelta(days=int(day))).strftime("%Y-%m-%d")
        point = {
            "date": date,
            "day": day,
            "average": round(predicted_avg_pullups[i], 1),
            "maximum": max_pullups_no_weight_predicted[i],
            "withWeight": max_pullups_with_weight_predicted[i],
        }
        chart_data.append(point)

    # Добавляем фактические данные, если есть
    if not df_2025_real.empty:
        for i, day in enumerate(df_2025_real["days"].tolist()):
            date = (start_date_2025 + timedelta(days=int(day))).strftime("%Y-%m-%d")
            existing_point = next(
                (point for point in chart_data if point["day"] == day), None
            )
            if existing_point:
                existing_point["actual"] = df_2025_real["avg_pullups"].tolist()[i]
            else:
                chart_data.append(
                    {
                        "date": date,
                        "day": day,
                        "actual": df_2025_real["avg_pullups"].tolist()[i],
                    }
                )

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
    }


def _calculate_metrics(
    initial_data_processed: pd.DataFrame,
    model: LinearRegression,
    poly: PolynomialFeatures,
    df_2025_real: pd.DataFrame,
    df_initial: pd.DataFrame,
    predict_year: int,
    initial_year: int,
) -> tuple:
    """Рассчитать метрики модели."""
    X_initial_pred = initial_data_processed[
        [
            "days",  # Используем дни
            "lag_avg_pullups_1",
            "lag_avg_pullups_2",
            "lag_avg_pullups_3",
            "pullups_growth",
        ]
    ].to_numpy()
    X_initial_pred_poly = poly.transform(X_initial_pred)
    predicted_initial = model.predict(X_initial_pred_poly)
    mae_initial = mean_absolute_error(
        initial_data_processed["avg_pullups"], predicted_initial
    )
    r2_initial = r2_score(initial_data_processed["avg_pullups"], predicted_initial)

    if not df_2025_real.empty and len(df_2025_real["date"]) > 1:
        df_2025_real_processed = df_2025_real.copy()
        if not df_2025_real_processed.empty:
            df_2025_real_processed = df_2025_real_processed.sort_values(by="date")
            df_2025_real_processed["lag_avg_pullups_1"] = df_2025_real_processed[
                "avg_pullups"
            ].shift(1)
            df_2025_real_processed["lag_avg_pullups_2"] = df_2025_real_processed[
                "avg_pullups"
            ].shift(2)
            df_2025_real_processed["lag_avg_pullups_3"] = df_2025_real_processed[
                "avg_pullups"
            ].shift(3)
            df_2025_real_processed["pullups_growth"] = df_2025_real_processed[
                "avg_pullups"
            ].diff()
            df_2025_real_processed = df_2025_real_processed.fillna(0)
        else:
            df_2025_real_processed["lag_avg_pullups_1"] = 0
            df_2025_real_processed["lag_avg_pullups_2"] = 0
            df_2025_real_processed["lag_avg_pullups_3"] = 0
            df_2025_real_processed["pullups_growth"] = 0

        real_days_2025 = df_2025_real_processed[
            [
                "days",  # Используем дни
                "lag_avg_pullups_1",
                "lag_avg_pullups_2",
                "lag_avg_pullups_3",
                "pullups_growth",
            ]
        ].to_numpy()
        real_days_2025_poly = poly.transform(real_days_2025)
        predicted_for_real_days = model.predict(real_days_2025_poly)
        real_values_2025 = df_2025_real["avg_pullups"].to_numpy()
        mae_2025 = mean_absolute_error(real_values_2025, predicted_for_real_days)
        r2_2025 = r2_score(real_values_2025, predicted_for_real_days)
    else:
        mae_2025 = None
        r2_2025 = None

    try:
        predict_days, predicted_pullups = _generate_forecast(
            model,
            poly,
            df_2025_real,
            df_initial,
            datetime(predict_year, 1, 14),
            90,
        )
        forecast_improvement = (
            predicted_pullups[-1] - predicted_pullups[0]
            if len(predicted_pullups) > 1
            else None
        )
    except UnboundLocalError:
        forecast_improvement = None

    return mae_initial, r2_initial, mae_2025, r2_2025, forecast_improvement


async def get_prediction_data(
    weight_category: str = "до 75", forecast_days: int = 90
) -> dict:
    """Получить данные для прогноза."""
    try:
        degree = 1  # Фиксируем линейную регрессию
        (
            df_2025_real,
            df_initial,
            pullup_standards,
        ) = await _load_all_data(weight_category)
        (
            df_initial,
            start_date_2025,
            initial_year,
            predict_year,
        ) = _prepare_data_for_regression(df_initial, df_2025_real)
        model, poly, X_poly_initial = _build_regression_model(df_initial, degree)
        predict_days, predicted_avg_pullups = _generate_forecast(
            model, poly, df_2025_real, df_initial, start_date_2025, forecast_days
        )
        achievement_dates = _calculate_achievement_dates(
            predict_days,
            predicted_avg_pullups,
            pullup_standards,
            weight_category,
            df_2025_real,  # Передаем фактические данные
        )
        chart1_data = _create_chart1_data(
            df_initial,
            model,
            X_poly_initial,
            initial_year,
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
        (
            mae_initial,
            r2_initial,
            mae_2025,
            r2_2025,
            forecast_improvement,
        ) = _calculate_metrics(
            df_initial,
            model,
            poly,
            df_2025_real,
            df_initial,
            predict_year,
            initial_year,
        )

        data_2025_prepared = []
        for i in range(len(df_2025_real)):
            data_2025_prepared.append(
                TrainingData(
                    date=df_2025_real["date"].iloc[i].strftime("%Y-%m-%d"),
                    avg_pullups=df_2025_real["avg_pullups"].iloc[i],
                )
            )

        return {
            "data_2025": data_2025_prepared,
            "chart1": json.dumps(convert_to_json_serializable(chart1_data)),
            "chart2": json.dumps(convert_to_json_serializable(chart2_data)),
            "mae_2021": mae_initial,
            "r2_2021": r2_initial,
            "mae_2025": mae_2025,
            "r2_2025": r2_2025,
            "forecast_improvement": forecast_improvement,
            "selected_degree": degree,
            "achievement_dates": achievement_dates,
            "pullup_standards": pullup_standards,
            "forecast_days": forecast_days,
            "initial_year": initial_year,
            "predict_year": predict_year,
        }

    except DataError as e:
        logger.error(f"Error preparing prediction data: {e}")
        raise
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        raise
