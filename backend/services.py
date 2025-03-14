import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime, timedelta
import json
import logging
import joblib
from backend.database import (
    load_data,
    save_data,
    load_initial_data,
    reset_data,
    load_original_standards,
    DataError,
    DataFormatError,
)
from backend.models.training import TrainingData


def setup_logger(name: str) -> logging.Logger:
    """Настройка логгера с форматированием."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


logger = setup_logger(__name__)

FORECAST_INTERVAL = 4

try:
    k1_model = joblib.load("backend/k1_poly_model.pkl")
    poly = joblib.load("backend/k1_poly_features.pkl")
except FileNotFoundError:
    k1_model = None
    poly = None
    print("Модель k1_poly не найдена. Будет использовано базовое значение K1.")


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
    return WEIGHT_CATEGORY_MAPPING.get(weight_category, 75)


def calculate_k2(weight_category: str, extra_weight: float = 24.0) -> float:
    body_weight = get_body_weight(weight_category)
    return 1 + (extra_weight / body_weight)


def calculate_pullup_standards(
    original_standards_df: pd.DataFrame, weight_category: str
) -> dict:
    filtered_df = original_standards_df[
        original_standards_df["weight_category"] == weight_category
    ]
    adapted_standards = {}
    for _, row in filtered_df.iterrows():
        rank = row["rank"]
        adapted_standards[rank] = row["max_pullups"]
    return adapted_standards


def process_dates(dates: pd.Series, start_date: datetime) -> list:
    return [(date - start_date).days for date in dates]


async def _load_all_data(weight_category: str) -> tuple:
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
        # Используем первое значение ряда для заполнения лагов
        initial_value = df_initial["avg_pullups"].iloc[0]
        df_initial["lag_avg_pullups_1"] = (
            df_initial["avg_pullups"].shift(1).fillna(initial_value)
        )
        df_initial["lag_avg_pullups_2"] = (
            df_initial["avg_pullups"].shift(2).fillna(initial_value)
        )
        df_initial["lag_avg_pullups_3"] = (
            df_initial["avg_pullups"].shift(3).fillna(initial_value)
        )
        # Рост в начале считаем как 0
        df_initial["pullups_growth"] = df_initial["avg_pullups"].diff().fillna(0)

    else:
        start_date_initial = datetime(2021, 7, 12)
        df_initial["days"] = []
        df_initial["lag_avg_pullups_1"] = []
        df_initial["lag_avg_pullups_2"] = []
        df_initial["lag_avg_pullups_3"] = []
        df_initial["pullups_growth"] = []

    initial_year = start_date_initial.year

    return df_initial, start_date_2025, initial_year, predict_year


def _build_regression_model(initial_data_processed: pd.DataFrame, degree: int) -> tuple:
    X_initial = initial_data_processed[
        [
            "days",  # Используем дни без масштабирования
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
    return model, poly, X_poly_initial


def _generate_forecast(
    model: LinearRegression,
    poly: PolynomialFeatures,
    df_2025_real: pd.DataFrame,
    df_initial: pd.DataFrame,
    start_date_2025: datetime,
    forecast_days: int,
) -> tuple:
    if not df_2025_real.empty:
        first_day = df_2025_real["days"].iloc[0]
        predict_days = np.arange(
            first_day, first_day + forecast_days, FORECAST_INTERVAL
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

    # Используем последнее известное значение для начальных лагов
    last_known_value = historical_data_for_forecast["avg_pullups"].iloc[-1]
    historical_data_for_forecast["lag_avg_pullups_1"] = (
        historical_data_for_forecast["avg_pullups"].shift(1).fillna(last_known_value)
    )
    historical_data_for_forecast["lag_avg_pullups_2"] = (
        historical_data_for_forecast["avg_pullups"].shift(2).fillna(last_known_value)
    )
    historical_data_for_forecast["lag_avg_pullups_3"] = (
        historical_data_for_forecast["avg_pullups"].shift(3).fillna(last_known_value)
    )
    historical_data_for_forecast["pullups_growth"] = (
        historical_data_for_forecast["avg_pullups"].diff().fillna(0)
    )

    predicted_pullups = []
    last_pullups = historical_data_for_forecast["avg_pullups"].tolist()[-3:]
    last_lags = historical_data_for_forecast[
        ["lag_avg_pullups_1", "lag_avg_pullups_2", "lag_avg_pullups_3"]
    ].values.tolist()[-3:]
    last_growth = historical_data_for_forecast["pullups_growth"].tolist()[-1]

    for day in predict_days:
        input_features = np.array(
            [
                [
                    day,  # Используем немасштабированные дни
                    (
                        last_lags[-1][0] if last_lags else last_known_value
                    ),  # lag_avg_pullups_1
                    (
                        last_lags[-1][1] if len(last_lags) > 1 else last_known_value
                    ),  # lag_avg_pullups_2
                    (
                        last_lags[-1][2] if len(last_lags) > 2 else last_known_value
                    ),  # lag_avg_pullups_3
                    last_growth,
                ]
            ]
        )

        input_features_poly = poly.transform(input_features)
        predicted_avg = model.predict(input_features_poly)[0]
        predicted_pullups.append(predicted_avg)

        last_pullups = last_pullups[1:]  # remove the oldest lag value
        last_pullups.append(
            predicted_avg
        )  # add the current prediction as the newest lag
        last_lags.append(
            [last_pullups[-1], last_pullups[-2], last_pullups[-3]]
        )  # update lags
        last_lags = last_lags[1:]  # keep lags list size 3
        last_growth = (
            predicted_avg - last_pullups[-2] if len(last_pullups) > 1 else 0
        )  # update growth

    return predict_days, np.array(predicted_pullups)


def _calculate_achievement_dates(
    predict_days: np.ndarray,
    predicted_avg_pullups: np.ndarray,
    pullup_standards: dict,
    weight_category: str,
) -> dict:
    achievement_dates = {}
    k2 = calculate_k2(weight_category)
    for rank, max_pullups_with_weight_standard in pullup_standards.items():
        max_pullups_no_weight_predicted = [
            max_from_avg(avg) for avg in predicted_avg_pullups
        ]
        max_pullups_with_weight_predicted = [
            round(max_p / k2) for max_p in max_pullups_no_weight_predicted
        ]
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
            achievement_dates[rank] = "Не достигнуто в прогнозе"
    return achievement_dates


def _create_chart1_data(
    initial_data_processed: pd.DataFrame,
    model: LinearRegression,
    X_poly_initial: np.ndarray,
    initial_year: int,
) -> dict:
    chart1_data = {
        "data": [
            {
                "x": initial_data_processed["days"].tolist(),
                "y": initial_data_processed["avg_pullups"].tolist(),
                "mode": "markers+lines",
                "name": f"Тренировки {initial_year}",
            },
            {
                "x": initial_data_processed["days"].tolist(),
                "y": model.predict(X_poly_initial).tolist(),
                "mode": "lines",
                "name": f"Линия регрессии ({initial_year})",
                "line": {"dash": "dash"},
            },
        ],
        "layout": {
            "title": f"Динамика подтягиваний в {initial_year} году",
            "xaxis": {"title": "Дни с начала наблюдений"},
            "yaxis": {"title": "Среднее количество подтягиваний"},
        },
    }
    return chart1_data


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
    chart2_data = {
        "data": [
            {
                "x": predict_days.tolist(),
                "y": predicted_avg_pullups.tolist(),
                "mode": "lines",
                "name": f"Прогноз (среднее без отягощения)",
                "line": {"dash": "dash"},
            },
        ],
        "layout": {
            "title": f"Прогноз и фактические тренировки в {predict_year} году",
            "xaxis": {"title": "Дата"},
            "yaxis": {"title": "Среднее количество подтягиваний"},
            "annotations": [],
        },
    }

    if not df_2025_real.empty:
        chart2_data["data"].append(
            {
                "x": df_2025_real["days"].tolist(),
                "y": df_2025_real["avg_pullups"].tolist(),
                "mode": "markers+lines",
                "name": f"Тренировки {predict_year}",
                "line": {"color": "red"},
            }
        )

        max_pullups_no_weight_predicted = [
            max_from_avg(avg) for avg in predicted_avg_pullups
        ]
        chart2_data["data"].append(
            {
                "x": predict_days.tolist(),
                "y": max_pullups_no_weight_predicted,
                "mode": "lines",
                "name": "Прогноз (максимум без отягощения)",
                "line": {"dash": "dash", "color": "blue"},
            }
        )

    k2 = calculate_k2(weight_category)
    max_pullups_with_weight_predicted = [
        round(max_p / k2) for max_p in max_pullups_no_weight_predicted
    ]

    chart2_data["data"].append(
        {
            "x": predict_days.tolist(),
            "y": max_pullups_with_weight_predicted,
            "mode": "lines",
            "name": f"Прогноз c отягощением",
            "line": {"dash": "dot", "color": "green"},
        }
    )

    for rank, pullups in pullup_standards.items():
        if isinstance(achievement_dates.get(rank), int):
            formatted_date = (
                start_date_2025 + timedelta(days=achievement_dates[rank])
            ).strftime("%Y-%m-%d")
        else:
            formatted_date = "Не достигнуто"

        chart2_data["data"].append(
            {
                "type": "scatter",
                "x": [predict_days[0], predict_days[-1]],
                "y": [pullups, pullups],
                "mode": "lines",
                "name": f"{rank} ({pullups})",
                "line": {"dash": "dot", "color": "gray"},
                "showlegend": False,
            }
        )
        chart2_data["layout"]["annotations"].append(
            {
                "x": predict_days[-1],
                "y": pullups,
                "xref": "x",
                "yref": "y",
                "text": f"{rank} ({pullups})",
                "showarrow": True,
                "arrowhead": 7,
                "ax": -30,
                "ay": 0,
            }
        )

        if formatted_date != "Не достигнуто":
            chart2_data["data"].append(
                {
                    "type": "scatter",
                    "x": [achievement_dates[rank], achievement_dates[rank]],
                    "y": [0, pullups],
                    "mode": "lines",
                    "name": f"Дата достижения {rank}",
                    "line": {"dash": "dashdot", "color": "yellow"},
                    "showlegend": False,
                }
            )
            chart2_data["layout"]["annotations"].append(
                {
                    "x": achievement_dates[rank],
                    "y": pullups,
                    "xref": "x",
                    "yref": "y",
                    "text": formatted_date,
                    "showarrow": True,
                    "arrowhead": 7,
                    "ax": 0,
                    "ay": -40,
                }
            )

    tickvals = predict_days.tolist()
    ticktext = [
        (start_date_2025 + timedelta(days=int(day))).strftime("%Y-%m-%d")
        for day in predict_days
    ]
    chart2_data["layout"]["xaxis"]["tickvals"] = tickvals
    chart2_data["layout"]["xaxis"]["ticktext"] = ticktext
    chart2_data["layout"]["xaxis"]["tickangle"] = -45

    return chart2_data


def _calculate_metrics(
    initial_data_processed: pd.DataFrame,
    model: LinearRegression,
    poly: PolynomialFeatures,
    df_2025_real: pd.DataFrame,
    df_initial: pd.DataFrame,
    predict_year: int,
    initial_year: int,
) -> tuple:
    X_initial_pred = initial_data_processed[
        [
            "days",  # Используем дни без масштабирования
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
                "days",  # Используем дни без масштабирования
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


async def add_training_data(date: str, avg_pullups: float):
    """Добавить новые данные о тренировке в базу данных."""
    df = await load_data()
    try:
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        formatted_date = date_obj.strftime("%d.%m.%Y")
    except ValueError as exc:
        raise DataFormatError("Неверный формат даты. Используйте YYYY-MM-DD.") from exc

    new_row = pd.DataFrame({"date": [formatted_date], "avg_pullups": [avg_pullups]})
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

    await save_data(current_data)


async def get_all_training_data():
    """Получить все данные о тренировках из базы данных."""
    df = await load_data()
    return {
        "date": pd.to_datetime(df["date"], format="%d.%m.%Y")
        .dt.strftime("%Y-%m-%d")
        .tolist(),
        "avg_pullups": df["avg_pullups"].tolist(),
    }
