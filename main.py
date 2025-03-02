# main.py
from fastapi import FastAPI, Request, Form
from fastapi.responses import RedirectResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime, timedelta
import os
import random

app = FastAPI()
templates = Jinja2Templates(directory="templates")
templates.env.globals["zip"] = zip
templates.env.globals["enumerate"] = enumerate
app.mount("/static", StaticFiles(directory="static"), name="static")


def process_dates(data, start_date):
    """Преобразует даты в количество дней с начальной даты."""
    return [(datetime.strptime(date, "%d.%m.%Y") - start_date).days for date in data]


def load_data_from_csv():
    """Загружает данные из CSV-файла data_2025.csv."""
    if os.path.exists("data_2025.csv"):
        df = pd.read_csv("data_2025.csv")
        if df.empty:
            return {"date": [], "avg_pullups": []}
        return {"date": df["date"].tolist(), "avg_pullups": df["avg_pullups"].tolist()}
    return {"date": [], "avg_pullups": []}


def save_data_to_csv(data):
    """Сохраняет данные в CSV-файл data_2025.csv."""
    df = pd.DataFrame(data)
    df.to_csv("data_2025.csv", index=False)


def load_initial_data():
    """Загружает начальные данные из CSV или возвращает дефолтные, если файла нет"""
    if os.path.exists("initial_data.csv"):
        df = pd.read_csv("initial_data.csv")
        if not df.empty:
            return {
                "date": df["date"].tolist(),
                "avg_pullups": df["avg_pullups"].tolist(),
            }
    # Дефолтные данные, если файла нет
    return {
        "date": [
            "12.07.2021",
            "14.07.2021",
            "17.07.2021",
            "20.07.2021",
            "22.07.2021",
            "24.07.2021",
            "29.07.2021",
            "31.07.2021",
            "02.08.2021",
            "05.08.2021",
            "07.08.2021",
            "12.08.2021",
            "16.08.2021",
            "18.08.2021",
            "20.08.2021",
            "23.08.2021",
            "25.08.2021",
            "28.08.2021",
            "02.09.2021",
            "06.09.2021",
            "07.09.2021",
            "11.09.2021",
            "14.09.2021",
            "17.09.2021",
            "20.09.2021",
            "26.09.2021",
        ],
        "avg_pullups": [
            3.4,
            3.6,
            4.6,
            4.0,
            4.8,
            5.6,
            6.2,
            6.0,
            6.8,
            7.0,
            7.4,
            6.8,
            6.8,
            7.2,
            7.6,
            8.2,
            9.0,
            8.2,
            9.8,
            9.0,
            10.0,
            10.4,
            9.0,
            10.0,
            9.4,
            9.2,
        ],
    }


@app.get("/")
async def index(request: Request, degree: int = 1):
    """Главная страница с графиками и формой добавления данных."""
    data_2025_real = load_data_from_csv()  # Загрузка пользовательских данных
    df_2025_real = pd.DataFrame(data_2025_real)

    initial_data = load_initial_data()  # Загрузка исходных данных
    df_initial = pd.DataFrame(initial_data)

    if not df_2025_real.empty:
        start_date_2025 = datetime.strptime(df_2025_real["date"].iloc[0], "%d.%m.%Y")
        df_2025_real["days"] = process_dates(df_2025_real["date"], start_date_2025)
        predict_year = start_date_2025.year
    else:
        df_2025_real["days"] = []
        predict_year = (
            datetime.now().year + 1
        )  # Если нет пользовательских данных, прогнозируем на следующий год

    start_date_initial = datetime.strptime(df_initial["date"].iloc[0], "%d.%m.%Y")
    initial_data["days"] = process_dates(initial_data["date"], start_date_initial)
    initial_year = start_date_initial.year

    X_initial = np.array(initial_data["days"]).reshape(-1, 1)
    y_initial = np.array(initial_data["avg_pullups"])

    poly = PolynomialFeatures(degree=degree)
    X_poly_initial = poly.fit_transform(X_initial)
    model = LinearRegression()
    model.fit(X_poly_initial, y_initial)

    if not df_2025_real.empty:
        first_day = df_2025_real["days"].iloc[0]
        predict_days_2025 = np.arange(first_day, first_day + 91, 4)

    else:
        predict_days_2025 = np.arange(0, 91, 4)

    if not df_2025_real.empty:
        dates_2025 = [
            start_date_2025 + timedelta(days=int(day)) for day in predict_days_2025
        ]
    else:
        # Если нет пользовательских данных, используем следующий год
        dates_2025 = [
            datetime(predict_year, 1, 14) + timedelta(days=int(day))
            for day in predict_days_2025
        ]

    predict_days_2025_poly = poly.transform(predict_days_2025.reshape(-1, 1))
    predicted_pullups_2025 = model.predict(predict_days_2025_poly)

    # Построение графика 1 (Исходные данные и линия регрессии)
    fig1 = go.Figure()
    fig1.add_trace(
        go.Scatter(
            x=initial_data["days"],
            y=initial_data["avg_pullups"],
            mode="markers+lines",
            name=f"Тренировки {initial_year}",
        )
    )
    fig1.add_trace(
        go.Scatter(
            x=initial_data["days"],
            y=model.predict(X_poly_initial),
            mode="lines",
            name=f"Линия регрессии ({initial_year})",
            line=dict(dash="dash"),
        )
    )
    fig1.update_layout(
        title=f"Динамика подтягиваний в {initial_year} году",
        xaxis_title="Дни с начала наблюдений",
        yaxis_title="Среднее количество подтягиваний",
        template="plotly_dark",
    )
    chart1 = fig1.to_html(full_html=False, include_plotlyjs="cdn")

    # Построение графика 2 (Прогноз и реальные данные)
    fig2 = go.Figure()
    dates_2025_str = [date.strftime("%d.%m.%Y") for date in dates_2025]
    date_objects = [datetime.strptime(date, "%d.%m.%Y") for date in dates_2025_str]
    fig2.add_trace(
        go.Scatter(
            x=date_objects,
            y=predicted_pullups_2025,
            mode="lines",
            name=f"Прогноз на 90 дней ({predict_year})",
            line=dict(dash="dash"),
        )
    )

    if not df_2025_real.empty:
        real_date_objects = [
            datetime.strptime(date, "%d.%m.%Y") for date in data_2025_real["date"]
        ]
        fig2.add_trace(
            go.Scatter(
                x=real_date_objects,
                y=data_2025_real["avg_pullups"],
                mode="markers+lines",
                name=f"Тренировки {predict_year}",
                line=dict(color="red"),
            )
        )

    fig2.update_layout(
        title=f"Прогноз и фактические тренировки в {predict_year} году",
        xaxis_title="Дата",
        yaxis_title="Среднее количество подтягиваний",
        xaxis=dict(tickformat="%d.%m.%Y"),
        template="plotly_dark",
    )
    chart2 = fig2.to_html(full_html=False, include_plotlyjs="cdn")

    # Расчет метрик качества
    X_initial_pred = np.array(initial_data["days"]).reshape(-1, 1)
    X_initial_pred_poly = poly.transform(X_initial_pred)
    predicted_initial = model.predict(X_initial_pred_poly)
    mae_initial = mean_absolute_error(y_initial, predicted_initial)
    r2_initial = r2_score(y_initial, predicted_initial)

    if len(data_2025_real["date"]) > 1:
        real_days_2025 = np.array(df_2025_real["days"]).reshape(-1, 1)
        real_days_2025_poly = poly.transform(real_days_2025)
        predicted_for_real_days = model.predict(real_days_2025_poly)
        real_values_2025 = np.array(data_2025_real["avg_pullups"])
        mae_2025 = mean_absolute_error(real_values_2025, predicted_for_real_days)
        r2_2025 = r2_score(real_values_2025, predicted_for_real_days)
        mae_2025_text = f"Ошибка прогноза (MAE) для {predict_year} года: {mae_2025:.2f}"
        r2_2025_text = f"R² для {predict_year} года: {r2_2025:.2f}"
    else:
        mae_2025_text = f"Недостаточно реальных данных для расчёта ошибки прогноза {predict_year} года."
        r2_2025_text = (
            f"Недостаточно реальных данных для расчёта R² {predict_year} года."
        )

    forecast_improvement = predicted_pullups_2025[-1] - predicted_pullups_2025[0]

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "data_2025": data_2025_real,
            "chart1": chart1,
            "chart2": chart2,
            "mae_2021": f"Ошибка прогноза (MAE) для {initial_year} года: {mae_initial:.2f}",
            "r2_2021": f"R² для {initial_year} года: {r2_initial:.2f}",
            "mae_2025": mae_2025_text,
            "r2_2025": r2_2025_text,
            "forecast_improvement": f"Прогнозируемый прирост за 90 дней: {forecast_improvement:.2f}",
            "selected_degree": degree,
        },
    )


@app.post("/add")
async def add_data(
    request: Request, date: str = Form(...), avg_pullups: float = Form(...)
):
    """Добавляет новую запись о тренировке в данные."""
    data = load_data_from_csv()
    data["date"].append(datetime.strptime(date, "%Y-%m-%d").strftime("%d.%m.%Y"))
    data["avg_pullups"].append(avg_pullups)
    save_data_to_csv(data)
    return RedirectResponse(url="/", status_code=303)


@app.get("/delete/{idx}")
async def delete_data(idx: int):
    """Удаляет запись о тренировке по индексу."""
    data = load_data_from_csv()
    try:
        data["date"].pop(idx)
        data["avg_pullups"].pop(idx)
    except IndexError:
        pass
    save_data_to_csv(data)
    return RedirectResponse(url="/", status_code=303)


@app.get("/reset")
async def reset_data():
    """Сбрасывает данные к пустому состоянию."""
    df = pd.DataFrame(columns=["date", "avg_pullups"])
    df.to_csv("data_2025.csv", index=False)
    return RedirectResponse(url="/", status_code=303)


@app.get("/download")
async def download_csv():
    """Позволяет скачать данные в формате CSV."""
    return FileResponse(
        "data_2025.csv", media_type="text/csv", filename="data_2025.csv"
    )


@app.get("/about")
async def about(request: Request):
    """Страница "О приложении"."""
    return templates.TemplateResponse("about.html", {"request": request})


@app.get("/history")
async def history(request: Request):
    """Страница с историей данных."""
    data = load_data_from_csv()
    return templates.TemplateResponse(
        "history.html", {"request": request, "data": data}
    )
