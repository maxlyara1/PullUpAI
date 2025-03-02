from fastapi import FastAPI, Request, Form
from fastapi.responses import RedirectResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from datetime import datetime, timedelta
import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")
templates.env.globals["zip"] = zip
templates.env.globals["enumerate"] = enumerate
app.mount("/static", StaticFiles(directory="static"), name="static")


def process_dates(data, start_date):
    return [(datetime.strptime(date, "%d.%m.%Y") - start_date).days for date in data]


def load_data_from_csv():
    if os.path.exists("data_2025.csv"):
        df = pd.read_csv("data_2025.csv")
        if df.empty:
            return {"date": [], "avg_pullups": []}
        return {"date": df["date"].tolist(), "avg_pullups": df["avg_pullups"].tolist()}
    return {"date": [], "avg_pullups": []}


def save_data_to_csv(data):
    df = pd.DataFrame(data)
    df.to_csv("data_2025.csv", index=False)


data_2021 = {
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
    data_2025_real = load_data_from_csv()
    df_2025_real = pd.DataFrame(data_2025_real)
    if not df_2025_real.empty:
        start_date_2025 = datetime.strptime(df_2025_real["date"].iloc[0], "%d.%m.%Y")
        df_2025_real["days"] = process_dates(df_2025_real["date"], start_date_2025)
    else:
        df_2025_real["days"] = []
    start_date_2021 = datetime.strptime("12.07.2021", "%d.%m.%Y")
    data_2021["days"] = process_dates(data_2021["date"], start_date_2021)
    X_2021 = np.array(data_2021["days"]).reshape(-1, 1)
    y_2021 = np.array(data_2021["avg_pullups"])
    poly = PolynomialFeatures(degree=degree)
    X_poly_2021 = poly.fit_transform(X_2021)
    model = LinearRegression()
    model.fit(X_poly_2021, y_2021)
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
        dates_2025 = [
            datetime(2025, 1, 14) + timedelta(days=int(day))
            for day in predict_days_2025
        ]
    predict_days_2025_poly = poly.transform(predict_days_2025.reshape(-1, 1))
    predicted_pullups_2025 = model.predict(predict_days_2025_poly)
    fig1 = go.Figure()
    fig1.add_trace(
        go.Scatter(
            x=data_2021["days"],
            y=data_2021["avg_pullups"],
            mode="markers+lines",
            name="Реальные данные 2021",
        )
    )
    fig1.add_trace(
        go.Scatter(
            x=predict_days_2025,
            y=predicted_pullups_2025,
            mode="lines",
            name="Прогноз на 2025 год (90 дней)",
            line=dict(dash="dash"),
        )
    )
    fig1.update_layout(
        title="Прогресс в подтягиваниях и прогноз для 2025 года",
        xaxis_title="Дни с начала периода",
        yaxis_title="Среднее число подтягиваний",
        template="plotly_dark",
    )
    chart1 = fig1.to_html(full_html=False, include_plotlyjs="cdn")
    fig2 = go.Figure()
    dates_2025_str = [date.strftime("%d.%m.%Y") for date in dates_2025]
    date_objects = [datetime.strptime(date, "%d.%m.%Y") for date in dates_2025_str]
    fig2.add_trace(
        go.Scatter(
            x=date_objects,
            y=predicted_pullups_2025,
            mode="lines",
            name="Прогноз на 2025 год (90 дней)",
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
                name="Реальные данные 2025",
                line=dict(color="red"),
            )
        )
    fig2.update_layout(
        title="Прогноз и реальные данные для 2025 года",
        xaxis_title="Дата",
        yaxis_title="Среднее число подтягиваний",
        xaxis=dict(tickformat="%d.%m.%Y"),
        template="plotly_dark",
    )
    chart2 = fig2.to_html(full_html=False, include_plotlyjs="cdn")
    X_2021_pred = np.array(data_2021["days"]).reshape(-1, 1)
    X_2021_pred_poly = poly.transform(X_2021_pred)
    predicted_2021 = model.predict(X_2021_pred_poly)
    mae_2021 = mean_absolute_error(y_2021, predicted_2021)
    if len(data_2025_real["date"]) > 1:
        real_days_2025 = np.array(df_2025_real["days"]).reshape(-1, 1)
        real_days_2025_poly = poly.transform(real_days_2025)
        predicted_for_real_days = model.predict(real_days_2025_poly)
        real_values_2025 = np.array(data_2025_real["avg_pullups"])
        mae_2025 = mean_absolute_error(real_values_2025, predicted_for_real_days)
        mae_2025_text = f"Ошибка прогноза (MAE) для 2025 года: {mae_2025:.2f}"
    else:
        mae_2025_text = (
            "Недостаточно реальных данных для расчёта ошибки прогноза 2025 года."
        )
    forecast_improvement = predicted_pullups_2025[-1] - predicted_pullups_2025[0]
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "data_2025": data_2025_real,
            "chart1": chart1,
            "chart2": chart2,
            "mae_2021": f"Ошибка прогноза (MAE) для 2021 года: {mae_2021:.2f}",
            "mae_2025": mae_2025_text,
            "forecast_improvement": f"Прогнозируемый прирост за 90 дней: {forecast_improvement:.2f}",
            "selected_degree": degree,
        },
    )


@app.post("/add")
async def add_data(
    request: Request, date: str = Form(...), avg_pullups: float = Form(...)
):
    data = load_data_from_csv()
    data["date"].append(datetime.strptime(date, "%Y-%m-%d").strftime("%d.%m.%Y"))
    data["avg_pullups"].append(avg_pullups)
    save_data_to_csv(data)
    return RedirectResponse(url="/", status_code=303)


@app.get("/delete/{idx}")
async def delete_data(idx: int):
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
    df = pd.DataFrame(columns=["date", "avg_pullups"])
    df.to_csv("data_2025.csv", index=False)
    return RedirectResponse(url="/", status_code=303)


@app.get("/download")
async def download_csv():
    return FileResponse(
        "data_2025.csv", media_type="text/csv", filename="data_2025.csv"
    )


@app.get("/about")
async def about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})


@app.get("/history")
async def history(request: Request):
    data = load_data_from_csv()
    return templates.TemplateResponse(
        "history.html", {"request": request, "data": data}
    )
