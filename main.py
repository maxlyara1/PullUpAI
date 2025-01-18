import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import os
from sklearn.metrics import mean_absolute_error


def process_dates(data, start_date):
    return [(datetime.strptime(date, "%d.%m.%Y") - start_date).days for date in data]


def save_data_to_csv():
    # Сохранение данных в CSV
    df = pd.DataFrame(st.session_state.data_2025_real)
    df.to_csv("data_2025.csv", index=False)


def load_data_from_csv():
    # Загрузка данных из CSV, если файл существует
    if os.path.exists("data_2025.csv"):
        df = pd.read_csv("data_2025.csv")
        # Загружаем данные в session_state
        st.session_state.data_2025_real = {
            "date": df["date"].tolist(),
            "avg_pullups": df["avg_pullups"].tolist(),
        }
    else:
        st.session_state.data_2025_real = {
            "date": ["14.01.2025", "18.01.2025"],
            "avg_pullups": [5, 3.6],
        }


st.title("Прогноз подтягиваний с использованием полиномиальной регрессии")

# Загружаем данные из CSV или инициализируем пустое состояние
load_data_from_csv()

st.subheader("Таблица данных для 2025 года:")

df_2025_real = pd.DataFrame(st.session_state.data_2025_real)
edited_df_2025 = st.dataframe(df_2025_real)

# Добавление новых данных
with st.form(key="add_data_form"):
    st.subheader("Добавить новые данные")
    new_date = st.date_input("Дата", value=datetime(2025, 1, 14))
    new_avg_pullups = st.number_input(
        "Среднее число подтягиваний", min_value=0.0, step=0.1, format="%.1f", value=0.0
    )

    submit_button = st.form_submit_button(label="Добавить данные")

    if submit_button:
        if new_date and new_avg_pullups >= 0:
            new_date_str = new_date.strftime("%d.%m.%Y")
            st.session_state.data_2025_real["date"].append(new_date_str)
            st.session_state.data_2025_real["avg_pullups"].append(new_avg_pullups)
            save_data_to_csv()  # Логируем изменения в CSV
            st.success(f"Данные добавлены: {new_date_str}, {new_avg_pullups}")
        else:
            st.error("Пожалуйста, заполните все поля корректно.")

# Данные 2021 года
data_2021 = {
    "date": [
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

# Преобразуем даты 2021 года в число дней с начала периода
start_date_2021 = datetime.strptime("22.07.2021", "%d.%m.%Y")
data_2021["days"] = process_dates(data_2021["date"], start_date_2021)

# Подготовка данных для 2025 года
data_2025_real = pd.DataFrame(st.session_state.data_2025_real)
if len(data_2025_real) > 0:
    start_date_2025 = datetime.strptime(data_2025_real["date"].iloc[0], "%d.%m.%Y")
    data_2025_real["days"] = process_dates(data_2025_real["date"], start_date_2025)
else:
    data_2025_real["days"] = []

# Полиномиальная регрессия второй степени
X_2021 = np.array(data_2021["days"]).reshape(-1, 1)
y_2021 = np.array(data_2021["avg_pullups"])

# Применяем PolynomialFeatures для 2-й степени
poly = PolynomialFeatures(degree=2)
X_poly_2021 = poly.fit_transform(X_2021)

# Обучаем модель
model = LinearRegression()
model.fit(X_poly_2021, y_2021)

# Прогноз для 2025 года
if len(data_2025_real) > 0:
    last_day = data_2025_real["days"].iloc[-1]
    predict_days_2025 = np.arange(last_day, last_day + 60, 4)
else:
    predict_days_2025 = np.arange(0, 60, 4)

predict_days_2025_poly = poly.transform(predict_days_2025.reshape(-1, 1))
predicted_pullups_2025 = model.predict(predict_days_2025_poly)

# Даты для 2025 года
if len(data_2025_real) > 0:
    dates_2025 = [
        start_date_2025 + timedelta(days=int(day)) for day in predict_days_2025
    ]
else:
    dates_2025 = [
        datetime(2025, 1, 14) + timedelta(days=int(day)) for day in predict_days_2025
    ]

# График для 2021 года
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=data_2021["days"],
        y=data_2021["avg_pullups"],
        mode="markers+lines",
        name="Реальные данные 2021",
    )
)

fig.add_trace(
    go.Scatter(
        x=predict_days_2025,
        y=predicted_pullups_2025,
        mode="lines",
        name="Прогноз на 2025 год (60 дней)",
        line=dict(dash="dash"),
    )
)

fig.update_layout(
    title="Прогресс в подтягиваниях и прогноз для 2025 года",
    xaxis_title="Дни с начала периода",
    yaxis_title="Среднее число подтягиваний за подход",
    template="plotly_dark",
)

st.plotly_chart(fig)

# График для 2025 года
fig_2025 = go.Figure()

fig_2025.add_trace(
    go.Scatter(
        x=dates_2025,
        y=predicted_pullups_2025,
        mode="lines",
        name="Прогноз на 2025 год (60 дней)",
        line=dict(dash="dash"),
    )
)

if len(data_2025_real) > 0:
    fig_2025.add_trace(
        go.Scatter(
            x=dates_2025,
            y=data_2025_real["avg_pullups"],
            mode="markers+lines",
            name="Реальные данные 2025",
            line=dict(color="red"),
        )
    )

fig_2025.update_layout(
    title="Прогноз и реальные данные для 2025 года",
    xaxis_title="Дата",
    yaxis_title="Среднее число подтягиваний за подход",
    xaxis=dict(tickformat="%d.%m.%Y"),
    template="plotly_dark",
)

st.plotly_chart(fig_2025)

# Расчет ошибки MAE для 2021 года
X_2021_pred = np.array(data_2021["days"]).reshape(-1, 1)
X_2021_pred_poly = poly.transform(X_2021_pred)
predicted_2021 = model.predict(X_2021_pred_poly)

mae_2021 = mean_absolute_error(y_2021, predicted_2021)
st.subheader(f"Ошибка прогноза (MAE) для 2021 года: {mae_2021:.2f}")

# Расчет ошибки MAE для 2025 года, только для тех данных, которые существуют
if len(data_2025_real) > 0:
    # Фильтрация данных для 2025 года, где есть реальные значения
    real_dates_2025 = data_2025_real[data_2025_real["avg_pullups"].notna()]
    predicted_2025_filtered = predicted_pullups_2025[: len(real_dates_2025)]
    y_2025_real_filtered = real_dates_2025["avg_pullups"].values

    mae_2025 = mean_absolute_error(y_2025_real_filtered, predicted_2025_filtered)
    st.subheader(f"Ошибка прогноза (MAE) для 2025 года: {mae_2025:.2f}")
else:
    st.subheader(
        "Ошибка прогноза для 2025 года не может быть рассчитана, так как нет реальных данных."
    )
