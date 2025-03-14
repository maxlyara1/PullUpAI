# backend/main.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from .services import (
    get_prediction_data,
    add_training_data,
    delete_training_data,
    reset_training_data,
    get_all_training_data,
    update_training_data,  #  Импортируем update_training_data!
)
from .models import (
    TrainingData,
    PredictionData,
)  # Assuming you have PredictionData model
from .database import DataError, get_current_data_filename, load_original_standards


import logging

# Базовая конфигурация логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

app = FastAPI()

origins = [
    "http://localhost:3000",  # Фронтенд (React)
    "http://localhost:8000",  # Бэкенд (FastAPI)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


#  Обработчик *своих* исключений DataError
@app.exception_handler(DataError)
async def data_error_handler(request: Request, exc: DataError):
    return JSONResponse(
        status_code=500,
        content={"message": f"Data error: {exc}"},
    )


@app.get("/api/prediction")
async def index(weight_category: str = "до 75", forecast_days: int = 90):
    try:
        data = await get_prediction_data(weight_category, forecast_days)
        return data
    except DataError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/data")
async def add_data(data: TrainingData):
    try:
        await add_training_data(data.date, data.avg_pullups, data.total_pullups)
        return {"message": "Data added successfully"}
    except DataError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/data/{index}")  # Добавили PUT endpoint
async def update_data_endpoint(index: int, data: TrainingData):
    try:
        await update_training_data(index, data)  # вызываем update_training_data
        return {"message": "Data updated successfully"}
    except DataError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except IndexError as e:
        raise HTTPException(
            status_code=404, detail=str(e)
        )  #  404 Not Found, если индекс неверный
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/data/{index}")
async def delete_data(index: int):
    try:
        await delete_training_data(index)
        return {"message": "Data deleted successfully"}
    except DataError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/reset")
async def reset_all_data():
    try:
        await reset_training_data()
        return {"message": "Data reset successfully"}
    except DataError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/download")
async def download_data():
    return FileResponse(
        get_current_data_filename(), filename="data_2025.csv", media_type="text/csv"
    )


@app.get("/api/history")
async def history():
    try:
        data = await get_all_training_data()
        return data
    except DataError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/original-standards")
async def get_original_standards():
    df = load_original_standards()
    if df.empty:
        raise HTTPException(status_code=404, detail="Original standards data not found")
    #  Преобразуем DataFrame в список словарей
    return df.to_dict(orient="records")
