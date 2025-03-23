# backend/main.py
"""
FastAPI приложение для прогнозирования и управления данными о подтягиваниях.
Предоставляет API endpoints для получения прогнозов, добавления, обновления и удаления данных.
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from .services import (
    get_prediction_data,
    add_training_data,
    delete_training_data,
    reset_training_data,
    get_all_training_data,
    update_training_data,
)
from .services.prediction import MODEL_CACHE
from .models import (
    TrainingData,
    PredictionData,
)
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

# Middleware для сброса кэша моделей при каждом обращении к API
@app.middleware("http")
async def reset_model_cache_middleware(request: Request, call_next):
    """Middleware для сброса кэша моделей при обращении к API прогнозов или данных."""
    # Сбрасываем кэш для всех запросов, связанных с данными или прогнозом
    if "/api/prediction" in request.url.path or "/api/data" in request.url.path:
        MODEL_CACHE.clear()
        logging.info(f"Кэш моделей сброшен при запросе {request.url.path}")
    
    response = await call_next(request)
    return response

#  Обработчик *своих* исключений DataError
@app.exception_handler(DataError)
async def data_error_handler(request: Request, exc: DataError):
    """Обработчик исключений DataError."""
    return JSONResponse(
        status_code=500,
        content={"message": f"Data error: {exc}"},
    )


@app.get("/api/prediction")
async def index(weight_category: str = "до 75", forecast_days: int = 90):
    """
    Получить прогноз подтягиваний на основе имеющихся данных.
    
    Args:
        weight_category: Весовая категория для прогноза
        forecast_days: Количество дней для прогноза
        
    Returns:
        Данные с прогнозом
    """
    try:
        data = await get_prediction_data(weight_category, forecast_days)
        return data
    except DataError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/data")
async def add_data_endpoint(data: TrainingData):
    """
    Добавить новые данные о тренировке.
    
    Args:
        data: Данные о тренировке
        
    Returns:
        Сообщение об успешном добавлении
    """
    try:
        await add_training_data(data.date, data.avg_pullups, data.total_pullups)
        # Сбрасываем кэш моделей при добавлении новых данных
        MODEL_CACHE.clear()
        return {"message": "Data added successfully"}
    except DataError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/data/{index}")  # Добавили PUT endpoint
async def update_data_endpoint(index: int, data: TrainingData):
    """
    Обновить данные о тренировке по индексу.
    
    Args:
        index: Индекс записи для обновления
        data: Новые данные о тренировке
        
    Returns:
        Сообщение об успешном обновлении
    """
    try:
        await update_training_data(index, data)  # вызываем update_training_data
        # Сбрасываем кэш моделей при обновлении данных
        MODEL_CACHE.clear()
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
async def delete_data_endpoint(index: int):
    """
    Удалить данные о тренировке по индексу.
    
    Args:
        index: Индекс записи для удаления
        
    Returns:
        Сообщение об успешном удалении
    """
    try:
        await delete_training_data(index)
        # Сбрасываем кэш моделей при удалении данных
        MODEL_CACHE.clear()
        return {"message": "Data deleted successfully"}
    except DataError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/reset")
async def reset_data_endpoint():
    """
    Сбросить все данные о тренировках.
    
    Returns:
        Сообщение об успешном сбросе
    """
    try:
        await reset_training_data()
        # Сбрасываем кэш моделей при сбросе всех данных
        MODEL_CACHE.clear()
        return {"message": "Data has been reset"}
    except DataError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/download")
async def download_data():
    """
    Скачать текущие данные в формате CSV.
    
    Returns:
        CSV файл с данными
    """
    return FileResponse(
        get_current_data_filename(), filename="data_2025.csv", media_type="text/csv"
    )


@app.get("/api/history")
async def history():
    """
    Получить все данные о тренировках.
    
    Returns:
        Исторические данные о тренировках
    """
    try:
        data = await get_all_training_data()
        return data
    except DataError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/original-standards")
async def get_original_standards():
    """
    Получить оригинальные стандарты подтягиваний.
    
    Returns:
        Стандарты подтягиваний в разных весовых категориях
    """
    df = load_original_standards()
    if df.empty:
        raise HTTPException(status_code=404, detail="Original standards data not found")
    #  Преобразуем DataFrame в список словарей
    return df.to_dict(orient="records")
