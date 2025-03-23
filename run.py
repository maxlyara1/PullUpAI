"""
Скрипт запуска FastAPI приложения с использованием Uvicorn.

Запускает сервер на порту 8000 с поддержкой автоматической перезагрузки при изменении кода.
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
