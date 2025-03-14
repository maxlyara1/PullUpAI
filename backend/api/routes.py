from fastapi import APIRouter
from backend.api.endpoints import training
from backend.core.config import API_PREFIX

api_router = APIRouter()

api_router.include_router(training.router, prefix=API_PREFIX, tags=["training"])
