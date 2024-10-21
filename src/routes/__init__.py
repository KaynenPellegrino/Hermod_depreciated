# src/routes/__init__.py

from fastapi import APIRouter

api_router = APIRouter()

@api_router.get("/")
async def root():
    return {"message": "Welcome to Hermod AI Assistant"}
