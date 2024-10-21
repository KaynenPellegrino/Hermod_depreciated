# src/asgi.py

import os
import sys
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from src.main import app

# Add the project directory to the sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Middleware and configurations can be added here if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for your security needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ASGI application callable
application = app
