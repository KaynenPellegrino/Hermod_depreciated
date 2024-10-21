# src/wsgi.py

import os
import sys
from asgiref.wsgi import AsgiToWsgi

# Add the project directory to the sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the FastAPI application
from src.main import app

# Wrap the ASGI application to create a WSGI application
application = AsgiToWsgi(app)
