# src/app/admin.py

import os
import sys
import asyncio
from fastapi import FastAPI
from fastapi_admin.factory import app as admin_app
from fastapi_admin.providers.login import UsernamePasswordProvider
from fastapi_admin.template import templates
from fastapi_admin.resources import Model, Field
from starlette.requests import Request
from sqlalchemy import Column, Integer, String, Boolean
from syft.server import uvicorn

from src.utils.database import SessionLocal, engine
from src.app.models import User  # Import your models here
from src.utils.database import Base

# Ensure the project directory is in the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define the Admin App
async def create_admin_app() -> FastAPI:
    """
    Creates and configures the FastAPI Admin application.
    """
    admin = FastAPI(title="Hermod Admin Interface")

    # Initialize FastAPI-Admin
    await admin_app.init(
        admin,
        admin_path="/admin",
        template_folders=[os.path.join(os.path.dirname(__file__), 'templates')],
        providers=[
            UsernamePasswordProvider(
                admin_model=User,
                login_logo_url="https://example.com/logo.png",
            )
        ],
        resources=[
            UserResource,
            # Add other resources (models) here
        ],
        engine=engine,
        session_maker=SessionLocal,
    )

    return admin

# Define the User Resource for Admin Interface
class UserResource(Model):
    label = "User"
    model = User
    page_size = 20  # Number of items per page

    # Fields to display in the admin interface
    fields = [
        "id",
        "username",
        "email",
        Field(name="is_active", label="Active", type="boolean"),
        Field(name="is_superuser", label="Superuser", type="boolean"),
    ]

# Entry point for running the admin app separately
def main():
    import uvicorn

    # Since we're in an async context, we need to run the event loop
    asyncio.run(run_admin())

async def run_admin():
    admin = await create_admin_app()
    uvicorn.run(admin, host="0.0.0.0", port=8001)

if __name__ == "__main__":
    main()
