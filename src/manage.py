#!/usr/bin/env python

# manage.py

import typer
import uvicorn
import os
import sys
from alembic import command
from alembic.config import Config

# Ensure the project directory is in the sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the application
from src.main import app

# Import database and models
from src.utils.database import engine, Base
import src.app.models  # Import models to register them with SQLAlchemy

cli = typer.Typer(help="Management script for the Hermod AI assistant framework.")

@cli.command()
def runserver(host: str = "127.0.0.1", port: int = 8000, reload: bool = True):
    """
    Runs the FastAPI development server.
    """
    uvicorn.run("src.main:app", host=host, port=port, reload=reload)

@cli.command()
def init_db():
    """
    Initializes the database by creating all tables.
    """
    # Import models to register them with SQLAlchemy
    import src.app.models

    Base.metadata.create_all(bind=engine)
    typer.echo("Database initialized.")

@cli.command()
def makemigrations(message: str = "Auto migration"):
    """
    Generates a new migration script.
    """
    alembic_cfg = Config("alembic.ini")
    command.revision(alembic_cfg, autogenerate=True, message=message)
    typer.echo("New migration script generated.")

@cli.command()
def migrate():
    """
    Runs database migrations using Alembic.
    """
    alembic_cfg = Config("alembic.ini")
    command.upgrade(alembic_cfg, "head")
    typer.echo("Database migrations executed.")

@cli.command()
def createsuperuser():
    """
    Creates a superuser/admin account.
    """
    from getpass import getpass
    from src.app.models import User  # Adjusted import based on your code structure
    from src.utils.database import SessionLocal

    db = SessionLocal()
    username = typer.prompt("Username")
    email = typer.prompt("Email")
    password = getpass("Password: ")
    confirm_password = getpass("Confirm Password: ")

    if password != confirm_password:
        typer.echo("Passwords do not match.")
        sys.exit(1)

    # Hash the password
    hashed_password = hash_password(password)

    # Check if user already exists
    existing_user = db.query(User).filter(User.username == username).first()
    if existing_user:
        typer.echo(f"User '{username}' already exists.")
        sys.exit(1)

    # Create the user
    user = User(
        username=username,
        email=email,
        hashed_password=hashed_password,
        is_superuser=True,
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.close()

    typer.echo(f"Superuser '{username}' created successfully.")

def hash_password(password: str) -> str:
    """
    Hashes a password using bcrypt.
    """
    import bcrypt

    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

if __name__ == "__main__":
    cli()
