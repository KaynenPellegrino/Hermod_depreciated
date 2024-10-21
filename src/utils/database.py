# src/utils/database.py

import os
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
from typing import Generator
from src.utils.config_loader import ConfigurationManager

# Load configuration
config_manager = ConfigurationManager()
DATABASE_URL = config_manager.get('database.url')
if not DATABASE_URL:
    # Construct the DATABASE_URL from individual components
    db_user = config_manager.get('database.user', 'user')
    db_password = config_manager.get('database.password', 'password')
    db_host = config_manager.get('database.host', 'localhost')
    db_port = config_manager.get('database.port', '5432')
    db_name = config_manager.get('database.name', 'hermod_db')
    db_type = config_manager.get('database.type', 'postgresql')
    DATABASE_URL = f"{db_type}://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

# Database connection pooling settings
POOL_SIZE = config_manager.get('database.pool_size', 5)
MAX_OVERFLOW = config_manager.get('database.max_overflow', 10)
POOL_TIMEOUT = config_manager.get('database.pool_timeout', 30)
POOL_RECYCLE = config_manager.get('database.pool_recycle', -1)
ECHO = config_manager.get('database.echo', False)

# Create the SQLAlchemy engine
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=POOL_SIZE,
    max_overflow=MAX_OVERFLOW,
    pool_timeout=POOL_TIMEOUT,
    pool_recycle=POOL_RECYCLE,
    echo=ECHO,
)

# Create a configured "Session" class
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Create a scoped session
SessionScoped = scoped_session(SessionLocal)

# Declarative base class
Base = declarative_base()

# Dependency for FastAPI routes
def get_db() -> Generator:
    """
    Dependency that provides a SQLAlchemy session to FastAPI endpoints.
    """
    db = SessionScoped()
    try:
        yield db
    finally:
        db.close()

# Event listener for handling connection checkouts
@event.listens_for(engine, "checkout")
def checkout_listener(dbapi_connection, connection_record, connection_proxy):
    """
    Event listener to handle disconnections.
    """
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("SELECT 1")
    except Exception as exc:
        # Discard the connection and raise an exception to trigger a retry
        connection_proxy._invalidate()
        raise exc
    finally:
        cursor.close()
