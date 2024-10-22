#!/usr/bin/env python3
"""
real_time_feedback.py

Function: Real-Time Feedback System
Purpose: Handles real-time feedback, allowing users to query Hermod about performance metrics or actions and get immediate responses.
"""

import os
import sys
import yaml
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, text
import pandas as pd
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()


# ----------------------------
# Configuration and Logging
# ----------------------------

def load_config(config_path='config.yaml'):
    """
    Load configuration from a YAML file.
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logging.error(f"Failed to load configuration file: {e}")
        sys.exit(1)


def setup_logging(log_dir='logs'):
    """
    Setup logging configuration.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'real_time_feedback_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


# ----------------------------
# Database Setup
# ----------------------------

class Database:
    """
    Database connection handler using SQLAlchemy.
    """

    def __init__(self, db_config):
        self.db_config = db_config
        self.engine = self.create_db_engine()

    def create_db_engine(self):
        """
        Create a SQLAlchemy engine based on configuration.
        """
        try:
            db_url = f"{self.db_config['dialect']}://{self.db_config['username']}:{self.db_config['password']}@" \
                     f"{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
            engine = create_engine(db_url, pool_pre_ping=True)
            logging.info("Database engine created successfully.")
            return engine
        except Exception as e:
            logging.error(f"Failed to create database engine: {e}")
            sys.exit(1)

    def query(self, query_text):
        """
        Execute a raw SQL query and return results as a DataFrame.
        """
        try:
            with self.engine.connect() as connection:
                result = connection.execute(text(query_text))
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
            logging.info(f"Executed query successfully: {query_text}")
            return df
        except Exception as e:
            logging.error(f"Error executing query: {e}")
            return pd.DataFrame()


# ----------------------------
# FastAPI Application Setup
# ----------------------------

app = FastAPI(
    title="Hermod AI Assistant Real-Time Feedback API",
    description="API for querying system performance metrics and actions in real-time.",
    version="1.0.0"
)


# ----------------------------
# Pydantic Models
# ----------------------------

class PerformanceMetrics(BaseModel):
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    bytes_sent: int
    bytes_recv: int


class ApplicationMetrics(BaseModel):
    timestamp: datetime
    endpoint: str
    method: str
    response_time_sec: Optional[float]
    status_code: Optional[int]
    success: bool


class Action(BaseModel):
    timestamp: datetime
    action_name: str
    details: Optional[str]


class QueryResponse(BaseModel):
    performance_metrics: Optional[List[PerformanceMetrics]]
    application_metrics: Optional[List[ApplicationMetrics]]
    actions: Optional[List[Action]]


# ----------------------------
# Load Configuration and Initialize DB
# ----------------------------

config = load_config()
setup_logging(config.get('log_dir', 'logs'))

# Initialize Database connections
db_config = config.get('database')
if not db_config:
    logging.error("Database configuration not found in config.yaml.")
    sys.exit(1)

db = Database(db_config)


# ----------------------------
# API Endpoints
# ----------------------------

@app.get("/metrics/system", response_model=List[PerformanceMetrics])
def get_system_metrics(start_time: Optional[datetime] = Query(None, description="Start time for metrics"),
                       end_time: Optional[datetime] = Query(None, description="End time for metrics"),
                       limit: Optional[int] = Query(100, description="Number of records to return")):
    """
    Retrieve system performance metrics.
    """
    try:
        query = "SELECT * FROM system_metrics"
        conditions = []
        if start_time:
            conditions.append(f"timestamp >= '{start_time}'")
        if end_time:
            conditions.append(f"timestamp <= '{end_time}'")
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += f" ORDER BY timestamp DESC LIMIT {limit}"

        df = db.query(query)
        if df.empty:
            raise HTTPException(status_code=404, detail="No system metrics found for the given parameters.")

        metrics = df.to_dict(orient='records')
        return metrics
    except HTTPException as he:
        raise he
    except Exception as e:
        logging.error(f"Error in get_system_metrics: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.get("/metrics/application", response_model=List[ApplicationMetrics])
def get_application_metrics(endpoint: Optional[str] = Query(None, description="Specific endpoint to filter"),
                            success: Optional[bool] = Query(None, description="Filter by success status"),
                            start_time: Optional[datetime] = Query(None, description="Start time for metrics"),
                            end_time: Optional[datetime] = Query(None, description="End time for metrics"),
                            limit: Optional[int] = Query(100, description="Number of records to return")):
    """
    Retrieve application performance metrics by endpoint.
    """
    try:
        query = "SELECT * FROM application_metrics"
        conditions = []
        if endpoint:
            conditions.append(f"endpoint = '{endpoint}'")
        if success is not None:
            conditions.append(f"success = {'TRUE' if success else 'FALSE'}")
        if start_time:
            conditions.append(f"timestamp >= '{start_time}'")
        if end_time:
            conditions.append(f"timestamp <= '{end_time}'")
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += f" ORDER BY timestamp DESC LIMIT {limit}"

        df = db.query(query)
        if df.empty:
            raise HTTPException(status_code=404, detail="No application metrics found for the given parameters.")

        metrics = df.to_dict(orient='records')
        return metrics
    except HTTPException as he:
        raise he
    except Exception as e:
        logging.error(f"Error in get_application_metrics: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.get("/actions", response_model=List[Action])
def get_actions(start_time: Optional[datetime] = Query(None, description="Start time for actions"),
                end_time: Optional[datetime] = Query(None, description="End time for actions"),
                action_name: Optional[str] = Query(None, description="Filter by action name"),
                limit: Optional[int] = Query(100, description="Number of records to return")):
    """
    Retrieve actions taken based on feedback or performance metrics.
    """
    try:
        query = "SELECT * FROM actions"
        conditions = []
        if action_name:
            conditions.append(f"action_name = '{action_name}'")
        if start_time:
            conditions.append(f"timestamp >= '{start_time}'")
        if end_time:
            conditions.append(f"timestamp <= '{end_time}'")
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += f" ORDER BY timestamp DESC LIMIT {limit}"

        df = db.query(query)
        if df.empty:
            raise HTTPException(status_code=404, detail="No actions found for the given parameters.")

        actions = df.to_dict(orient='records')
        return actions
    except HTTPException as he:
        raise he
    except Exception as e:
        logging.error(f"Error in get_actions: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.get("/query", response_model=QueryResponse)
def query_feedback(performance: Optional[bool] = Query(False, description="Include performance metrics"),
                   application: Optional[bool] = Query(False, description="Include application metrics"),
                   actions: Optional[bool] = Query(False, description="Include actions"),
                   start_time: Optional[datetime] = Query(None, description="Start time for data"),
                   end_time: Optional[datetime] = Query(None, description="End time for data"),
                   limit: Optional[int] = Query(100, description="Number of records to return")):
    """
    General query endpoint to retrieve performance metrics, application metrics, and actions.
    """
    try:
        response = {}
        if performance:
            query = "SELECT * FROM system_metrics"
            conditions = []
            if start_time:
                conditions.append(f"timestamp >= '{start_time}'")
            if end_time:
                conditions.append(f"timestamp <= '{end_time}'")
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            query += f" ORDER BY timestamp DESC LIMIT {limit}"

            df = db.query(query)
            if not df.empty:
                response['performance_metrics'] = df.to_dict(orient='records')
            else:
                response['performance_metrics'] = []

        if application:
            query = "SELECT * FROM application_metrics"
            conditions = []
            if start_time:
                conditions.append(f"timestamp >= '{start_time}'")
            if end_time:
                conditions.append(f"timestamp <= '{end_time}'")
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            query += f" ORDER BY timestamp DESC LIMIT {limit}"

            df = db.query(query)
            if not df.empty:
                response['application_metrics'] = df.to_dict(orient='records')
            else:
                response['application_metrics'] = []

        if actions:
            query = "SELECT * FROM actions"
            conditions = []
            if start_time:
                conditions.append(f"timestamp >= '{start_time}'")
            if end_time:
                conditions.append(f"timestamp <= '{end_time}'")
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            query += f" ORDER BY timestamp DESC LIMIT {limit}"

            df = db.query(query)
            if not df.empty:
                response['actions'] = df.to_dict(orient='records')
            else:
                response['actions'] = []

        return QueryResponse(**response)
    except Exception as e:
        logging.error(f"Error in query_feedback: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


# ----------------------------
# Main Function
# ----------------------------

def main():
    """
    Entry point for the Real-Time Feedback System.
    """
    import uvicorn
    # Run the FastAPI app using Uvicorn
    uvicorn.run("real_time_feedback:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
