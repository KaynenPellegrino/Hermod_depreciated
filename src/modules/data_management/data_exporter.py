# data_management/data_exporter.py

import logging
import os
from typing import Dict, Any, Optional, Union, List

import joblib
import pandas as pd
import json
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from pymongo import MongoClient, errors as pymongo_errors
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging with RotatingFileHandler to prevent log files from becoming too large
from logging.handlers import RotatingFileHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a rotating file handler
handler = RotatingFileHandler('logs/hermod_data_exporter.log', maxBytes=10**6, backupCount=5)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)


class DataExporter:
    """
    Provides functionalities to export processed data and model outputs to various formats
    and external systems, facilitating their use outside of Hermod for analysis or reporting.
    """

    def __init__(self):
        """
        Initializes the DataExporter with necessary configurations.
        """
        self.database_engines = {}
        self.mongo_clients = {}
        self._initialize_database_connections()
        logger.info("DataExporter initialized successfully.")

    def _initialize_database_connections(self):
        """
        Initializes connections to SQL and NoSQL databases using environment variables.
        """
        # Initialize PostgreSQL connection
        postgres_host = os.getenv('POSTGRES_HOST')
        postgres_port = os.getenv('POSTGRES_PORT', '5432')
        postgres_db = os.getenv('POSTGRES_DB')
        postgres_user = os.getenv('POSTGRES_USER')
        postgres_password = os.getenv('POSTGRES_PASSWORD')

        if all([postgres_host, postgres_port, postgres_db, postgres_user, postgres_password]):
            try:
                postgres_engine = create_engine(f"postgresql+psycopg2://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_db}")
                self.database_engines['postgresql'] = postgres_engine
                logger.info("PostgreSQL engine initialized successfully.")
            except SQLAlchemyError as e:
                logger.error(f"Failed to initialize PostgreSQL engine: {e}")
        else:
            logger.warning("PostgreSQL credentials are incomplete. Skipping PostgreSQL initialization.")

        # Initialize MongoDB connection
        mongo_uri = os.getenv('MONGODB_URI')
        if mongo_uri:
            try:
                mongo_client = MongoClient(mongo_uri)
                # Test connection
                mongo_client.admin.command('ping')
                self.mongo_clients['mongodb'] = mongo_client
                logger.info("MongoDB client initialized successfully.")
            except pymongo_errors.ConnectionFailure as e:
                logger.error(f"Failed to connect to MongoDB: {e}")
        else:
            logger.warning("MongoDB URI not provided. Skipping MongoDB initialization.")

    def export_to_csv(self, df: pd.DataFrame, file_path: str, **kwargs):
        """
        Exports DataFrame to a CSV file.

        :param df: DataFrame to export
        :param file_path: Destination file path
        :param kwargs: Additional pandas.to_csv parameters
        """
        logger.info(f"Exporting data to CSV at '{file_path}'.")
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            df.to_csv(file_path, index=False, **kwargs)
            logger.info(f"Data exported to CSV successfully at '{file_path}'.")
        except Exception as e:
            logger.error(f"Failed to export data to CSV: {e}")

    def export_to_json(self, df: pd.DataFrame, file_path: str, orient: str = 'records', lines: bool = False, **kwargs):
        """
        Exports DataFrame to a JSON file.

        :param df: DataFrame to export
        :param file_path: Destination file path
        :param orient: Format of the JSON string
        :param lines: Whether to write JSON objects per line
        :param kwargs: Additional pandas.to_json parameters
        """
        logger.info(f"Exporting data to JSON at '{file_path}'.")
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            df.to_json(file_path, orient=orient, lines=lines, **kwargs)
            logger.info(f"Data exported to JSON successfully at '{file_path}'.")
        except Exception as e:
            logger.error(f"Failed to export data to JSON: {e}")

    def export_to_excel(self, df: pd.DataFrame, file_path: str, sheet_name: str = 'Sheet1', **kwargs):
        """
        Exports DataFrame to an Excel file.

        :param df: DataFrame to export
        :param file_path: Destination file path
        :param sheet_name: Name of the Excel sheet
        :param kwargs: Additional pandas.to_excel parameters
        """
        logger.info(f"Exporting data to Excel at '{file_path}'.")
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            df.to_excel(file_path, sheet_name=sheet_name, index=False, **kwargs)
            logger.info(f"Data exported to Excel successfully at '{file_path}'.")
        except Exception as e:
            logger.error(f"Failed to export data to Excel: {e}")

    def export_to_parquet(self, df: pd.DataFrame, file_path: str, **kwargs):
        """
        Exports DataFrame to a Parquet file.

        :param df: DataFrame to export
        :param file_path: Destination file path
        :param kwargs: Additional pandas.to_parquet parameters
        """
        logger.info(f"Exporting data to Parquet at '{file_path}'.")
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            df.to_parquet(file_path, index=False, **kwargs)
            logger.info(f"Data exported to Parquet successfully at '{file_path}'.")
        except Exception as e:
            logger.error(f"Failed to export data to Parquet: {e}")

    def export_to_sql(self, df: pd.DataFrame, table_name: str, db_type: str = 'postgresql', if_exists: str = 'append', **kwargs):
        """
        Exports DataFrame to a SQL database table.

        :param df: DataFrame to export
        :param table_name: Destination table name
        :param db_type: Type of database ('postgresql', etc.)
        :param if_exists: Behavior if the table already exists ('fail', 'replace', 'append')
        :param kwargs: Additional pandas.to_sql parameters
        """
        logger.info(f"Exporting data to SQL table '{table_name}' in '{db_type}' database.")
        engine = self.database_engines.get(db_type.lower())
        if not engine:
            logger.error(f"No engine found for database type '{db_type}'. Ensure it is initialized properly.")
            return

        try:
            df.to_sql(table_name, engine, if_exists=if_exists, index=False, **kwargs)
            logger.info(f"Data exported to SQL table '{table_name}' successfully in '{db_type}' database.")
        except SQLAlchemyError as e:
            logger.error(f"Failed to export data to SQL table '{table_name}': {e}")

    def export_to_mongodb(self, df: pd.DataFrame, db_name: str, collection_name: str, if_exists: str = 'append', **kwargs):
        """
        Exports DataFrame to a MongoDB collection.

        :param df: DataFrame to export
        :param db_name: Name of the MongoDB database
        :param collection_name: Name of the MongoDB collection
        :param if_exists: Behavior if the collection already exists ('fail', 'replace', 'append')
        :param kwargs: Additional parameters for MongoDB insertion
        """
        logger.info(f"Exporting data to MongoDB collection '{collection_name}' in database '{db_name}'.")
        mongo_client = self.mongo_clients.get('mongodb')
        if not mongo_client:
            logger.error("MongoDB client is not initialized. Ensure MongoDB URI is provided and connection is successful.")
            return

        try:
            db = mongo_client[db_name]
            collection = db[collection_name]
            if if_exists == 'replace':
                collection.drop()
                logger.info(f"Existing collection '{collection_name}' dropped.")
            records = df.to_dict(orient='records')
            collection.insert_many(records)
            logger.info(f"Data exported to MongoDB collection '{collection_name}' successfully in database '{db_name}'.")
        except pymongo_errors.PyMongoError as e:
            logger.error(f"Failed to export data to MongoDB collection '{collection_name}': {e}")

    def export_to_api(self, data: Union[pd.DataFrame, Dict[str, Any]], api_endpoint: str, method: str = 'POST', headers: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Exports data to an external API.

        :param data: Data to export (DataFrame or dictionary)
        :param api_endpoint: URL of the API endpoint
        :param method: HTTP method ('POST', 'PUT', etc.)
        :param headers: Optional HTTP headers
        :param kwargs: Additional parameters for the request
        """
        logger.info(f"Exporting data to API endpoint '{api_endpoint}' using method '{method}'.")
        if isinstance(data, pd.DataFrame):
            payload = data.to_dict(orient='records')
        elif isinstance(data, dict):
            payload = data
        else:
            logger.error("Data must be a pandas DataFrame or a dictionary.")
            return

        try:
            response = requests.request(method, api_endpoint, json=payload, headers=headers, timeout=30, **kwargs)
            response.raise_for_status()
            logger.info(f"Data exported to API endpoint '{api_endpoint}' successfully. Response: {response.text}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to export data to API endpoint '{api_endpoint}': {e}")

    def export_model(self, model: Any, file_path: str, **kwargs):
        """
        Exports a trained model to a file using joblib.

        :param model: Trained machine learning model
        :param file_path: Destination file path
        :param kwargs: Additional joblib.dump parameters
        """
        logger.info(f"Exporting model to '{file_path}'.")
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            joblib.dump(model, file_path, **kwargs)
            logger.info(f"Model exported successfully to '{file_path}'.")
        except Exception as e:
            logger.error(f"Failed to export model to '{file_path}': {e}")

    def export_all(self, df: pd.DataFrame, export_config: Dict[str, Any]):
        """
        Exports data based on a configuration dictionary.

        :param df: DataFrame to export
        :param export_config: Dictionary specifying export actions
        """
        logger.info("Starting bulk data export based on configuration.")
        for export_type, params in export_config.items():
            try:
                if export_type == 'csv':
                    self.export_to_csv(df, params['file_path'], **params.get('kwargs', {}))
                elif export_type == 'json':
                    self.export_to_json(df, params['file_path'], **params.get('kwargs', {}))
                elif export_type == 'excel':
                    self.export_to_excel(df, params['file_path'], **params.get('kwargs', {}))
                elif export_type == 'parquet':
                    self.export_to_parquet(df, params['file_path'], **params.get('kwargs', {}))
                elif export_type == 'sql':
                    self.export_to_sql(df, params['table_name'], db_type=params.get('db_type', 'postgresql'), if_exists=params.get('if_exists', 'append'), **params.get('kwargs', {}))
                elif export_type == 'mongodb':
                    self.export_to_mongodb(df, params['db_name'], params['collection_name'], if_exists=params.get('if_exists', 'append'), **params.get('kwargs', {}))
                elif export_type == 'api':
                    self.export_to_api(df, params['api_endpoint'], method=params.get('method', 'POST'), headers=params.get('headers'), **params.get('kwargs', {}))
                elif export_type == 'model':
                    self.export_model(params['model'], params['file_path'], **params.get('kwargs', {}))
                else:
                    logger.warning(f"Unsupported export type: {export_type}")
            except KeyError as e:
                logger.error(f"Missing required parameter for export type '{export_type}': {e}")
            except Exception as e:
                logger.error(f"Error during export type '{export_type}': {e}")

        logger.info("Bulk data export completed.")

    # Example usage and test cases
if __name__ == "__main__":
    # Initialize DataExporter
    exporter = DataExporter()

    # Example DataFrame to export
    data = {
        'id': [1, 2, 3],
        'name': ['Alice', 'Bob', 'Charlie'],
        'score': [85.5, 92.3, 78.9]
    }
    df = pd.DataFrame(data)

    # Define export configurations
    export_config = {
        'csv': {
            'file_path': 'data/exported_data/output_data.csv',
            'kwargs': {'sep': ',', 'encoding': 'utf-8'}
        },
        'json': {
            'file_path': 'data/exported_data/output_data.json',
            'kwargs': {'orient': 'records', 'lines': True}
        },
        'excel': {
            'file_path': 'data/exported_data/output_data.xlsx',
            'kwargs': {'sheet_name': 'Data'}
        },
        'parquet': {
            'file_path': 'data/exported_data/output_data.parquet',
            'kwargs': {}
        },
        'sql': {
            'table_name': 'exported_scores',
            'db_type': 'postgresql',
            'if_exists': 'replace',
            'kwargs': {}
        },
        'mongodb': {
            'db_name': 'hermod_export_db',
            'collection_name': 'exported_scores',
            'if_exists': 'replace',
            'kwargs': {}
        },
        'api': {
            'api_endpoint': 'https://example.com/api/data',
            'method': 'POST',
            'headers': {'Authorization': 'Bearer YOUR_API_TOKEN'},
            'kwargs': {}
        },
        'model': {
            'model': 'dummy_model',  # Replace with an actual trained model object
            'file_path': 'model/exported_models/trained_model.joblib',
            'kwargs': {}
        }
    }

    # Perform bulk export
    exporter.export_all(df, export_config)

    # Example: Export a single DataFrame to CSV
    exporter.export_to_csv(df, 'data/exported_data/single_output.csv', sep=',', encoding='utf-8')

    # Example: Export a single DataFrame to a PostgreSQL table
    exporter.export_to_sql(df, table_name='single_exported_scores', db_type='postgresql', if_exists='append')

    # Example: Export a trained model to a file
    # Assuming 'trained_pipeline' is a trained scikit-learn pipeline
    # trained_pipeline = ...  # Obtain the trained model from data_trainer.py
    # exporter.export_model(trained_pipeline, 'model/exported_models/trained_pipeline.joblib')
