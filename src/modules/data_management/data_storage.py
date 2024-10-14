# src/modules/data_management/data_storage.py
import base64
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional, Union, List
import pandas as pd
import json

import requests  # Corrected import
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from pymongo import MongoClient, errors as pymongo_errors
import joblib
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging with RotatingFileHandler to prevent log files from becoming too large
from logging.handlers import RotatingFileHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a rotating file handler
handler = RotatingFileHandler('logs/hermod_data_storage.log', maxBytes=10 ** 6, backupCount=5)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)


class DataStorage:
    """
    Manages persistent data storage within Hermod, handling databases and file storage systems.
    Provides methods to save and retrieve data securely and efficiently.
    """

    def __init__(self):
        """
        Initializes the DataStorage with necessary configurations.
        """
        self.database_engines = {}
        self.mongo_clients = {}
        self._initialize_database_connections()
        logger.info("DataStorage initialized successfully.")

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
                postgres_engine = create_engine(
                    f"postgresql+psycopg2://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_db}"
                )
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

    def save_to_sql(self, df: pd.DataFrame, table_name: str, db_type: str = 'postgresql', if_exists: str = 'append',
                   **kwargs):
        """
        Saves a DataFrame to a SQL database table.

        :param df: DataFrame to save
        :param table_name: Name of the target table
        :param db_type: Type of database ('postgresql', etc.)
        :param if_exists: Behavior if the table already exists ('fail', 'replace', 'append')
        :param kwargs: Additional parameters for pandas.to_sql
        """
        logger.info(f"Saving DataFrame to SQL table '{table_name}' in '{db_type}' database.")
        engine = self.database_engines.get(db_type.lower())
        if not engine:
            logger.error(f"No engine found for database type '{db_type}'. Ensure it is initialized properly.")
            return

        try:
            # Ensure raw data is saved in the designated raw directory
            raw_data_dir = os.path.join('src', 'modules', 'data_management', 'datasets', 'raw', 'behavioral')
            os.makedirs(raw_data_dir, exist_ok=True)
            raw_file_path = os.path.join(raw_data_dir, f"{table_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            df.to_csv(raw_file_path, index=False)
            logger.info(f"Raw DataFrame saved to '{raw_file_path}' successfully.")

            # Save to SQL
            df.to_sql(table_name, engine, if_exists=if_exists, index=False, **kwargs)
            logger.info(f"DataFrame saved to SQL table '{table_name}' successfully in '{db_type}' database.")
        except SQLAlchemyError as e:
            logger.error(f"Failed to save DataFrame to SQL table '{table_name}': {e}")

    def load_from_sql(self, query: str, db_type: str = 'postgresql') -> Optional[pd.DataFrame]:
        """
        Loads data from a SQL database using a query.

        :param query: SQL query to execute
        :param db_type: Type of database ('postgresql', etc.)
        :return: DataFrame containing the query results or None if failed
        """
        logger.info(f"Loading data from SQL database '{db_type}' with query: {query}")
        engine = self.database_engines.get(db_type.lower())
        if not engine:
            logger.error(f"No engine found for database type '{db_type}'. Ensure it is initialized properly.")
            return None

        try:
            df = pd.read_sql_query(query, engine)
            logger.info(f"Data loaded successfully from SQL database '{db_type}'.")
            return df
        except SQLAlchemyError as e:
            logger.error(f"Failed to load data from SQL database '{db_type}': {e}")
            return None

    def save_to_mongodb(self, df: pd.DataFrame, db_name: str, collection_name: str, if_exists: str = 'append',
                        **kwargs):
        """
        Saves a DataFrame to a MongoDB collection.

        :param df: DataFrame to save
        :param db_name: Name of the MongoDB database
        :param collection_name: Name of the MongoDB collection
        :param if_exists: Behavior if the collection already exists ('fail', 'replace', 'append')
        :param kwargs: Additional parameters for pymongo insert operations
        """
        logger.info(f"Saving DataFrame to MongoDB collection '{collection_name}' in database '{db_name}'.")
        mongo_client = self.mongo_clients.get('mongodb')
        if not mongo_client:
            logger.error(
                "MongoDB client is not initialized. Ensure MongoDB URI is provided and connection is successful.")
            return

        try:
            db = mongo_client[db_name]
            collection = db[collection_name]
            if if_exists == 'replace':
                collection.drop()
                logger.info(f"Existing collection '{collection_name}' dropped.")

            records = df.to_dict(orient='records')
            collection.insert_many(records, **kwargs)
            logger.info(
                f"Successfully stored data to MongoDB collection '{collection_name}' in database '{db_name}'.")
        except pymongo_errors.PyMongoError as e:
            logger.error(f"Failed to save DataFrame to MongoDB collection '{collection_name}': {e}")

    def load_from_mongodb(self, db_name: str, collection_name: str, query: Optional[Dict[str, Any]] = None) -> Optional[
        pd.DataFrame]:
        """
        Loads data from a MongoDB collection using a query.

        :param db_name: Name of the MongoDB database
        :param collection_name: Name of the MongoDB collection
        :param query: MongoDB query filter
        :return: DataFrame containing the query results or None if failed
        """
        logger.info(
            f"Loading data from MongoDB collection '{collection_name}' in database '{db_name}' with query: {query}")
        mongo_client = self.mongo_clients.get('mongodb')
        if not mongo_client:
            logger.error(
                "MongoDB client is not initialized. Ensure MongoDB URI is provided and connection is successful.")
            return None

        try:
            db = mongo_client[db_name]
            collection = db[collection_name]
            cursor = collection.find(query or {})
            data = list(cursor)
            if not data:
                logger.warning(f"No data found in MongoDB collection '{collection_name}' with query {query}.")
                return pd.DataFrame()
            df = pd.DataFrame(data)
            logger.info(
                f"Data loaded successfully from MongoDB collection '{collection_name}' in database '{db_name}'.")
            return df
        except pymongo_errors.PyMongoError as e:
            logger.error(f"Failed to load data from MongoDB collection '{collection_name}': {e}")
            return None

    def save_to_file(self, df: pd.DataFrame, file_path: str, format: str = 'csv', **kwargs):
        """
        Saves a DataFrame to a file in the specified format.

        :param df: DataFrame to save
        :param file_path: Destination file path
        :param format: File format ('csv', 'json', 'excel', 'parquet')
        :param kwargs: Additional parameters for pandas export functions
        """
        logger.info(f"Saving DataFrame to file '{file_path}' in '{format}' format.")
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            if format.lower() == 'csv':
                df.to_csv(file_path, index=False, **kwargs)
            elif format.lower() == 'json':
                df.to_json(file_path, orient='records', lines=True, **kwargs)
            elif format.lower() in ['xls', 'xlsx']:
                df.to_excel(file_path, index=False, **kwargs)
            elif format.lower() == 'parquet':
                df.to_parquet(file_path, index=False, **kwargs)
            else:
                logger.error(f"Unsupported file format for storage: {format}")
                raise ValueError(f"Unsupported file format for storage: {format}")
            logger.info(f"DataFrame saved to file '{file_path}' successfully.")
        except Exception as e:
            logger.error(f"Failed to save DataFrame to file '{file_path}': {e}")

    def load_from_file(self, file_path: str, format: str = 'csv', **kwargs) -> Optional[pd.DataFrame]:
        """
        Loads data from a file in the specified format into a DataFrame.

        :param file_path: Path to the data file
        :param format: File format ('csv', 'json', 'excel', 'parquet')
        :param kwargs: Additional parameters for pandas read functions
        :return: DataFrame containing the loaded data or None if failed
        """
        logger.info(f"Loading data from file '{file_path}' in '{format}' format.")
        try:
            if format.lower() == 'csv':
                df = pd.read_csv(file_path, **kwargs)
            elif format.lower() == 'json':
                df = pd.read_json(file_path, **kwargs)
            elif format.lower() in ['xls', 'xlsx']:
                df = pd.read_excel(file_path, **kwargs)
            elif format.lower() == 'parquet':
                df = pd.read_parquet(file_path, **kwargs)
            else:
                logger.error(f"Unsupported file format: {format}")
                return None
            logger.info(f"Data loaded successfully from file '{file_path}'.")
            return df
        except FileNotFoundError as e:
            logger.error(f"File not found: {file_path} - {e}")
            return None
        except pd.errors.ParserError as e:
            logger.error(f"Pandas parser error while reading file '{file_path}': {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to load data from file '{file_path}': {e}")
            return None

    def save_model(self, model: Any, file_path: str, **kwargs):
        """
        Saves a trained machine learning model to a file using joblib.

        :param model: Trained machine learning model
        :param file_path: Destination file path
        :param kwargs: Additional parameters for joblib.dump
        """
        logger.info(f"Saving model to file '{file_path}'.")
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            joblib.dump(model, file_path, **kwargs)
            logger.info(f"Model saved successfully to '{file_path}'.")
        except Exception as e:
            logger.error(f"Failed to save model to '{file_path}': {e}")

    def load_model(self, file_path: str) -> Optional[Any]:
        """
        Loads a trained machine learning model from a file using joblib.

        :param file_path: Path to the model file
        :return: Loaded model object or None if failed
        """
        logger.info(f"Loading model from file '{file_path}'.")
        try:
            model = joblib.load(file_path)
            logger.info(f"Model loaded successfully from '{file_path}'.")
            return model
        except FileNotFoundError as e:
            logger.error(f"Model file not found: {file_path} - {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to load model from '{file_path}': {e}")
            return None

    def export_model_to_external_system(self, model: Any, api_endpoint: str, headers: Optional[Dict[str, Any]] = None,
                                        **kwargs):
        """
        Exports a trained model to an external system via API.

        :param model: Trained machine learning model
        :param api_endpoint: URL of the external system's API endpoint
        :param headers: Optional HTTP headers for the API request
        :param kwargs: Additional parameters for the request
        """
        logger.info(f"Exporting model to external API endpoint '{api_endpoint}'.")
        try:
            # Serialize the model to a byte stream
            model_bytes = joblib.dumps(model)
            # Encode the bytes in base64 to safely transmit binary data
            import base64
            model_b64 = base64.b64encode(model_bytes).decode('utf-8')
            payload = {'model': model_b64}
            response = requests.post(api_endpoint, json=payload, headers=headers, timeout=60, **kwargs)
            response.raise_for_status()
            logger.info(f"Model exported to external API '{api_endpoint}' successfully. Response: {response.text}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to export model to external API '{api_endpoint}': {e}")
        except Exception as e:
            logger.error(f"Error during model export to external API '{api_endpoint}': {e}")

    def retrieve_model_from_external_system(self, api_endpoint: str, headers: Optional[Dict[str, Any]] = None,
                                            **kwargs) -> Optional[Any]:
        """
        Retrieves a trained model from an external system via API.

        :param api_endpoint: URL of the external system's API endpoint
        :param headers: Optional HTTP headers for the API request
        :param kwargs: Additional parameters for the request
        :return: Loaded model object or None if failed
        """
        logger.info(f"Retrieving model from external API endpoint '{api_endpoint}'.")
        try:
            response = requests.get(api_endpoint, headers=headers, timeout=60, **kwargs)
            response.raise_for_status()
            data = response.json()
            model_b64 = data.get('model')
            if not model_b64:
                logger.error(f"No model data found in response from '{api_endpoint}'.")
                return None
            # Decode the model bytes from base64
            model_bytes = base64.b64decode(model_b64.encode('utf-8'))
            model = joblib.loads(model_bytes)
            logger.info(f"Model retrieved successfully from external API '{api_endpoint}'.")
            return model
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to retrieve model from external API '{api_endpoint}': {e}")
            return None
        except Exception as e:
            logger.error(f"Error during model retrieval from external API '{api_endpoint}': {e}")
            return None

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
                    self.save_to_file(df, params['file_path'], format='csv', **params.get('kwargs', {}))
                elif export_type == 'json':
                    self.save_to_file(df, params['file_path'], format='json', **params.get('kwargs', {}))
                elif export_type == 'excel':
                    self.save_to_file(df, params['file_path'], format='excel', **params.get('kwargs', {}))
                elif export_type == 'parquet':
                    self.save_to_file(df, params['file_path'], format='parquet', **params.get('kwargs', {}))
                elif export_type == 'sql':
                    self.save_to_sql(df, params['table_name'], db_type=params.get('db_type', 'postgresql'),
                                     if_exists=params.get('if_exists', 'append'), **params.get('kwargs', {}))
                elif export_type == 'mongodb':
                    self.save_to_mongodb(df, params['db_name'], params['collection_name'],
                                         if_exists=params.get('if_exists', 'append'), **params.get('kwargs', {}))
                elif export_type == 'api':
                    self.save_to_file(df, 'temp_export.json', format='json')  # Temporarily save data
                    with open('temp_export.json', 'r') as f:
                        data = json.load(f)
                    self.export_model_to_external_system(data, params['api_endpoint'], method=params.get('method', 'POST'),
                                                          headers=params.get('headers'), **params.get('kwargs', {}))
                    os.remove('temp_export.json')  # Clean up temporary file
                elif export_type == 'model':
                    self.save_model(params['model'], params['file_path'], **params.get('kwargs', {}))
                else:
                    logger.warning(f"Unsupported export type: {export_type}")
            except KeyError as e:
                logger.error(f"Missing required parameter for export type '{export_type}': {e}")
            except Exception as e:
                logger.error(f"Error during export type '{export_type}': {e}")

        logger.info("Bulk data export completed.")

# Example usage and test cases
if __name__ == "__main__":
    # Initialize DataStorage
    storage = DataStorage()

    # Example DataFrame to save
    data = {
        'id': [1, 2, 3],
        'name': ['Alice', 'Bob', 'Charlie'],
        'score': [85.5, 92.3, 78.9]
    }
    df = pd.DataFrame(data)

    # Example: Save DataFrame to PostgreSQL
    storage.save_to_sql(df, table_name='students_scores', db_type='postgresql', if_exists='replace')

    # Example: Load DataFrame from PostgreSQL
    query = "SELECT * FROM students_scores;"
    loaded_df = storage.load_from_sql(query, db_type='postgresql')
    if loaded_df is not None:
        print("Loaded DataFrame from PostgreSQL:")
        print(loaded_df)

    # Example: Save DataFrame to MongoDB
    storage.save_to_mongodb(df, db_name='hermod_db', collection_name='students_scores', if_exists='replace')

    # Example: Load DataFrame from MongoDB
    loaded_df_mongo = storage.load_from_mongodb(db_name='hermod_db', collection_name='students_scores')
    if loaded_df_mongo is not None:
        print("\nLoaded DataFrame from MongoDB:")
        print(loaded_df_mongo)

    # Example: Save DataFrame to CSV
    storage.save_to_file(df, file_path='data/exported_data/students_scores.csv', format='csv', sep=',',
                         encoding='utf-8')

    # Example: Load DataFrame from CSV
    loaded_df_csv = storage.load_from_file(file_path='data/exported_data/students_scores.csv', format='csv')
    if loaded_df_csv is not None:
        print("\nLoaded DataFrame from CSV:")
        print(loaded_df_csv)

    # Example: Save and Load a Model
    from sklearn.ensemble import RandomForestClassifier

    # Initialize and train a dummy model
    model = RandomForestClassifier()
    X_train = loaded_df[['id', 'score']]
    y_train = loaded_df['name']
    model.fit(X_train, y_train)

    # Save the model
    storage.save_model(model, file_path='model/trained_models/random_forest.joblib')

    # Load the model
    loaded_model = storage.load_model(file_path='model/trained_models/random_forest.joblib')
    if loaded_model:
        predictions = loaded_model.predict(X_train)
        print("\nModel Predictions:")
        print(predictions)

    # Example: Export model to external API (replace with actual API endpoint)
    # storage.export_model_to_external_system(model, api_endpoint='https://example.com/api/upload_model', headers={'Authorization': 'Bearer YOUR_API_TOKEN'})
