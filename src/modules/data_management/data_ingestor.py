# src/modules/data_management/data_ingestor.py

import logging
import os
from typing import Dict, Any, Optional, Union
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from pymongo import MongoClient, errors as pymongo_errors
import requests
from pydantic import ValidationError
from datetime import datetime
from dotenv import load_dotenv
import time
import hashlib

from data_labeler import DataLabeler

# Import Data Cleaner
from data_cleaner import DataCleaner

# Import Data Preprocessor
from data_preprocessor import DataPreprocessor

# Import Data Models from the models package
from .models.data_models import (
    BaseDataModel,
    GitHubRepoDataModel,
    APIDataModel,
    FileDataModel,
    TwitterDataModel  # Example additional model
)

# Import DataValidator
from data_validator import DataValidator

# Import BehavioralAuthenticationManager
from src.modules.advanced_security.behavioral_authentication import BehavioralAuthenticationManager

# Load environment variables from .env file
load_dotenv()

# Configure logging with RotatingFileHandler to prevent log files from becoming too large
from logging.handlers import RotatingFileHandler

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Create a rotating file handler
handler = RotatingFileHandler('logs/hermod_data_ingestor.log', maxBytes=10 ** 6, backupCount=5)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)


# DataIngestorInterface (Abstract Base Class)
class DataIngestorInterface:
    """
    Interface for Data Ingestion.
    Defines methods for ingesting data from various sources.
    """

    def ingest(self, source: str, params: Dict[str, Any]) -> pd.DataFrame:
        raise NotImplementedError("Ingest method not implemented.")


# Database Ingestor
class DatabaseIngestor(DataIngestorInterface):
    """
    Ingests data from SQL and NoSQL databases.
    Supports PostgreSQL and MongoDB as examples.
    """

    def __init__(self):
        # Initialize database connections
        self.postgres_engine = self._create_postgres_engine()
        self.mongo_client = self._create_mongo_client()

    def _create_postgres_engine(self):
        try:
            pg_host = os.getenv('POSTGRES_HOST')
            pg_port = os.getenv('POSTGRES_PORT', '5432')
            pg_db = os.getenv('POSTGRES_DB')
            pg_user = os.getenv('POSTGRES_USER')
            pg_password = os.getenv('POSTGRES_PASSWORD')
            if not all([pg_host, pg_port, pg_db, pg_user, pg_password]):
                logging.error("PostgreSQL credentials are not fully set in the environment variables.")
                raise ValueError("Missing PostgreSQL configuration.")
            engine = create_engine(f"postgresql+psycopg2://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_db}")
            logging.info("PostgreSQL engine created successfully.")
            return engine
        except SQLAlchemyError as e:
            logging.error(f"Error creating PostgreSQL engine: {e}")
            raise e

    def _create_mongo_client(self):
        try:
            mongo_uri = os.getenv('MONGODB_URI')
            if not mongo_uri:
                logging.error("MongoDB URI is not set in the environment variables.")
                raise ValueError("Missing MongoDB URI.")
            client = MongoClient(mongo_uri)
            # Test connection
            client.admin.command('ping')
            logging.info("MongoDB client connected successfully.")
            return client
        except pymongo_errors.ConnectionFailure as e:
            logging.error(f"MongoDB connection failed: {e}")
            raise e

    def ingest_postgres(self, query: str) -> pd.DataFrame:
        """
        Executes a SQL query and returns the result as a pandas DataFrame.

        :param query: SQL query string
        :return: DataFrame containing query results
        """
        try:
            df = pd.read_sql_query(query, self.postgres_engine)
            logging.info(f"Successfully ingested data from PostgreSQL with query: {query}")
            return df
        except SQLAlchemyError as e:
            logging.error(f"Error ingesting data from PostgreSQL: {e}")
            raise e

    def ingest_mongodb(self, db_name: str, collection_name: str, filter_query: Dict[str, Any] = {}) -> pd.DataFrame:
        """
        Fetches data from a MongoDB collection and returns it as a pandas DataFrame.

        :param db_name: MongoDB database name
        :param collection_name: MongoDB collection name
        :param filter_query: MongoDB query filter
        :return: DataFrame containing fetched data
        """
        try:
            collection = self.mongo_client[db_name][collection_name]
            cursor = collection.find(filter_query)
            data = list(cursor)
            if not data:
                logging.warning(f"No data found in MongoDB collection '{collection_name}' with filter {filter_query}.")
                return pd.DataFrame()
            df = pd.DataFrame(data)
            logging.info(
                f"Successfully ingested data from MongoDB collection '{collection_name}' in database '{db_name}'.")
            return df
        except pymongo_errors.PyMongoError as e:
            logging.error(f"Error ingesting data from MongoDB: {e}")
            raise e

    def ingest(self, source: str, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Determines the database type and ingests data accordingly.

        :param source: Type of database ('postgresql', 'mongodb')
        :param params: Parameters required for ingestion
        :return: DataFrame containing ingested data
        """
        if source.lower() == 'postgresql':
            query = params.get('query')
            if not query:
                logging.error("SQL query not provided for PostgreSQL ingestion.")
                raise ValueError("SQL query is required for PostgreSQL ingestion.")
            return self.ingest_postgres(query)
        elif source.lower() == 'mongodb':
            db_name = params.get('db_name')
            collection_name = params.get('collection_name')
            filter_query = params.get('filter_query', {})
            if not all([db_name, collection_name]):
                logging.error("Database name and collection name are required for MongoDB ingestion.")
                raise ValueError("db_name and collection_name are required for MongoDB ingestion.")
            return self.ingest_mongodb(db_name, collection_name, filter_query)
        else:
            logging.error(f"Unsupported database source: {source}")
            raise NotImplementedError(f"Ingestion for source '{source}' is not implemented.")


# API Ingestor
class APIIngestor(DataIngestorInterface):
    """
    Ingests data from RESTful APIs.
    """

    def __init__(self):
        # Initialize any required attributes or authentication here
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': os.getenv('USER_AGENT', 'HermodBot/1.0')
        })
        self.api_token = os.getenv('API_TOKEN')  # Generic API token
        if self.api_token:
            self.session.headers.update({'Authorization': f'Bearer {self.api_token}'})
        logging.info("APIIngestor initialized.")

    def ingest(self, source: str, params: Dict[str, Any] = {}) -> pd.DataFrame:
        """
        Fetches data from a specified API endpoint and returns it as a DataFrame.

        :param source: Full URL of the API endpoint
        :param params: Query parameters for the API request
        :return: DataFrame containing API response data
        """
        logging.info(f"Ingesting data from API source: {source} with params: {params}")
        max_retries = params.get('max_retries', 5)
        backoff_factor = params.get('backoff_factor', 0.3)
        timeout = params.get('timeout', 10)
        try:
            for retry in range(max_retries):
                try:
                    response = self.session.get(source, params=params.get('query_params', {}), timeout=timeout)
                    response.raise_for_status()
                    data = response.json()

                    # Normalize JSON data into DataFrame
                    if isinstance(data, list):
                        df = pd.json_normalize(data)
                    elif isinstance(data, dict):
                        df = pd.json_normalize(data)
                    else:
                        logging.error(f"Unsupported API response format: {type(data)}")
                        raise ValueError(f"Unsupported API response format: {type(data)}")

                    logging.info(f"Successfully ingested data from API source: {source}")
                    return df
                except requests.exceptions.HTTPError as e:
                    status_code = response.status_code
                    if status_code in [429, 500, 502, 503, 504]:
                        sleep_time = backoff_factor * (2 ** retry)
                        logging.warning(f"API request error {status_code}. Retrying in {sleep_time} seconds...")
                        time.sleep(sleep_time)
                        continue
                    else:
                        logging.error(f"API request failed for source '{source}': {e}")
                        raise e
                except requests.exceptions.RequestException as e:
                    sleep_time = backoff_factor * (2 ** retry)
                    logging.warning(f"API request exception: {e}. Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                    continue
                except ValueError as e:
                    logging.error(f"Invalid JSON response from API source '{source}': {e}")
                    raise e
            logging.error(f"Failed to ingest data from API source '{source}' after {max_retries} retries.")
            raise Exception(f"API ingestion failed for source '{source}'.")
        except Exception as e:
            logging.error(f"Unhandled exception during API ingestion: {e}")
            raise e


# File Ingestor
class FileIngestor(DataIngestorInterface):
    """
    Ingests data from various file formats such as CSV, JSON, Excel, etc.
    """

    def __init__(self):
        # Initialize any required attributes here
        logging.info("FileIngestor initialized.")

    def ingest(self, source: str, params: Dict[str, Any] = {}) -> pd.DataFrame:
        """
        Reads data from a specified file and returns it as a DataFrame.

        :param source: File path or URL to the data file
        :param params: Parameters dict to specify file format and read options
        :return: DataFrame containing file data
        """
        logging.info(f"Ingesting data from file source: {source} with params: {params}")
        file_format = params.get('format', 'csv').lower()
        read_kwargs = params.get('read_kwargs', {})
        try:
            if file_format == 'csv':
                df = pd.read_csv(source, **read_kwargs)
            elif file_format == 'json':
                df = pd.read_json(source, **read_kwargs)
            elif file_format in ['xls', 'xlsx']:
                df = pd.read_excel(source, **read_kwargs)
            elif file_format == 'parquet':
                df = pd.read_parquet(source, **read_kwargs)
            else:
                logging.error(f"Unsupported file format: {file_format}")
                raise ValueError(f"Unsupported file format: {file_format}")

            # Calculate file size and checksum if possible
            if os.path.exists(source):
                size_bytes = os.path.getsize(source)
                checksum = self.calculate_checksum(source)
                df['size_bytes'] = size_bytes
                df['checksum'] = checksum
                logging.info(f"File size: {size_bytes} bytes, Checksum: {checksum}")
            else:
                logging.warning(f"File '{source}' does not exist locally. Skipping size and checksum calculation.")

            logging.info(f"Successfully ingested data from file source: {source}")
            return df
        except FileNotFoundError as e:
            logging.error(f"File not found: {source} - {e}")
            raise e
        except pd.errors.ParserError as e:
            logging.error(f"Pandas parser error while reading file '{source}': {e}")
            raise e
        except Exception as e:
            logging.error(f"Error ingesting data from file '{source}': {e}")
            raise e

    @staticmethod
    def calculate_checksum(file_path: str, algorithm: str = 'md5') -> str:
        """
        Calculates the checksum of a file.

        :param file_path: Path to the file
        :param algorithm: Hash algorithm ('md5', 'sha1', etc.)
        :return: Hexadecimal checksum string
        """
        hash_func = hashlib.md5() if algorithm == 'md5' else hashlib.sha1()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_func.update(chunk)
            return hash_func.hexdigest()
        except Exception as e:
            logging.error(f"Error calculating checksum for file '{file_path}': {e}")
            return ""


# Composite DataIngestor orchestrating different ingestors
class DataIngestor:
    """
    Orchestrates data ingestion from various sources, validates data, cleans data, preprocesses data, and handles storage.
    """

    def __init__(self,
                 db_ingestor: Optional[DatabaseIngestor] = None,
                 api_ingestor: Optional[APIIngestor] = None,
                 file_ingestor: Optional[FileIngestor] = None):
        """
        Initializes the DataIngestor with specific ingestors.

        :param db_ingestor: Instance of DatabaseIngestor
        :param api_ingestor: Instance of APIIngestor
        :param file_ingestor: Instance of FileIngestor
        """
        self.db_ingestor = db_ingestor
        self.api_ingestor = api_ingestor
        self.file_ingestor = file_ingestor
        self.validator = DataValidator()
        self.cleaner = DataCleaner()
        self.preprocessor = DataPreprocessor()  # Initialize DataPreprocessor
        self.behavioral_auth_manager = BehavioralAuthenticationManager()  # Initialize BehavioralAuthenticationManager
        logging.info("DataIngestor initialized with all components.")

    def ingest_data(self, source_type: str, source: str, params: Dict[str, Any] = {}) -> Optional[pd.DataFrame]:
        logging.info(f"Starting data ingestion for source type '{source_type}' and source '{source}'.")
        try:
            # Extraction Phase
            if source_type.lower() == 'database':
                db_type = params.get('db_type', 'postgresql').lower()
                if db_type == 'postgresql' and self.db_ingestor:
                    df = self.db_ingestor.ingest('postgresql', params)
                elif db_type == 'mongodb' and self.db_ingestor:
                    df = self.db_ingestor.ingest('mongodb', params)
                else:
                    logging.error(f"Unsupported database type '{db_type}' or DatabaseIngestor not initialized.")
                    raise NotImplementedError(f"Ingestion for database type '{db_type}' is not implemented.")
            elif source_type.lower() == 'api':
                if self.api_ingestor:
                    df = self.api_ingestor.ingest(source, params)
                else:
                    logging.error("APIIngestor not initialized.")
                    raise NotImplementedError("APIIngestor is not initialized.")
            elif source_type.lower() == 'file':
                if self.file_ingestor:
                    df = self.file_ingestor.ingest(source, params)
                else:
                    logging.error("FileIngestor not initialized.")
                    raise NotImplementedError("FileIngestor is not initialized.")
            else:
                logging.error(f"Unsupported source type: {source_type}")
                raise ValueError(f"Unsupported source type: {source_type}")

            if df is None or df.empty:
                logging.warning(f"No data ingested from source '{source}'.")
                return None

            # Validation Phase
            try:
                model = self._select_data_model(source_type, params)
                validated_df = self.validator.validate_data_integrity(df, model)
            except Exception as e:
                logging.error(f"Error during data validation: {e}")
                return None

            if validated_df is None or validated_df.empty:
                logging.warning("No valid data after validation.")
                return None

            # Cleaning Phase
            try:
                cleaned_df = self.cleaner.preprocess_data(validated_df)
            except Exception as e:
                logging.error(f"Error during data cleaning: {e}")
                return None

            if cleaned_df is None or cleaned_df.empty:
                logging.warning("No data after cleaning.")
                return None

            # Preprocessing Phase
            try:
                # Define feature lists based on your data schema
                numerical_features = params.get('preprocess_params', {}).get('numerical_features', [])
                categorical_features = params.get('preprocess_params', {}).get('categorical_features', [])
                text_features = params.get('preprocess_params', {}).get('text_features', [])
                date_features = params.get('preprocess_params', {}).get('date_features', [])
                pca_components = params.get('preprocess_params', {}).get('pca_components', None)

                # Build preprocessing pipeline
                preprocessor_pipeline = self.preprocessor.build_preprocessing_pipeline(
                    numerical_features=numerical_features,
                    categorical_features=categorical_features,
                    text_features=text_features,
                    date_features=date_features,
                    pca_components=pca_components
                )

                # Fit and transform the data
                preprocessed_df = self.preprocessor.fit_transform(cleaned_df)
            except Exception as e:
                logging.error(f"Error during data preprocessing: {e}")
                return None

            if preprocessed_df is None or preprocessed_df.empty:
                logging.warning("No data after preprocessing.")
                return None

            # Behavioral Data Handling
            try:
                if 'behavioral_metrics' in params.get('storage_params', {}).get('additional_info', {}):
                    user_id = params.get('storage_params', {}).get('additional_info', {}).get('user_id')
                    if user_id and isinstance(user_id, int):
                        # Register or update behavioral profile
                        behavior_data = {
                            'typing_speed': cleaned_df['typing_speed'].iloc[0],
                            'typing_pattern_similarity': cleaned_df['typing_pattern_similarity'].iloc[0],
                            'mouse_movement_similarity': cleaned_df['mouse_movement_similarity'].iloc[0],
                            'login_time_variance': cleaned_df['login_time_variance'].iloc[0],
                            'device_fingerprint': cleaned_df['device_fingerprint'].iloc[0]
                        }
                        registration_success = self.behavioral_auth_manager.save_behavioral_profile(user_id,
                                                                                                    behavior_data)
                        if not registration_success:
                            logging.warning(f"Failed to register/update behavioral profile for user ID {user_id}.")
            except Exception as e:
                logging.error(f"Error during behavioral data handling: {e}")

            # Log ingestion event
            try:
                from src.modules.data_management.metadata_storage import MetadataStorage
                metadata_storage = MetadataStorage()
                metadata_storage.save_metadata({
                    'event': 'data_ingestion',
                    'source_type': source_type,
                    'source': source,
                    'timestamp': datetime.utcnow().isoformat()
                }, storage_type='data_ingestion_event')
            except Exception as e:
                logging.error(f"Error logging ingestion event: {e}")

            logging.info("Data ingestion, validation, cleaning, and preprocessing completed successfully.")
            return preprocessed_df
        except Exception as e:
            logging.error(f"An error occurred during the data ingestion process: {e}")
            return None

    def ingest(self, source_type: str, source: str, params: Dict[str, Any],
               storage_type: str, storage_params: Dict[str, Any],
               preprocess_params: Optional[Dict[str, Any]] = None,
               labeling_params: Optional[Dict[str, Any]] = None) -> Optional[pd.DataFrame]:
        """
        Implements the ingest method from DataIngestorInterface.
        This method wraps the ingest_data method and integrates labeling if required.

        :param source_type: Type of the data source ('database', 'api', 'file')
        :param source: Identifier for the data source (e.g., DB type, API URL, file path)
        :param params: Parameters dict specific to the source type
        :param storage_type: Type of storage ('database', 'file', etc.)
        :param storage_params: Parameters specific to the storage type
        :param preprocess_params: Parameters for preprocessing pipeline
        :param labeling_params: Parameters for labeling process
        :return: Preprocessed and labeled DataFrame or None if ETL fails
        """
        # This method serves as a public interface for ingestion, allowing for labeling integration
        # It calls ingest_data and handles labeling if labeling_params are provided

        # To avoid confusion with 'ingest_data', 'ingest' serves as the primary method
        logging.info("Starting ingest method from DataIngestorInterface.")

        # Initialize DataLabeler if labeling_params are provided
        if labeling_params:
            label_model_path = labeling_params.get('label_model_path')
            labeling_threshold = labeling_params.get('labeling_threshold', 0.8)
            human_involvement = labeling_params.get('human_involvement', True)
            data_labeler = DataLabeler(label_model_path=label_model_path)
        else:
            data_labeler = None

        # Perform ETL
        preprocessed_df = self.ingest_data(source_type, source, params)
        if preprocessed_df is None or preprocessed_df.empty:
            logging.warning("ETL process returned no data. Ingest method aborted.")
            return None

        # Perform Labeling if DataLabeler is initialized
        if data_labeler:
            try:
                labeled_df = data_labeler.label_data(preprocessed_df,
                                                    threshold=labeling_threshold,
                                                    human_involvement=human_involvement)
                if labeled_df is None or labeled_df.empty:
                    logging.warning("Labeling process returned no data.")
                    return None
            except Exception as e:
                logging.error(f"Labeling process failed: {e}")
                return None
        else:
            labeled_df = preprocessed_df

        # Store the labeled data
        success = self.store_data(labeled_df, storage_type, storage_params)
        if not success:
            logging.error("Data storage failed after labeling.")
            return None

        logging.info("Ingest method completed successfully.")
        return labeled_df

    def _select_data_model(self, source_type: str, params: Dict[str, Any]) -> Any:
        """
        Selects the appropriate DataModel based on the source type and parameters.

        :param source_type: Type of the data source
        :param params: Parameters dict specific to the source type
        :return: Corresponding DataModel class
        """
        if source_type.lower() == 'database':
            db_type = params.get('db_type', 'postgresql').lower()
            if db_type == 'postgresql':
                # Assuming PostgreSQL ingests GitHub repository data
                return GitHubRepoDataModel
            elif db_type == 'mongodb':
                # Replace with specific MongoDB data model if available
                return BaseDataModel
        elif source_type.lower() == 'api':
            # Determine API type from params or source URL
            api_identifier = params.get('api_identifier', 'generic').lower()
            if api_identifier == 'twitter':
                return TwitterDataModel
            else:
                return APIDataModel
        elif source_type.lower() == 'file':
            return FileDataModel
        # Add more conditions for additional source types or specific APIs
        return BaseDataModel  # Fallback model

    def store_data(self, df: pd.DataFrame, storage_type: str, storage_params: Dict[str, Any] = {}) -> bool:
        """
        Stores the validated and preprocessed data into the specified storage system.

        :param df: Cleaned and preprocessed DataFrame to store
        :param storage_type: Type of storage ('database', 'file', etc.)
        :param storage_params: Parameters specific to the storage type
        :return: True if storage is successful, False otherwise
        """
        logging.info(f"Storing data to storage type '{storage_type}' with params: {storage_params}")
        try:
            if storage_type.lower() == 'database':
                db_type = storage_params.get('db_type', 'postgresql').lower()
                table_name = storage_params.get('table_name', 'ingested_data')
                if db_type == 'postgresql' and self.db_ingestor:
                    engine = self.db_ingestor.postgres_engine
                    df.to_sql(table_name, engine, if_exists='append', index=False)
                    logging.info(f"Successfully stored data to PostgreSQL table '{table_name}'.")
                elif db_type == 'mongodb' and self.db_ingestor:
                    db_name = storage_params.get('db_name')
                    collection_name = storage_params.get('collection_name')
                    if not all([db_name, collection_name]):
                        logging.error("Database name and collection name are required for MongoDB storage.")
                        raise ValueError("db_name and collection_name are required for MongoDB storage.")
                    collection = self.db_ingestor.mongo_client[db_name][collection_name]
                    records = df.to_dict(orient='records')
                    collection.insert_many(records)
                    logging.info(
                        f"Successfully stored data to MongoDB collection '{collection_name}' in database '{db_name}'.")
                else:
                    logging.error(f"Unsupported database type '{db_type}' or DatabaseIngestor not initialized.")
                    raise NotImplementedError(f"Storage for database type '{db_type}' is not implemented.")
            elif storage_type.lower() == 'file':
                # Determine the destination path based on storage_params
                # Example: 'file_path' can be a directory or a file
                file_path = storage_params.get('file_path', 'output/ingested_data.csv')
                file_format = storage_params.get('file_format', 'csv').lower()
                read_kwargs = storage_params.get('read_kwargs', {})

                # If file_path is a directory, construct a file name based on timestamp or other logic
                if os.path.isdir(file_path):
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    base_name = storage_params.get('base_name', 'ingested_data')
                    file_path = os.path.join(file_path, f"{base_name}_{timestamp}.{file_format}")

                # Ensure the directory exists
                os.makedirs(os.path.dirname(file_path), exist_ok=True)

                if file_format == 'csv':
                    df.to_csv(file_path, index=False, **read_kwargs)
                elif file_format == 'json':
                    df.to_json(file_path, orient='records', lines=True, **read_kwargs)
                elif file_format in ['xls', 'xlsx']:
                    df.to_excel(file_path, index=False, **read_kwargs)
                elif file_format == 'parquet':
                    df.to_parquet(file_path, index=False, **read_kwargs)
                else:
                    logging.error(f"Unsupported file format for storage: {file_format}")
                    raise ValueError(f"Unsupported file format for storage: {file_format}")
                logging.info(f"Successfully stored data to file '{file_path}'.")
            else:
                logging.error(f"Unsupported storage type: {storage_type}")
                raise ValueError(f"Unsupported storage type: {storage_type}")
            return True
        except Exception as e:
            logging.error(f"Failed to store data: {e}")
            return False

    # Example usage and test cases
if __name__ == "__main__":
    # Initialize Ingestors
    try:
        db_ingestor = DatabaseIngestor()
    except Exception as e:
        logging.error(f"Failed to initialize DatabaseIngestor: {e}")
        db_ingestor = None

    api_ingestor = APIIngestor()
    file_ingestor = FileIngestor()

    # Initialize DataIngestor
    data_ingestor = DataIngestor(
        db_ingestor=db_ingestor,
        api_ingestor=api_ingestor,
        file_ingestor=file_ingestor
    )

    # Example 1: Ingest data from PostgreSQL (GitHub Repositories)
    if db_ingestor:
        try:
            postgres_query = """
                SELECT 
                    id, 
                    name, 
                    value, 
                    timestamp, 
                    full_name, 
                    description, 
                    language, 
                    stargazers_count, 
                    forks_count, 
                    open_issues_count, 
                    watchers_count, 
                    topics, 
                    license, 
                    url 
                FROM github_repositories;
            """
            postgres_params = {'query': postgres_query, 'db_type': 'postgresql'}
            # Define storage parameters to save processed NLU data
            storage_params = {
                'file_path': 'data/processed/nlu_data',  # Directory path
                'file_format': 'csv',
                'base_name': 'ingested_github_repos'  # Base name for the file
            }
            # Example labeling parameters (assuming labeling is desired)
            labeling_params = {
                'label_model_path': 'model/nlu_models/nlu_labeling_model.joblib',
                'labeling_threshold': 0.8,
                'human_involvement': True
            }
            # Ingest and label data
            processed_df = data_ingestor.ingest(
                source_type='database',
                source='postgresql',
                params=postgres_params,
                storage_type='file',
                storage_params=storage_params,
                preprocess_params={
                    'numerical_features': ['value', 'stargazers_count', 'forks_count', 'open_issues_count', 'watchers_count'],
                    'categorical_features': ['language', 'license'],
                    'text_features': ['description'],
                    'date_features': ['timestamp'],
                    'pca_components': 5
                },
                labeling_params=labeling_params  # Pass labeling parameters
            )
            if processed_df is not None:
                print("ETL and Labeling from PostgreSQL completed successfully.")
                print(processed_df.head())
        except Exception as e:
            print(f"Failed ETL and Labeling from PostgreSQL: {e}")

    # Example 2: Ingest data from MongoDB (Cybersecurity Data) without Labeling
    if db_ingestor:
        try:
            mongodb_params = {
                'db_name': 'hermod_db',
                'collection_name': 'sample_collection',
                'filter_query': {},
                'db_type': 'mongodb'
            }
            storage_params = {
                'file_path': 'data/processed/cybersecurity_data',  # Directory path
                'file_format': 'parquet',
                'base_name': 'ingested_cybersecurity_data'  # Base name for the file
            }
            # No labeling_params provided; labeling is skipped
            processed_df = data_ingestor.ingest(
                source_type='database',
                source='mongodb',
                params=mongodb_params,
                storage_type='file',
                storage_params=storage_params,
                preprocess_params={
                    'numerical_features': ['value'],
                    'categorical_features': ['name'],
                    'text_features': [],
                    'date_features': ['timestamp'],
                    'pca_components': None
                }
            )
            if processed_df is not None:
                print("\nETL from MongoDB completed successfully.")
                print(processed_df.head())
        except Exception as e:
            print(f"Failed ETL from MongoDB: {e}")

    # Example 3: Ingest data from an API (Twitter) and store in raw data with Labeling
    try:
        api_source = 'https://api.twitter.com/2/tweets/search/recent'  # Example Twitter API endpoint
        api_params = {
            'query_params': {
                'query': '#OpenAI',
                'max_results': 10,
                'tweet.fields': 'created_at,public_metrics,lang'
            },
            'api_identifier': 'twitter',
            'max_retries': 5,
            'backoff_factor': 0.3,
            'timeout': 10
        }
        storage_params = {
            'file_path': 'data/raw/nlu_data',  # Directory path
            'file_format': 'json',
            'base_name': 'ingested_twitter_data'  # Base name for the file
        }
        labeling_params = {
            'label_model_path': 'model/nlu_models/nlu_labeling_model.joblib',
            'labeling_threshold': 0.75,
            'human_involvement': True
        }
        processed_df = data_ingestor.ingest(
            source_type='api',
            source=api_source,
            params=api_params,
            storage_type='file',
            storage_params=storage_params,
            preprocess_params={
                'numerical_features': ['id', 'public_metrics_retweet_count', 'public_metrics_like_count'],
                'categorical_features': ['lang'],
                'text_features': ['text'],
                'date_features': ['created_at'],
                'pca_components': 10
            },
            labeling_params=labeling_params  # Pass labeling parameters
        )
        if processed_df is not None:
            print("\nETL and Labeling from API (Twitter) completed successfully.")
            print(processed_df.head())
    except Exception as e:
        print(f"Failed ETL and Labeling from API: {e}")

    # Example 4: Ingest data from a CSV file and store in raw data with Labeling
    try:
        csv_file_path = 'data/raw/code_samples/sample_data.csv'  # Replace with your CSV file path
        csv_params = {'format': 'csv', 'read_kwargs': {'delimiter': ',', 'encoding': 'utf-8'}}
        storage_params = {
            'file_path': 'data/raw/code_samples',  # Directory path
            'file_format': 'csv',
            'base_name': 'ingested_code_samples'  # Base name for the file
        }
        labeling_params = {
            'label_model_path': 'model/nlu_models/nlu_labeling_model.joblib',
            'labeling_threshold': 0.85,
            'human_involvement': True
        }
        processed_df = data_ingestor.ingest(
            source_type='file',
            source=csv_file_path,
            params=csv_params,
            storage_type='file',
            storage_params=storage_params,
            preprocess_params={
                'numerical_features': ['value'],
                'categorical_features': ['name'],
                'text_features': [],
                'date_features': ['timestamp'],
                'pca_components': None
            },
            labeling_params=labeling_params  # Pass labeling parameters
        )
        if processed_df is not None:
            print("\nETL and Labeling from CSV file completed successfully.")
            print(processed_df.head())
    except Exception as e:
        print(f"Failed ETL and Labeling from CSV file: {e}")

    # Example 5: Ingest data from an Excel file and store in raw data without Labeling
    try:
        excel_file_path = 'data/raw/multimodal_data/sample_data.xlsx'  # Replace with your Excel file path
        excel_params = {'format': 'xlsx', 'read_kwargs': {'sheet_name': 'Sheet1'}}
        storage_params = {
            'file_path': 'data/raw/multimodal_data',  # Directory path
            'file_format': 'xlsx',
            'base_name': 'ingested_multimodal_data'  # Base name for the file
        }
        # No labeling_params provided; labeling is skipped
        processed_df = data_ingestor.ingest(
            source_type='file',
            source=excel_file_path,
            params=excel_params,
            storage_type='file',
            storage_params=storage_params,
            preprocess_params={
                'numerical_features': ['value'],
                'categorical_features': ['name'],
                'text_features': [],
                'date_features': ['timestamp'],
                'pca_components': None
            }
        )
        if processed_df is not None:
            print("\nETL from Excel file completed successfully.")
            print(processed_df.head())
    except Exception as e:
        print(f"Failed ETL from Excel file: {e}")

    # Example 6: Run Real-Time ETL (Optional)
    # Uncomment the following lines to run ETL every 60 seconds
    """
    try:
        real_time_task = {
            'source_type': 'api',
            'source': api_source,
            'params': api_params,
            'storage_type': 'file',
            'storage_params': {
                'file_path': 'data/raw/nlu_data',  # Directory path
                'file_format': 'json',
                'base_name': 'ingested_real_time_twitter_data'  # Base name for the file
            },
            'preprocess_params': {
                'numerical_features': ['id', 'public_metrics_retweet_count', 'public_metrics_like_count'],
                'categorical_features': ['lang'],
                'text_features': ['text'],
                'date_features': ['created_at'],
                'pca_components': 10
            },
            'labeling_params': {
                'label_model_path': 'model/nlu_models/nlu_labeling_model.joblib',
                'labeling_threshold': 0.75,
                'human_involvement': True
            }
        }

        data_ingestor.run_realtime_etl(
            source_type=real_time_task['source_type'],
            source=real_time_task['source'],
            params=real_time_task['params'],
            storage_type=real_time_task['storage_type'],
            storage_params=real_time_task['storage_params'],
            preprocess_params=real_time_task['preprocess_params'],
            labeling_params=real_time_task['labeling_params'],
            interval=60  # Run every 60 seconds
        )

        # Let it run for a certain period then stop (e.g., 5 minutes)
        time.sleep(300)
        data_ingestor.stop_realtime_etl()
        print("\nReal-time ETL process completed.")
    except Exception as e:
        print(f"Failed Real-time ETL: {e}")
    """
