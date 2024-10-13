# src/modules/data_management/datasets/custom_dataset.py

import logging
import os
from typing import Dict, Any, Optional, List, Union, Generator
import pandas as pd
import json
from datetime import datetime
from abc import ABC, abstractmethod
import joblib
import requests
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging with RotatingFileHandler to prevent log files from becoming too large
from logging.handlers import RotatingFileHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a rotating file handler
handler = RotatingFileHandler('logs/hermod_custom_dataset.log', maxBytes=10**6, backupCount=5)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)


# Import DataStorage and MetadataStorage from their respective modules
from src.modules.data_management.data_storage import DataStorage
from src.modules.data_management.metadata_storage import MetadataStorage


class BaseDataset(ABC):
    """
    Abstract base class for custom datasets. Defines the interface for dataset classes.
    """

    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """
        Loads data from the specified source.
        """
        pass

    @abstractmethod
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses the loaded data.
        """
        pass

    @abstractmethod
    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies transformations to the preprocessed data.
        """
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """
        Retrieves metadata related to the dataset.
        """
        return {
            'dataset_name': self.dataset_name,
            'source': self.source,
            'creation_date': datetime.utcnow(),
            'description': self.description,
            'preprocessing_steps': self.preprocessing_steps,
            'transformation_steps': self.transformation_steps
        }

    def save_metadata(self):
        """
        Saves metadata using MetadataStorage.
        """
        metadata = self.get_metadata()
        metadata_storage = MetadataStorage()
        metadata_storage.save_metadata(metadata, storage_type='sql')  # or 'mongodb' based on preference
        logger.info(f"Metadata for dataset '{self.dataset_name}' saved successfully.")


    def get_data(self) -> pd.DataFrame:
        """
        Complete pipeline to get the final processed data.
        """
        df = self.load_data()
        df = self.preprocess_data(df)
        df = self.transform_data(df)
        self.save_metadata()
        return df


class CSVCustomDataset(BaseDataset):
    """
    Custom dataset class for handling CSV files.
    """

    def __init__(self, file_path: str, dataset_name: str, description: str = "",
                 preprocessing_steps: Optional[List[str]] = None,
                 transformation_steps: Optional[List[str]] = None):
        self.file_path = file_path
        self.dataset_name = dataset_name
        self.description = description
        self.source = 'CSV'
        self.preprocessing_steps = preprocessing_steps or []
        self.transformation_steps = transformation_steps or []
        logger.info(f"Initialized CSVCustomDataset for file '{self.file_path}'.")

    def load_data(self) -> pd.DataFrame:
        logger.info(f"Loading data from CSV file '{self.file_path}'.")
        try:
            df = pd.read_csv(self.file_path)
            logger.info(f"Data loaded successfully from '{self.file_path}'. Shape: {df.shape}")
            return df
        except FileNotFoundError:
            logger.error(f"CSV file '{self.file_path}' not found.")
            raise
        except pd.errors.ParserError as e:
            logger.error(f"Error parsing CSV file '{self.file_path}': {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading CSV file '{self.file_path}': {e}")
            raise

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Preprocessing data for dataset '{self.dataset_name}'.")
        try:
            # Example preprocessing: Handle missing values
            if 'handle_missing' in self.preprocessing_steps:
                df = df.dropna()  # Simple example: drop rows with missing values
                logger.debug("Dropped rows with missing values.")
            # Additional preprocessing steps can be added here
            logger.info("Data preprocessing completed.")
            return df
        except Exception as e:
            logger.error(f"Error during preprocessing: {e}")
            raise

    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Transforming data for dataset '{self.dataset_name}'.")
        try:
            # Example transformation: Encode categorical variables
            if 'encode_categorical' in self.transformation_steps:
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns
                df = pd.get_dummies(df, columns=categorical_cols)
                logger.debug(f"Encoded categorical columns: {list(categorical_cols)}.")
            # Additional transformation steps can be added here
            logger.info("Data transformation completed.")
            return df
        except Exception as e:
            logger.error(f"Error during transformation: {e}")
            raise


class DatabaseCustomDataset(BaseDataset):
    """
    Custom dataset class for handling data from SQL databases.
    """

    def __init__(self, query: str, db_type: str = 'postgresql', dataset_name: str = "DatabaseDataset",
                 description: str = "", preprocessing_steps: Optional[List[str]] = None,
                 transformation_steps: Optional[List[str]] = None):
        self.query = query
        self.db_type = db_type.lower()
        self.dataset_name = dataset_name
        self.description = description
        self.source = f'Database ({self.db_type})'
        self.preprocessing_steps = preprocessing_steps or []
        self.transformation_steps = transformation_steps or []
        logger.info(f"Initialized DatabaseCustomDataset with query '{self.query}' on '{self.db_type}'.")

    def load_data(self) -> pd.DataFrame:
        logger.info(f"Loading data from database using query: {self.query}")
        try:
            data_storage = DataStorage()
            df = data_storage.load_from_sql(self.query, db_type=self.db_type)
            if df is not None:
                logger.info(f"Data loaded successfully from database. Shape: {df.shape}")
                return df
            else:
                logger.warning("No data returned from database query.")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading data from database: {e}")
            raise

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Preprocessing data for dataset '{self.dataset_name}' from database.")
        try:
            # Example preprocessing: Remove duplicates
            if 'remove_duplicates' in self.preprocessing_steps:
                initial_shape = df.shape
                df = df.drop_duplicates()
                final_shape = df.shape
                logger.debug(f"Dropped duplicates. Shape before: {initial_shape}, after: {final_shape}.")
            # Additional preprocessing steps can be added here
            logger.info("Data preprocessing completed.")
            return df
        except Exception as e:
            logger.error(f"Error during preprocessing: {e}")
            raise

    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Transforming data for dataset '{self.dataset_name}' from database.")
        try:
            # Example transformation: Normalize numerical features
            if 'normalize_numeric' in self.transformation_steps:
                numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
                logger.debug(f"Normalized numerical columns: {list(numeric_cols)}.")
            # Additional transformation steps can be added here
            logger.info("Data transformation completed.")
            return df
        except Exception as e:
            logger.error(f"Error during transformation: {e}")
            raise


class APICustomDataset(BaseDataset):
    """
    Custom dataset class for handling data from external APIs.
    """

    def __init__(self, api_endpoint: str, params: Optional[Dict[str, Any]] = None,
                 headers: Optional[Dict[str, str]] = None, dataset_name: str = "APIDataset",
                 description: str = "", preprocessing_steps: Optional[List[str]] = None,
                 transformation_steps: Optional[List[str]] = None):
        self.api_endpoint = api_endpoint
        self.params = params or {}
        self.headers = headers or {}
        self.dataset_name = dataset_name
        self.description = description
        self.source = 'API'
        self.preprocessing_steps = preprocessing_steps or []
        self.transformation_steps = transformation_steps or []
        logger.info(f"Initialized APICustomDataset with endpoint '{self.api_endpoint}'.")

    def load_data(self) -> pd.DataFrame:
        logger.info(f"Fetching data from API endpoint '{self.api_endpoint}' with params {self.params}.")
        try:
            response = requests.get(self.api_endpoint, params=self.params, headers=self.headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            df = pd.json_normalize(data)
            logger.info(f"Data fetched successfully from API. Shape: {df.shape}")
            return df
        except requests.exceptions.RequestException as e:
            logger.error(f"Request to API failed: {e}")
            raise
        except ValueError as e:
            logger.error(f"Error parsing JSON response from API: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error fetching data from API: {e}")
            raise

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Preprocessing data for dataset '{self.dataset_name}' from API.")
        try:
            # Example preprocessing: Convert date strings to datetime objects
            if 'convert_dates' in self.preprocessing_steps:
                date_cols = df.select_dtypes(include=['object']).columns
                for col in date_cols:
                    try:
                        df[col] = pd.to_datetime(df[col], errors='ignore')
                        logger.debug(f"Converted column '{col}' to datetime.")
                    except Exception:
                        logger.debug(f"Column '{col}' could not be converted to datetime.")
            # Additional preprocessing steps can be added here
            logger.info("Data preprocessing completed.")
            return df
        except Exception as e:
            logger.error(f"Error during preprocessing: {e}")
            raise

    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Transforming data for dataset '{self.dataset_name}' from API.")
        try:
            # Example transformation: Filter rows based on a condition
            if 'filter_rows' in self.transformation_steps:
                # Placeholder condition: keep rows where 'active' column is True
                if 'active' in df.columns:
                    initial_shape = df.shape
                    df = df[df['active'] == True]
                    final_shape = df.shape
                    logger.debug(f"Filtered rows where 'active' == True. Shape before: {initial_shape}, after: {final_shape}.")
            # Additional transformation steps can be added here
            logger.info("Data transformation completed.")
            return df
        except Exception as e:
            logger.error(f"Error during transformation: {e}")
            raise
