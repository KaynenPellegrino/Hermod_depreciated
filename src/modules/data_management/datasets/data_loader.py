# src/modules/data_management/datasets/data_loader.py

import logging
import os
from typing import Dict, Any, Optional, List, Union, Generator
import pandas as pd
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
from imblearn.over_sampling import SMOTE
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging with RotatingFileHandler to prevent log files from becoming too large
from logging.handlers import RotatingFileHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a rotating file handler
handler = RotatingFileHandler('logs/hermod_data_loader.log', maxBytes=10**6, backupCount=5)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)


# Import DataStorage and MetadataStorage from their respective modules
from src.modules.data_management.data_storage import DataStorage
from src.modules.data_management.metadata_storage import MetadataStorage
from src.modules.data_management.datasets.custom_dataset import BaseDataset


def load_dataset(dataset_config: Dict[str, Any]) -> pd.DataFrame:
    """
    Factory function to create and process a dataset based on configuration.

    :param dataset_config: Dictionary containing dataset configuration
    :return: Processed pandas DataFrame
    """
    logger.info(f"Loading dataset with configuration: {dataset_config}")
    dataset_type = dataset_config.get('type')
    if not dataset_type:
        logger.error("Dataset configuration must include 'type' field.")
        raise ValueError("Dataset configuration must include 'type' field.")

    # Import dataset classes dynamically if not already imported
    from custom_dataset import CSVCustomDataset, DatabaseCustomDataset, APICustomDataset

    if dataset_type.lower() == 'csv':
        dataset = CSVCustomDataset(
            file_path=dataset_config['file_path'],
            dataset_name=dataset_config.get('dataset_name', 'CSV_Dataset'),
            description=dataset_config.get('description', ''),
            preprocessing_steps=dataset_config.get('preprocessing_steps', []),
            transformation_steps=dataset_config.get('transformation_steps', [])
        )
    elif dataset_type.lower() == 'database':
        dataset = DatabaseCustomDataset(
            query=dataset_config['query'],
            db_type=dataset_config.get('db_type', 'postgresql'),
            dataset_name=dataset_config.get('dataset_name', 'Database_Dataset'),
            description=dataset_config.get('description', ''),
            preprocessing_steps=dataset_config.get('preprocessing_steps', []),
            transformation_steps=dataset_config.get('transformation_steps', [])
        )
    elif dataset_type.lower() == 'api':
        dataset = APICustomDataset(
            api_endpoint=dataset_config['api_endpoint'],
            params=dataset_config.get('params', {}),
            headers=dataset_config.get('headers', {}),
            dataset_name=dataset_config.get('dataset_name', 'API_Dataset'),
            description=dataset_config.get('description', ''),
            preprocessing_steps=dataset_config.get('preprocessing_steps', []),
            transformation_steps=dataset_config.get('transformation_steps', [])
        )
    else:
        logger.error(f"Unsupported dataset type: {dataset_type}")
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    df = dataset.get_data()
    logger.info(f"Dataset '{dataset.dataset_name}' loaded and processed successfully.")
    return df


def get_batches(df: pd.DataFrame, batch_size: int, shuffle_data: bool = True) -> Generator[pd.DataFrame, None, None]:
    """
    Generator that yields batches of data from the DataFrame.

    :param df: pandas DataFrame containing the data
    :param batch_size: Number of samples per batch
    :param shuffle_data: Whether to shuffle data before batching
    :yield: pandas DataFrame batch
    """
    logger.info(f"Generating batches of size {batch_size}. Shuffle data: {shuffle_data}")
    try:
        if shuffle_data:
            df = shuffle(df)
            logger.debug("Data shuffled.")
        total_samples = len(df)
        for start in range(0, total_samples, batch_size):
            end = start + batch_size
            batch = df.iloc[start:end]
            logger.debug(f"Yielding batch from index {start} to {end} (exclusive).")
            yield batch
        logger.info("All batches generated successfully.")
    except Exception as e:
        logger.error(f"Error during batching: {e}")
        raise


def apply_data_augmentation(df: pd.DataFrame, augmentation_config: Dict[str, Any]) -> pd.DataFrame:
    """
    Applies data augmentation techniques to the DataFrame based on the provided configuration.

    :param df: pandas DataFrame to augment
    :param augmentation_config: Dictionary specifying augmentation techniques
    :return: Augmented pandas DataFrame
    """
    logger.info(f"Applying data augmentation with configuration: {augmentation_config}")
    try:
        # Example augmentation: Add Gaussian noise to numerical columns
        if augmentation_config.get('add_gaussian_noise'):
            noise_level = augmentation_config['add_gaussian_noise'].get('noise_level', 0.01)
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            logger.debug(f"Adding Gaussian noise with noise level {noise_level} to columns: {list(numeric_cols)}.")
            df[numeric_cols] += np.random.normal(0, noise_level, size=df[numeric_cols].shape)

        # Example augmentation: SMOTE for imbalanced classification
        if augmentation_config.get('smote'):
            target_column = augmentation_config['smote'].get('target_column')
            if target_column and target_column in df.columns:
                X = df.drop(columns=[target_column])
                y = df[target_column]
                smote = SMOTE()
                X_res, y_res = smote.fit_resample(X, y)
                df = pd.concat([X_res, y_res], axis=1)
                logger.debug(f"Applied SMOTE. New class distribution: {y_res.value_counts().to_dict()}.")
            else:
                logger.error("SMOTE augmentation requires a valid 'target_column' in the configuration.")

        # Additional augmentation techniques can be added here

        logger.info("Data augmentation completed successfully.")
        return df
    except ImportError as e:
        logger.error(f"Missing required library for data augmentation: {e}")
        raise
    except Exception as e:
        logger.error(f"Error during data augmentation: {e}")
        raise
