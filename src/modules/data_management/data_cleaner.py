# data_management/data_cleaner.py

import logging
from typing import Optional, List, Any, Dict
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator
from .data_validator import DataValidator

# Configure logging
logging.basicConfig(
    filename='logs/hermod_data_cleaner.log',
    level=logging.DEBUG,
    format='%(asctime)s:%(levelname)s:%(message)s'
)


class DataCleaner:
    """
    Provides functions to clean and preprocess raw data, handling tasks like missing value imputation,
    outlier detection, and data normalization. Ensures that datasets used by the AI models are of high quality.
    """

    def __init__(self):
        self.validator = DataValidator()
        self.scaler = None  # To be initialized based on strategy
        logging.info("DataCleaner initialized.")

    def impute_missing_values(self, df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        """
        Imputes missing values in the DataFrame based on the specified strategy.

        :param df: DataFrame to process
        :param strategy: Strategy for imputation ('mean', 'median', 'most_frequent', 'constant')
        :return: DataFrame with imputed missing values
        """
        logging.info(f"Imputing missing values using strategy: {strategy}.")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns

        # Impute numeric columns
        if numeric_cols.any():
            numeric_imputer = SimpleImputer(strategy=strategy)
            df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])
            logging.info(f"Missing values imputed in numeric columns using {strategy}.")

        # Impute categorical columns
        if categorical_cols.any():
            categorical_imputer = SimpleImputer(strategy='most_frequent')
            df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])
            logging.info("Missing values imputed in categorical columns using most_frequent.")

        return df

    def detect_and_handle_outliers(self, df: pd.DataFrame, threshold: float = 3.0,
                                   method: str = 'remove') -> pd.DataFrame:
        """
        Detects and handles outliers in the DataFrame using the Z-score method.

        :param df: DataFrame to process
        :param threshold: Z-score threshold to identify outliers
        :param method: Method to handle outliers ('remove', 'cap')
        :return: DataFrame after handling outliers
        """
        logging.info(f"Detecting and handling outliers with threshold: {threshold}, method: {method}.")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            mean = df[col].mean()
            std = df[col].std()
            if std == 0:
                logging.warning(f"Standard deviation for column '{col}' is zero. Skipping outlier detection.")
                continue
            z_scores = (df[col] - mean) / std
            if method == 'remove':
                original_count = len(df)
                df = df[(z_scores.abs() <= threshold)]
                removed = original_count - len(df)
                if removed > 0:
                    logging.info(f"Removed {removed} outliers from column '{col}'.")
            elif method == 'cap':
                lower_bound = mean - (threshold * std)
                upper_bound = mean + (threshold * std)
                before_count = len(df)
                df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
                df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
                after_count = len(df)
                capped = before_count - after_count
                if capped > 0:
                    logging.info(f"Capped {capped} outliers in column '{col}'.")
            else:
                logging.error(f"Unsupported outlier handling method: {method}.")
                raise ValueError(f"Unsupported outlier handling method: {method}.")
        return df

    def normalize_data(self, df: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """
        Normalizes or scales numerical data in the DataFrame.

        :param df: DataFrame to process
        :param method: Normalization method ('standard', 'minmax')
        :return: DataFrame with normalized data
        """
        logging.info(f"Normalizing data using method: {method}.")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if method == 'standard':
            scaler = StandardScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            logging.info("Data normalized using StandardScaler.")
        elif method == 'minmax':
            scaler = MinMaxScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            logging.info("Data normalized using MinMaxScaler.")
        else:
            logging.error(f"Unsupported normalization method: {method}.")
            raise ValueError(f"Unsupported normalization method: {method}.")
        return df

    def encode_categorical_variables(self, df: pd.DataFrame, columns: Optional[List[str]] = None,
                                     strategy: str = 'onehot') -> pd.DataFrame:
        """
        Encodes categorical variables in the DataFrame.

        :param df: DataFrame to process
        :param columns: List of columns to encode. If None, all categorical columns are encoded.
        :param strategy: Encoding strategy ('onehot', 'label')
        :return: DataFrame with encoded categorical variables
        """
        logging.info(f"Encoding categorical variables using strategy: {strategy}.")
        if columns is None:
            columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if strategy == 'onehot':
            df = pd.get_dummies(df, columns=columns, drop_first=True)
            logging.info("Categorical variables encoded using one-hot encoding.")
        elif strategy == 'label':
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            for col in columns:
                df[col] = le.fit_transform(df[col].astype(str))
                logging.info(f"Categorical variable '{col}' encoded using label encoding.")
        else:
            logging.error(f"Unsupported encoding strategy: {strategy}.")
            raise ValueError(f"Unsupported encoding strategy: {strategy}.")
        return df

    def remove_unnecessary_columns(self, df: pd.DataFrame, columns_to_remove: List[str]) -> pd.DataFrame:
        """
        Removes specified columns from the DataFrame.

        :param df: DataFrame to process
        :param columns_to_remove: List of column names to remove
        :return: DataFrame after removing specified columns
        """
        logging.info(f"Removing unnecessary columns: {columns_to_remove}.")
        existing_columns = [col for col in columns_to_remove if col in df.columns]
        df = df.drop(columns=existing_columns)
        logging.info(f"Removed columns: {existing_columns}.")
        return df

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs feature engineering tasks such as creating new features or transforming existing ones.
        This is a placeholder for any domain-specific feature engineering.

        :param df: DataFrame to process
        :return: DataFrame after feature engineering
        """
        logging.info("Starting feature engineering.")
        # Example: Create a new feature based on existing ones
        if 'created_at' in df.columns and 'timestamp' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['time_diff'] = (df['created_at'] - df['timestamp']).dt.total_seconds()
            logging.info("Feature 'time_diff' created based on 'created_at' and 'timestamp'.")
        # Add more feature engineering steps as needed
        return df

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the full data cleaning and preprocessing pipeline.

        :param df: Raw DataFrame to process
        :return: Cleaned and preprocessed DataFrame
        """
        logging.info("Starting full data preprocessing pipeline.")

        # Impute missing values
        df = self.impute_missing_values(df, strategy='mean')

        # Remove unnecessary columns if needed (example)
        # df = self.remove_unnecessary_columns(df, ['unnecessary_column1', 'unnecessary_column2'])

        # Detect and handle outliers
        df = self.detect_and_handle_outliers(df, threshold=3.0, method='remove')

        # Normalize data
        df = self.normalize_data(df, method='standard')

        # Encode categorical variables
        df = self.encode_categorical_variables(df, strategy='onehot')

        # Feature engineering
        df = self.feature_engineering(df)

        logging.info("Data preprocessing pipeline completed.")
        return df

    # Additional cleaning methods can be added here as needed
