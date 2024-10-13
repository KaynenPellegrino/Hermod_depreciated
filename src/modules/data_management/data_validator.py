# data_management/data_validator.py

import logging
from typing import Optional, List, Any, Dict
import pandas as pd
import numpy as np
from pydantic import ValidationError
from .models.data_models import BaseDataModel
from datetime import datetime

# Configure logging
logging.basicConfig(
    filename='logs/hermod_data_validator.log',
    level=logging.DEBUG,
    format='%(asctime)s:%(levelname)s:%(message)s'
)


class DataValidator:
    """
    Provides tools for validating data integrity and quality.
    """

    def __init__(self):
        logging.info("DataValidator initialized.")

    def validate_schema(self, df: pd.DataFrame, model: Any) -> Optional[pd.DataFrame]:
        """
        Validates the DataFrame schema against a Pydantic model.

        :param df: DataFrame to validate
        :param model: Pydantic model class for validation
        :return: Validated DataFrame or None if validation fails
        """
        logging.info("Starting schema validation.")
        validated_records = []
        for index, row in df.iterrows():
            try:
                record = model(**row.to_dict())
                validated_records.append(record.dict())
            except ValidationError as ve:
                logging.warning(f"Schema validation error at row {index}: {ve}")
                continue  # Skip invalid records
        if not validated_records:
            logging.error("All records failed schema validation.")
            return None
        validated_df = pd.DataFrame(validated_records)
        logging.info(f"Schema validation completed. Valid records: {len(validated_df)}")
        return validated_df

    def check_missing_values(self, df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        """
        Checks for missing values in the DataFrame and removes columns exceeding the threshold.

        :param df: DataFrame to check
        :param threshold: Proportion of missing values allowed
        :return: DataFrame after removing columns with excessive missing values
        """
        logging.info(f"Checking for missing values with threshold {threshold}.")
        missing_percentage = df.isnull().mean()
        columns_to_drop = missing_percentage[missing_percentage > threshold].index.tolist()
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
            logging.warning(f"Dropped columns due to missing values exceeding {threshold * 100}%: {columns_to_drop}")
        else:
            logging.info("No columns exceeded the missing values threshold.")
        return df

    def fill_missing_values(self, df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        """
        Fills missing values in the DataFrame based on the specified strategy.

        :param df: DataFrame to fill missing values
        :param strategy: Strategy to fill missing values ('mean', 'median', 'mode', 'ffill', 'bfill')
        :return: DataFrame after filling missing values
        """
        logging.info(f"Filling missing values using strategy: {strategy}.")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns

        if strategy == 'mean':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif strategy == 'median':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif strategy == 'mode':
            df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
        elif strategy == 'ffill':
            df.fillna(method='ffill', inplace=True)
        elif strategy == 'bfill':
            df.fillna(method='bfill', inplace=True)
        else:
            logging.error(f"Unsupported fill strategy: {strategy}.")
            raise ValueError(f"Unsupported fill strategy: {strategy}.")

        logging.info("Missing values filled successfully.")
        return df

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes duplicate records from the DataFrame.

        :param df: DataFrame to process
        :return: DataFrame after removing duplicates
        """
        initial_count = len(df)
        df = df.drop_duplicates()
        final_count = len(df)
        duplicates_removed = initial_count - final_count
        if duplicates_removed > 0:
            logging.warning(f"Removed {duplicates_removed} duplicate records.")
        else:
            logging.info("No duplicate records found.")
        return df

    def check_data_types(self, df: pd.DataFrame, expected_types: Dict[str, Any]) -> pd.DataFrame:
        """
        Ensures that each column in the DataFrame matches the expected data type.
        Attempts to cast columns to the expected type.

        :param df: DataFrame to check
        :param expected_types: Dictionary mapping column names to expected data types
        :return: DataFrame after enforcing data types
        """
        logging.info("Checking and enforcing data types.")
        for column, dtype in expected_types.items():
            if column in df.columns:
                try:
                    df[column] = df[column].astype(dtype)
                    logging.info(f"Column '{column}' cast to {dtype}.")
                except ValueError as ve:
                    logging.warning(f"Failed to cast column '{column}' to {dtype}: {ve}")
        return df

    def detect_outliers(self, df: pd.DataFrame, columns: List[str], threshold: float = 3.0) -> Dict[str, pd.Series]:
        """
        Detects outliers in specified numeric columns using the Z-score method.

        :param df: DataFrame to analyze
        :param columns: List of numeric columns to check for outliers
        :param threshold: Z-score threshold to identify outliers
        :return: Dictionary mapping column names to boolean Series indicating outliers
        """
        logging.info(f"Detecting outliers in columns {columns} with threshold {threshold}.")
        outliers = {}
        for column in columns:
            if column in df.columns:
                mean = df[column].mean()
                std = df[column].std()
                if std == 0:
                    logging.warning(f"Standard deviation for column '{column}' is zero. Skipping outlier detection.")
                    outliers[column] = pd.Series([False] * len(df))
                    continue
                z_scores = (df[column] - mean) / std
                outlier_mask = z_scores.abs() > threshold
                outliers[column] = outlier_mask
                num_outliers = outlier_mask.sum()
                logging.info(f"Detected {num_outliers} outliers in column '{column}'.")
        return outliers

    def standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardizes column names to lowercase and replaces spaces with underscores.

        :param df: DataFrame to process
        :return: DataFrame with standardized column names
        """
        logging.info("Standardizing column names.")
        df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
        logging.info("Column names standardized.")
        return df

    def validate_data_integrity(self, df: pd.DataFrame, model: Any) -> Optional[pd.DataFrame]:
        """
        Performs a series of data integrity checks on the DataFrame.

        :param df: DataFrame to validate
        :param model: Pydantic model class for schema validation
        :return: Validated DataFrame or None if validation fails
        """
        logging.info("Starting comprehensive data integrity validation.")
        # Standardize column names
        df = self.standardize_column_names(df)

        # Remove duplicates
        df = self.remove_duplicates(df)

        # Check and fill missing values
        df = self.check_missing_values(df, threshold=0.5)
        df = self.fill_missing_values(df, strategy='mean')

        # Enforce data types based on Pydantic model
        expected_types = self.get_expected_types(model)
        df = self.check_data_types(df, expected_types)

        # Detect outliers in numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        outliers = self.detect_outliers(df, numeric_columns, threshold=3.0)
        for column, mask in outliers.items():
            df = df[~mask]  # Remove outliers
            logging.info(f"Outliers removed from column '{column}'.")

        # Schema validation using Pydantic
        validated_df = self.validate_schema(df, model)
        return validated_df

    def get_expected_types(self, model: Any) -> Dict[str, Any]:
        """
        Extracts expected data types from the Pydantic model.

        :param model: Pydantic model class
        :return: Dictionary mapping field names to data types
        """
        expected_types = {}
        for field_name, field in model.__fields__.items():
            python_type = field.type_
            # Map Pydantic/Python types to pandas dtypes
            if python_type == int:
                expected_types[field_name] = 'Int64'
            elif python_type == float:
                expected_types[field_name] = 'float'
            elif python_type == str:
                expected_types[field_name] = 'string'
            elif python_type == bool:
                expected_types[field_name] = 'bool'
            elif python_type == datetime:
                expected_types[field_name] = 'datetime64[ns]'
            else:
                expected_types[field_name] = 'object'  # Default to object for complex types
        logging.info(f"Expected data types extracted from model: {expected_types}")
        return expected_types

    # Additional validation methods can be added here as needed
