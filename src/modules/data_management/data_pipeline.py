# data_management/data_pipeline.py

import logging
from typing import Dict, Any, Optional
import pandas as pd
import threading
import time

# Import other necessary modules
from .data_ingestor import DataIngestor
from .data_validator import DataValidator
from .data_cleaner import DataCleaner
from .data_preprocessor import DataPreprocessor
from .data_labeler import DataLabeler  # Import DataLabeler

# Configure logging with RotatingFileHandler to prevent log files from becoming too large
from logging.handlers import RotatingFileHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a rotating file handler
handler = RotatingFileHandler('logs/hermod_data_pipeline.log', maxBytes=10**6, backupCount=5)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)


class DataPipeline:
    """
    Manages the ETL (Extract, Transform, Load) pipeline.
    Handles data extraction, validation, cleaning, preprocessing, labeling, and loading.
    Can also manage real-time data processing.
    """

    def __init__(self,
                 ingestor: Optional[DataIngestor] = None,
                 validator: Optional[DataValidator] = None,
                 cleaner: Optional[DataCleaner] = None,
                 preprocessor: Optional[DataPreprocessor] = None,
                 labeler: Optional[DataLabeler] = None):
        """
        Initializes the DataPipeline with necessary components.

        :param ingestor: Instance of DataIngestor for data extraction and loading
        :param validator: Instance of DataValidator for data validation
        :param cleaner: Instance of DataCleaner for data cleaning
        :param preprocessor: Instance of DataPreprocessor for data preprocessing
        :param labeler: Instance of DataLabeler for data labeling
        """
        self.ingestor = ingestor if ingestor else DataIngestor()
        self.validator = validator if validator else DataValidator()
        self.cleaner = cleaner if cleaner else DataCleaner()
        self.preprocessor = preprocessor if preprocessor else DataPreprocessor()
        self.labeler = labeler if labeler else DataLabeler()
        self.stop_event = threading.Event()
        logger.info("DataPipeline initialized with all components.")

    def run_etl(self, source_type: str, source: str, params: Dict[str, Any],
                storage_type: str, storage_params: Dict[str, Any],
                preprocess_params: Optional[Dict[str, Any]] = None,
                labeling_params: Optional[Dict[str, Any]] = None) -> Optional[pd.DataFrame]:
        """
        Executes the ETL process: Extract, Validate, Clean, Preprocess, Label, and Load.

        :param source_type: Type of the data source ('database', 'api', 'file')
        :param source: Identifier for the data source (e.g., DB type, API URL, file path)
        :param params: Parameters dict specific to the source type
        :param storage_type: Type of storage ('database', 'file', etc.)
        :param storage_params: Parameters specific to the storage type
        :param preprocess_params: Parameters for preprocessing pipeline
        :param labeling_params: Parameters for labeling process
        :return: Preprocessed and labeled DataFrame or None if ETL fails
        """
        logger.info(f"Starting ETL process for source type '{source_type}' and source '{source}'.")
        try:
            # Extraction Phase
            df = self.ingestor.ingest_data(source_type, source, params)
            if df is None or df.empty:
                logger.warning("No data extracted. ETL process aborted.")
                return None

            # Validation Phase
            model = self.ingestor._select_data_model(source_type, params)
            validated_df = self.validator.validate_data_integrity(df, model)
            if validated_df is None or validated_df.empty:
                logger.warning("No valid data after validation. ETL process aborted.")
                return None

            # Cleaning Phase
            cleaned_df = self.cleaner.preprocess_data(validated_df)
            if cleaned_df is None or cleaned_df.empty:
                logger.warning("No data after cleaning. ETL process aborted.")
                return None

            # Preprocessing Phase
            if preprocess_params:
                numerical_features = preprocess_params.get('numerical_features',
                                                            cleaned_df.select_dtypes(include=['number']).columns.tolist())
                categorical_features = preprocess_params.get('categorical_features',
                                                              cleaned_df.select_dtypes(include=['object', 'category']).columns.tolist())
                text_features = preprocess_params.get('text_features',
                                                      [col for col in cleaned_df.columns if 'description' in col.lower() or 'text' in col.lower()])
                date_features = preprocess_params.get('date_features',
                                                      [col for col in cleaned_df.columns if 'timestamp' in col.lower() or 'date' in col.lower()])
                pca_components = preprocess_params.get('pca_components', None)

                # Build preprocessing pipeline
                self.preprocessor.build_preprocessing_pipeline(
                    numerical_features=numerical_features,
                    categorical_features=categorical_features,
                    text_features=text_features,
                    date_features=date_features,
                    pca_components=pca_components
                )

                # Fit and transform the data
                preprocessed_df = self.preprocessor.fit_transform(cleaned_df)
                if preprocessed_df is None or preprocessed_df.empty:
                    logger.warning("No data after preprocessing. ETL process aborted.")
                    return None
            else:
                preprocessed_df = cleaned_df  # If no preprocessing parameters, skip preprocessing

            # Labeling Phase
            if labeling_params:
                label_model_path = labeling_params.get('label_model_path')
                labeling_threshold = labeling_params.get('labeling_threshold', 0.8)
                human_involvement = labeling_params.get('human_involvement', True)

                # Initialize DataLabeler with label_model_path if provided
                if label_model_path:
                    self.labeler = DataLabeler(label_model_path=label_model_path)
                else:
                    logger.warning("No label_model_path provided in labeling_params.")

                # Perform labeling
                labeled_df = self.labeler.label_data(preprocessed_df,
                                                     threshold=labeling_threshold,
                                                     human_involvement=human_involvement)
                if labeled_df is None or labeled_df.empty:
                    logger.warning("No data after labeling. ETL process aborted.")
                    return None
            else:
                labeled_df = preprocessed_df  # If no labeling parameters, skip labeling

            # Loading Phase
            success = self.ingestor.store_data(labeled_df, storage_type, storage_params)
            if not success:
                logger.error("Data loading failed. ETL process aborted.")
                return None

            logger.info("ETL process completed successfully.")
            return labeled_df
        except Exception as e:
            logger.error(f"ETL process failed: {e}")
            return None  # Optionally, re-raise the exception if needed

    def run_realtime_etl(self, source_type: str, source: str, params: Dict[str, Any],
                         storage_type: str, storage_params: Dict[str, Any],
                         preprocess_params: Optional[Dict[str, Any]] = None,
                         labeling_params: Optional[Dict[str, Any]] = None,
                         interval: int = 60):
        """
        Runs the ETL process in real-time at specified intervals.

        :param source_type: Type of the data source ('database', 'api', 'file')
        :param source: Identifier for the data source (e.g., DB type, API URL, file path)
        :param params: Parameters dict specific to the source type
        :param storage_type: Type of storage ('database', 'file', etc.)
        :param storage_params: Parameters specific to the storage type
        :param preprocess_params: Parameters for preprocessing pipeline
        :param labeling_params: Parameters for labeling process
        :param interval: Time interval in seconds between ETL runs
        """
        logger.info(f"Starting real-time ETL process with interval {interval} seconds.")

        def etl_job():
            while not self.stop_event.is_set():
                logger.info("Initiating scheduled ETL job.")
                self.run_etl(source_type, source, params, storage_type, storage_params,
                            preprocess_params, labeling_params)
                logger.info(f"ETL job completed. Sleeping for {interval} seconds.")
                time.sleep(interval)

        thread = threading.Thread(target=etl_job, daemon=True)
        thread.start()
        logger.info("Real-time ETL thread started.")

    def stop_realtime_etl(self):
        """
        Stops the real-time ETL process.
        """
        logger.info("Stopping real-time ETL process.")
        self.stop_event.set()


# Example usage and test cases
if __name__ == "__main__":
    # Initialize Ingestors
    try:
        db_ingestor = DataIngestor()
    except Exception as e:
        logger.error(f"Failed to initialize DataIngestor: {e}")
        db_ingestor = None

    # Initialize DataPipeline
    data_pipeline = DataPipeline(ingestor=db_ingestor)

    # Example 1: ETL from PostgreSQL with Labeling
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
            storage_params = {
                'file_path': 'data/processed/nlu_data',  # Directory path
                'file_format': 'csv',
                'base_name': 'ingested_github_repos'  # Base name for the file
            }
            labeling_params = {
                'label_model_path': 'model/nlu_models/nlu_labeling_model.joblib',
                'labeling_threshold': 0.8,
                'human_involvement': True
            }
            processed_df = data_pipeline.run_etl(
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
                labeling_params=labeling_params
            )
            if processed_df is not None:
                print("ETL and Labeling from PostgreSQL completed successfully.")
                print(processed_df.head())
        except Exception as e:
            print(f"Failed ETL and Labeling from PostgreSQL: {e}")

    # Example 2: ETL from MongoDB without Labeling
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
            processed_df = data_pipeline.run_etl(
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
                # No labeling_params provided; labeling is skipped
            )
            if processed_df is not None:
                print("\nETL from MongoDB completed successfully.")
                print(processed_df.head())
        except Exception as e:
            print(f"Failed ETL from MongoDB: {e}")

    # Example 3: ETL from API (Twitter) with Labeling
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
        processed_df = data_pipeline.run_etl(
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

    # Example 4: ETL from CSV file with Labeling
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
        processed_df = data_pipeline.run_etl(
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

    # Example 5: ETL from Excel file without Labeling
    try:
        excel_file_path = 'data/raw/multimodal_data/sample_data.xlsx'  # Replace with your Excel file path
        excel_params = {'format': 'xlsx', 'read_kwargs': {'sheet_name': 'Sheet1'}}
        storage_params = {
            'file_path': 'data/raw/multimodal_data',  # Directory path
            'file_format': 'xlsx',
            'base_name': 'ingested_multimodal_data'  # Base name for the file
        }
        # No labeling_params provided; labeling is skipped
        processed_df = data_pipeline.run_etl(
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

        data_pipeline.run_realtime_etl(
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
        data_pipeline.stop_realtime_etl()
        print("\nReal-time ETL process completed.")
    except Exception as e:
        print(f"Failed Real-time ETL: {e}")
    """
