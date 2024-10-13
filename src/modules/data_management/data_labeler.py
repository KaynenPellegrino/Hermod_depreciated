# data_management/data_labeler.py

import logging
import os
from typing import Dict, Any, Optional
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import hashlib

# Import Data Validator
from .data_validator import DataValidator

# Import Data Models
from .models.data_models import BaseDataModel, GitHubRepoDataModel, APIDataModel, FileDataModel, TwitterDataModel

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

# Configure logging with RotatingFileHandler to prevent log files from becoming too large
from logging.handlers import RotatingFileHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a rotating file handler
handler = RotatingFileHandler('logs/hermod_data_labeler.log', maxBytes=10**6, backupCount=5)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)


class DataLabeler:
    """
    Automates the process of labeling and annotating datasets.
    Incorporates human-in-the-loop for complex or ambiguous cases.
    """

    def __init__(self,
                 validator: Optional[DataValidator] = None,
                 label_model_path: Optional[str] = None):
        """
        Initializes the DataLabeler with necessary components.

        :param validator: Instance of DataValidator for data validation
        :param label_model_path: Path to a pre-trained labeling model
        """
        self.validator = validator if validator else DataValidator()
        self.label_model = self._load_label_model(label_model_path)
        logging.info("DataLabeler initialized with all components.")

    def _load_label_model(self, model_path: Optional[str]) -> Optional[Any]:
        """
        Loads a pre-trained labeling model if provided.

        :param model_path: Path to the pre-trained model
        :return: Loaded model or None
        """
        if model_path and os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                logging.info(f"Labeling model loaded from '{model_path}'.")
                return model
            except Exception as e:
                logging.error(f"Failed to load labeling model from '{model_path}': {e}")
                return None
        else:
            logging.warning("No labeling model path provided or file does not exist.")
            return None

    def automate_labeling(self, df: pd.DataFrame, model: Optional[Any] = None, threshold: float = 0.8) -> pd.DataFrame:
        """
        Applies automated labeling to the dataset.

        :param df: DataFrame to label
        :param model: Pre-trained model for labeling
        :param threshold: Confidence threshold for labeling
        :return: DataFrame with added 'automated_label' and 'confidence' columns
        """
        if model is None:
            model = self.label_model

        if model is None:
            logging.error("No labeling model available for automated labeling.")
            raise ValueError("Labeling model not provided and no pre-trained model loaded.")

        try:
            # Example: Assuming the model has a predict_proba method
            features = self._extract_features_for_labeling(df)
            probabilities = model.predict_proba(features)
            labels = model.predict(features)
            confidence = probabilities.max(axis=1)

            df['automated_label'] = labels
            df['confidence'] = confidence

            logging.info("Automated labeling completed.")
            return df
        except Exception as e:
            logging.error(f"Automated labeling failed: {e}")
            raise e

    def _extract_features_for_labeling(self, df: pd.DataFrame) -> Any:
        """
        Extracts features required for the labeling model from the DataFrame.

        :param df: DataFrame to extract features from
        :return: Features suitable for the labeling model
        """
        # Placeholder for feature extraction logic
        # This should be customized based on the labeling model's requirements
        # For example, selecting specific columns, encoding categorical variables, etc.
        features = df.select_dtypes(include=['number', 'object']).copy()
        # Example: Fill NaNs and encode categorical variables
        features.fillna('', inplace=True)
        features = pd.get_dummies(features)
        return features

    def human_in_the_loop(self, df: pd.DataFrame, confidence_threshold: float = 0.8) -> pd.DataFrame:
        """
        Identifies samples that require human annotation based on confidence scores.

        :param df: DataFrame with 'automated_label' and 'confidence' columns
        :param confidence_threshold: Threshold below which human annotation is required
        :return: DataFrame with updated labels after human annotation
        """
        # Identify samples below the confidence threshold
        uncertain_samples = df[df['confidence'] < confidence_threshold]
        logging.info(f"Found {len(uncertain_samples)} samples requiring human annotation.")

        for idx, row in uncertain_samples.iterrows():
            print(f"\nSample ID: {idx}")
            print(row)
            user_label = input("Please provide the correct label (or press Enter to skip): ")
            if user_label:
                df.at[idx, 'automated_label'] = user_label
                df.at[idx, 'confidence'] = 1.0  # Assuming human label is correct
                logging.info(f"Human label applied to sample ID {idx}.")
            else:
                logging.info(f"No label provided for sample ID {idx}. Skipping.")

        return df

    def label_data(self, df: pd.DataFrame, model: Optional[Any] = None,
                  threshold: float = 0.8, human_involvement: bool = True) -> pd.DataFrame:
        """
        Orchestrates the labeling process: automated labeling and human-in-the-loop.

        :param df: DataFrame to label
        :param model: Pre-trained model for labeling
        :param threshold: Confidence threshold for requiring human annotation
        :param human_involvement: Whether to enable human-in-the-loop
        :return: Fully labeled DataFrame
        """
        try:
            # Step 1: Automated Labeling
            df = self.automate_labeling(df, model, threshold)

            # Step 2: Human-in-the-Loop for uncertain samples
            if human_involvement:
                df = self.human_in_the_loop(df, threshold)

            logging.info("Data labeling process completed.")
            return df
        except Exception as e:
            logging.error(f"Data labeling process failed: {e}")
            return df

    def save_labeled_data(self, df: pd.DataFrame, output_path: str, file_format: str = 'csv') -> bool:
        """
        Saves the labeled DataFrame to the specified path in the desired format.

        :param df: Labeled DataFrame
        :param output_path: Path to save the labeled data
        :param file_format: Format to save the data ('csv', 'json', 'xlsx', 'parquet')
        :return: True if successful, False otherwise
        """
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            if file_format == 'csv':
                df.to_csv(output_path, index=False)
            elif file_format == 'json':
                df.to_json(output_path, orient='records', lines=True)
            elif file_format in ['xls', 'xlsx']:
                df.to_excel(output_path, index=False)
            elif file_format == 'parquet':
                df.to_parquet(output_path, index=False)
            else:
                logging.error(f"Unsupported file format: {file_format}")
                return False

            logging.info(f"Labeled data saved to '{output_path}' successfully.")
            return True
        except Exception as e:
            logging.error(f"Failed to save labeled data to '{output_path}': {e}")
            return False

    def evaluate_labeling(self, df: pd.DataFrame, true_labels: Optional[pd.Series] = None) -> Optional[str]:
        """
        Evaluates the labeling performance if true labels are available.

        :param df: DataFrame with 'automated_label' and 'confidence' columns
        :param true_labels: Series containing the true labels
        :return: Classification report as a string or None
        """
        if true_labels is None:
            logging.warning("True labels not provided. Evaluation cannot be performed.")
            return None

        try:
            report = classification_report(true_labels, df['automated_label'])
            logging.info("Labeling evaluation completed.")
            return report
        except Exception as e:
            logging.error(f"Failed to evaluate labeling: {e}")
            return None


# Example usage and test cases
if __name__ == "__main__":
    # Initialize DataLabeler
    data_labeler = DataLabeler(label_model_path='model/nlu_models/nlu_labeling_model.joblib')

    # Load a sample dataset
    sample_data_path = 'data/processed/nlu_data/ingested_github_repos_20231012_101530.csv'
    if os.path.exists(sample_data_path):
        df = pd.read_csv(sample_data_path)
        logging.info("Sample data loaded successfully.")

        # Perform labeling
        labeled_df = data_labeler.label_data(df, threshold=0.8, human_involvement=True)

        # Save labeled data
        output_path = 'data/processed/nlu_data/labeled_github_repos_20231012_101530.csv'
        success = data_labeler.save_labeled_data(labeled_df, output_path, file_format='csv')
        if success:
            print(f"Labeled data saved to '{output_path}'.")
    else:
        logging.error(f"Sample data file '{sample_data_path}' does not exist.")
