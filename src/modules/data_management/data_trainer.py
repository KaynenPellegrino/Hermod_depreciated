# data_management/data_trainer.py

import logging
import os
import sys
from typing import Any, Dict, Optional, Tuple, Union
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

# Example ML models (extend as needed)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

# Configure logging with RotatingFileHandler to prevent log files from becoming too large
from logging.handlers import RotatingFileHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a rotating file handler
handler = RotatingFileHandler('logs/hermod_data_trainer.log', maxBytes=10**6, backupCount=5)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)


class DataTrainer:
    """
    Provides utilities for training machine learning models, including model selection,
    hyperparameter tuning, training pipelines, evaluation, and model persistence.
    """

    def __init__(self,
                 model_type: str = 'classification',
                 target_column: str = 'label',
                 test_size: float = 0.2,
                 random_state: int = 42):
        """
        Initializes the DataTrainer with configuration parameters.

        :param model_type: Type of machine learning task ('classification' or 'regression')
        :param target_column: Name of the target variable in the dataset
        :param test_size: Proportion of the dataset to include in the test split
        :param random_state: Controls the shuffling applied to the data before applying the split
        """
        self.model_type = model_type.lower()
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.pipeline = None
        logger.info(f"DataTrainer initialized for {self.model_type} tasks.")

    def load_data(self, file_path: str, file_format: str = 'csv') -> pd.DataFrame:
        """
        Loads the dataset from the specified file path.

        :param file_path: Path to the data file
        :param file_format: Format of the data file ('csv', 'json', 'xlsx', 'parquet')
        :return: Pandas DataFrame containing the dataset
        """
        logger.info(f"Loading data from {file_path} with format {file_format}.")
        try:
            if file_format.lower() == 'csv':
                df = pd.read_csv(file_path)
            elif file_format.lower() == 'json':
                df = pd.read_json(file_path, lines=True)
            elif file_format.lower() in ['xls', 'xlsx']:
                df = pd.read_excel(file_path, sheet_name='Sheet1')
            elif file_format.lower() == 'parquet':
                df = pd.read_parquet(file_path)
            else:
                logger.error(f"Unsupported file format: {file_format}")
                raise ValueError(f"Unsupported file format: {file_format}")

            logger.info(f"Data loaded successfully with shape {df.shape}.")
            return df
        except FileNotFoundError as e:
            logger.error(f"File not found: {file_path} - {e}")
            raise e
        except pd.errors.ParserError as e:
            logger.error(f"Pandas parser error: {e}")
            raise e
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise e

    def preprocess_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Splits the dataset into features and target variable.

        :param df: Input DataFrame
        :return: Tuple containing features DataFrame and target Series
        """
        logger.info("Preprocessing features and target variable.")
        try:
            X = df.drop(columns=[self.target_column])
            y = df[self.target_column]
            logger.info(f"Features and target variable separated successfully.")
            return X, y
        except KeyError as e:
            logger.error(f"Target column '{self.target_column}' not found in the dataset.")
            raise e
        except Exception as e:
            logger.error(f"Error preprocessing features: {e}")
            raise e

    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple:
        """
        Splits the dataset into training and testing sets.

        :param X: Features DataFrame
        :param y: Target Series
        :return: Tuple containing X_train, X_test, y_train, y_test
        """
        logger.info("Splitting data into training and testing sets.")
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y if self.model_type == 'classification' else None
            )
            logger.info(f"Data split successfully. Training set: {X_train.shape}, Testing set: {X_test.shape}")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logger.error(f"Error splitting data: {e}")
            raise e

    def select_model(self) -> BaseEstimator:
        """
        Selects a machine learning model based on the task type.

        :return: Scikit-learn estimator instance
        """
        logger.info(f"Selecting model for {self.model_type} task.")
        try:
            if self.model_type == 'classification':
                self.model = RandomForestClassifier(random_state=self.random_state)
            elif self.model_type == 'regression':
                from sklearn.ensemble import RandomForestRegressor
                self.model = RandomForestRegressor(random_state=self.random_state)
            else:
                logger.error(f"Unsupported model type: {self.model_type}")
                raise ValueError(f"Unsupported model type: {self.model_type}")
            logger.info(f"Model selected: {self.model.__class__.__name__}")
            return self.model
        except Exception as e:
            logger.error(f"Error selecting model: {e}")
            raise e

    def build_pipeline(self, X_train: pd.DataFrame) -> Pipeline:
        """
        Builds a machine learning pipeline with preprocessing and the selected model.

        :param X_train: Training features DataFrame
        :return: Scikit-learn Pipeline instance
        """
        logger.info("Building machine learning pipeline.")
        try:
            numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

            logger.debug(f"Numerical features: {numerical_features}")
            logger.debug(f"Categorical features: {categorical_features}")

            # Define preprocessing steps
            numerical_transformer = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])

            categorical_transformer = Pipeline(steps=[
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

            preprocessor = ColumnTransformer(transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])

            # Create the pipeline
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', self.model)
            ])

            logger.info("Pipeline built successfully.")
            return pipeline
        except Exception as e:
            logger.error(f"Error building pipeline: {e}")
            raise e

    def tune_hyperparameters(self, pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        """
        Performs hyperparameter tuning using GridSearchCV or RandomizedSearchCV.

        :param pipeline: Scikit-learn Pipeline instance
        :param X_train: Training features DataFrame
        :param y_train: Training target Series
        :return: Pipeline with the best found parameters
        """
        logger.info("Starting hyperparameter tuning.")
        try:
            if self.model_type == 'classification':
                param_grid = {
                    'model__n_estimators': [100, 200],
                    'model__max_depth': [None, 10, 20],
                    'model__min_samples_split': [2, 5],
                    'model__min_samples_leaf': [1, 2]
                }
                search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
            elif self.model_type == 'regression':
                param_grid = {
                    'model__n_estimators': [100, 200],
                    'model__max_depth': [None, 10, 20],
                    'model__min_samples_split': [2, 5],
                    'model__min_samples_leaf': [1, 2]
                }
                search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
            else:
                logger.error(f"Unsupported model type for tuning: {self.model_type}")
                raise ValueError(f"Unsupported model type for tuning: {self.model_type}")

            search.fit(X_train, y_train)
            logger.info(f"Hyperparameter tuning completed. Best parameters: {search.best_params_}")
            self.best_params = search.best_params_
            return search.best_estimator_
        except Exception as e:
            logger.error(f"Error during hyperparameter tuning: {e}")
            raise e

    def train(self, pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
        """
        Trains the machine learning model using the provided pipeline.

        :param pipeline: Scikit-learn Pipeline instance
        :param X_train: Training features DataFrame
        :param y_train: Training target Series
        :return: Trained Pipeline instance
        """
        logger.info("Starting model training.")
        try:
            pipeline.fit(X_train, y_train)
            logger.info("Model training completed successfully.")
            return pipeline
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise e

    def evaluate(self, pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluates the trained model on the test set and returns performance metrics.

        :param pipeline: Trained Scikit-learn Pipeline instance
        :param X_test: Testing features DataFrame
        :param y_test: Testing target Series
        :return: Dictionary containing evaluation metrics
        """
        logger.info("Evaluating the trained model.")
        try:
            predictions = pipeline.predict(X_test)
            metrics = {}

            if self.model_type == 'classification':
                metrics['accuracy'] = accuracy_score(y_test, predictions)
                metrics['precision'] = precision_score(y_test, predictions, average='weighted', zero_division=0)
                metrics['recall'] = recall_score(y_test, predictions, average='weighted', zero_division=0)
                metrics['f1_score'] = f1_score(y_test, predictions, average='weighted', zero_division=0)
                if hasattr(pipeline.named_steps['model'], "predict_proba"):
                    metrics['roc_auc'] = roc_auc_score(y_test, pipeline.predict_proba(X_test), multi_class='ovr')
                logger.info(f"Classification Metrics: {metrics}")
                logger.debug(f"Classification Report:\n{classification_report(y_test, predictions)}")
                logger.debug(f"Confusion Matrix:\n{confusion_matrix(y_test, predictions)}")
            elif self.model_type == 'regression':
                metrics['mean_squared_error'] = mean_squared_error(y_test, predictions)
                metrics['mean_absolute_error'] = mean_absolute_error(y_test, predictions)
                logger.info(f"Regression Metrics: {metrics}")
            else:
                logger.error(f"Unsupported model type for evaluation: {self.model_type}")
                raise ValueError(f"Unsupported model type for evaluation: {self.model_type}")

            return metrics
        except NotFittedError as e:
            logger.error(f"Model is not fitted yet: {e}")
            raise e
        except Exception as e:
            logger.error(f"Error during model evaluation: {e}")
            raise e

    def save_model(self, pipeline: Pipeline, model_path: str) -> bool:
        """
        Saves the trained model pipeline to the specified path using joblib.

        :param pipeline: Trained Scikit-learn Pipeline instance
        :param model_path: File path to save the model
        :return: True if saved successfully, False otherwise
        """
        logger.info(f"Saving the trained model to {model_path}.")
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            joblib.dump(pipeline, model_path)
            logger.info(f"Model saved successfully to {model_path}.")
            return True
        except Exception as e:
            logger.error(f"Error saving the model: {e}")
            return False

    def load_model(self, model_path: str) -> Pipeline:
        """
        Loads a trained model pipeline from the specified path.

        :param model_path: File path of the saved model
        :return: Loaded Scikit-learn Pipeline instance
        """
        logger.info(f"Loading the trained model from {model_path}.")
        try:
            pipeline = joblib.load(model_path)
            logger.info(f"Model loaded successfully from {model_path}.")
            return pipeline
        except FileNotFoundError as e:
            logger.error(f"Model file not found: {model_path} - {e}")
            raise e
        except Exception as e:
            logger.error(f"Error loading the model: {e}")
            raise e

    def run_training_pipeline(self, data_path: str, file_format: str = 'csv', model_save_path: str = 'model/trained_model.joblib') -> Dict[str, Any]:
        """
        Executes the entire training pipeline: data loading, preprocessing, splitting, model selection,
        hyperparameter tuning, training, evaluation, and saving the model.

        :param data_path: Path to the processed and labeled data file
        :param file_format: Format of the data file ('csv', 'json', 'xlsx', 'parquet')
        :param model_save_path: File path to save the trained model
        :return: Dictionary containing evaluation metrics
        """
        logger.info("Running the complete training pipeline.")
        try:
            # Load data
            df = self.load_data(data_path, file_format=file_format)

            # Preprocess features
            X, y = self.preprocess_features(df)

            # Split data
            X_train, X_test, y_train, y_test = self.split_data(X, y)

            # Select model
            self.select_model()

            # Build pipeline
            pipeline = self.build_pipeline(X_train)

            # Hyperparameter tuning
            tuned_pipeline = self.tune_hyperparameters(pipeline, X_train, y_train)

            # Train model
            trained_pipeline = self.train(tuned_pipeline, X_train, y_train)

            # Evaluate model
            metrics = self.evaluate(trained_pipeline, X_test, y_test)

            # Save model
            if self.save_model(trained_pipeline, model_save_path):
                logger.info(f"Training pipeline completed successfully. Model saved at {model_save_path}.")
            else:
                logger.error("Training pipeline completed but failed to save the model.")

            return metrics

        except Exception as e:
            logger.error(f"Error running training pipeline: {e}")
            raise e


# Example usage and test cases
if __name__ == "__main__":
    # Initialize DataTrainer for classification
    data_trainer = DataTrainer(model_type='classification', target_column='automated_label')

    # Define paths (adjust these paths as per your project structure)
    data_file_path = 'data/processed/nlu_data/labeled_github_repos_20231012_101530.csv'
    model_save_path = 'model/nlu_models/random_forest_classifier.joblib'

    try:
        # Run the training pipeline
        evaluation_metrics = data_trainer.run_training_pipeline(
            data_path=data_file_path,
            file_format='csv',
            model_save_path=model_save_path
        )
        print("Training completed successfully. Evaluation Metrics:")
        for metric, value in evaluation_metrics.items():
            print(f"{metric}: {value}")

    except Exception as e:
        print(f"Training pipeline failed: {e}")
