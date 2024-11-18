# src/modules/ml/model_ensemble_builder.py

import logging
import os
# Configure logging with RotatingFileHandler to prevent log files from becoming too large
from logging.handlers import RotatingFileHandler
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from sklearn.base import BaseEstimator
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier

# Import DataStorage from data_management module
from src.modules.data_management.staging import DataStorage

# Load environment variables
load_dotenv()

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = RotatingFileHandler(
    os.path.join('logs', 'model_ensemble_builder.log'),
    maxBytes=5 * 1024 * 1024,  # 5 MB
    backupCount=5
)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)


class ModelEnsembleBuilder:
    """
    Automates the creation of model ensembles, combining multiple machine learning models
    to improve prediction accuracy and generalizability. Manages different ensemble
    techniques like bagging, boosting, and stacking.
    """

    def __init__(self):
        """
        Initializes the ModelEnsembleBuilder with necessary configurations.
        """
        try:
            self.data_storage = DataStorage()
            logger.info("ModelEnsembleBuilder initialized successfully.")
        except Exception as e:
            logger.exception(f"Failed to initialize ModelEnsembleBuilder: {e}")
            raise e

    def create_bagging_ensemble(self, base_estimator: BaseEstimator = DecisionTreeClassifier(),
                                n_estimators: int = 10, random_state: Optional[int] = None) -> BaggingClassifier:
        """
        Creates a Bagging ensemble model.

        :param base_estimator: The base estimator to fit on random subsets of the dataset.
        :param n_estimators: The number of base estimators in the ensemble.
        :param random_state: Controls the randomness of the estimator.
        :return: Configured BaggingClassifier instance.
        """
        logger.info("Creating Bagging ensemble model.")
        try:
            bagging = BaggingClassifier(
                base_estimator=base_estimator,
                n_estimators=n_estimators,
                random_state=random_state
            )
            logger.info("Bagging ensemble model created successfully.")
            return bagging
        except Exception as e:
            logger.error(f"Failed to create Bagging ensemble: {e}")
            raise e

    def create_boosting_ensemble(self, n_estimators: int = 100, learning_rate: float = 0.1,
                                 random_state: Optional[int] = None) -> GradientBoostingClassifier:
        """
        Creates a Boosting ensemble model.

        :param n_estimators: The number of boosting stages to perform.
        :param learning_rate: Learning rate shrinks the contribution of each tree.
        :param random_state: Controls the randomness of the estimator.
        :return: Configured GradientBoostingClassifier instance.
        """
        logger.info("Creating Boosting ensemble model.")
        try:
            boosting = GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                random_state=random_state
            )
            logger.info("Boosting ensemble model created successfully.")
            return boosting
        except Exception as e:
            logger.error(f"Failed to create Boosting ensemble: {e}")
            raise e

    def create_stacking_ensemble(self, estimators: List[tuple], final_estimator: BaseEstimator = LogisticRegression(),
                                 cv: int = 5) -> StackingClassifier:
        """
        Creates a Stacking ensemble model.

        :param estimators: List of (name, estimator) tuples.
        :param final_estimator: The final estimator to fit on the stacked features.
        :param cv: Number of cross-validation folds.
        :return: Configured StackingClassifier instance.
        """
        logger.info("Creating Stacking ensemble model.")
        try:
            stacking = StackingClassifier(
                estimators=estimators,
                final_estimator=final_estimator,
                cv=cv
            )
            logger.info("Stacking ensemble model created successfully.")
            return stacking
        except Exception as e:
            logger.error(f"Failed to create Stacking ensemble: {e}")
            raise e

    def train_ensemble(self, ensemble_model: BaseEstimator, X_train: pd.DataFrame, y_train: pd.Series,
                       X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """
        Trains the ensemble model and evaluates its performance.

        :param ensemble_model: The ensemble model to train.
        :param X_train: Training features.
        :param y_train: Training labels.
        :param X_val: Validation features.
        :param y_val: Validation labels.
        :return: Dictionary containing training and validation metrics.
        """
        logger.info(f"Training ensemble model: {ensemble_model.__class__.__name__}")
        try:
            ensemble_model.fit(X_train, y_train)
            logger.info("Ensemble model training completed.")

            # Predictions
            y_train_pred = ensemble_model.predict(X_train)
            y_val_pred = ensemble_model.predict(X_val)

            # Evaluation
            train_accuracy = accuracy_score(y_train, y_train_pred)
            val_accuracy = accuracy_score(y_val, y_val_pred)
            classification_rep = classification_report(y_val, y_val_pred, output_dict=True)

            metrics = {
                'train_accuracy': train_accuracy,
                'validation_accuracy': val_accuracy,
                'classification_report': classification_rep
            }

            logger.info(f"Ensemble model evaluation metrics: {metrics}")
            return metrics
        except Exception as e:
            logger.error(f"Failed to train ensemble model: {e}")
            raise e

    def save_model(self, model: BaseEstimator, model_name: str, file_path: str) -> bool:
        """
        Saves the trained ensemble model to a file using joblib.

        :param model: Trained ensemble model.
        :param model_name: Name identifier for the model.
        :param file_path: Destination file path.
        :return: True if saving is successful, False otherwise.
        """
        logger.info(f"Saving ensemble model '{model_name}' to '{file_path}'.")
        try:
            self.data_storage.save_model(model, file_path=file_path)
            logger.info(f"Ensemble model '{model_name}' saved successfully to '{file_path}'.")
            return True
        except Exception as e:
            logger.error(f"Failed to save ensemble model '{model_name}': {e}")
            return False

    def load_model(self, file_path: str) -> Optional[BaseEstimator]:
        """
        Loads a trained ensemble model from a file using joblib.

        :param file_path: Path to the model file.
        :return: Loaded model object or None if failed.
        """
        logger.info(f"Loading ensemble model from '{file_path}'.")
        try:
            model = self.data_storage.load_model(file_path=file_path)
            if model:
                logger.info(f"Ensemble model loaded successfully from '{file_path}'.")
                return model
            else:
                logger.warning(f"No model found at '{file_path}'.")
                return None
        except Exception as e:
            logger.error(f"Failed to load ensemble model from '{file_path}': {e}")
            return None

    def evaluate_ensemble(self, model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluates the ensemble model on the test set.

        :param model: Trained ensemble model.
        :param X_test: Test features.
        :param y_test: Test labels.
        :return: Dictionary containing evaluation metrics.
        """
        logger.info(f"Evaluating ensemble model: {model.__class__.__name__}")
        try:
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            classification_rep = classification_report(y_test, y_pred, output_dict=True)

            metrics = {
                'test_accuracy': accuracy,
                'classification_report': classification_rep
            }

            logger.info(f"Ensemble model evaluation metrics: {metrics}")
            return metrics
        except Exception as e:
            logger.error(f"Failed to evaluate ensemble model: {e}")
            raise e

    def generate_feature_importance(self, model: BaseEstimator, feature_names: List[str]) -> Optional[Dict[str, float]]:
        """
        Generates feature importance scores for the ensemble model.

        :param model: Trained ensemble model.
        :param feature_names: List of feature names.
        :return: Dictionary mapping feature names to their importance scores or None if not available.
        """
        logger.info(f"Generating feature importance for model: {model.__class__.__name__}")
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_importance = dict(zip(feature_names, importances))
                logger.info(f"Feature importances: {feature_importance}")
                return feature_importance
            else:
                logger.warning(f"Model '{model.__class__.__name__}' does not support feature importances.")
                return None
        except Exception as e:
            logger.error(f"Failed to generate feature importances: {e}")
            return None

    # --------------------- Example Usage --------------------- #

    def example_usage(self):
        """
        Demonstrates example usage of the ModelEnsembleBuilder class.
        """
        import pandas as pd
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split

        try:
            # Initialize ModelEnsembleBuilder
            builder = ModelEnsembleBuilder()

            # Load dataset
            iris = load_iris()
            X = pd.DataFrame(iris.data, columns=iris.feature_names)
            y = pd.Series(iris.target)

            # Split data
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)

            # Create ensemble models
            bagging = builder.create_bagging_ensemble(n_estimators=20, random_state=42)
            boosting = builder.create_boosting_ensemble(n_estimators=100, learning_rate=0.1, random_state=42)
            stacking = builder.create_stacking_ensemble(
                estimators=[
                    ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
                    ('dt', DecisionTreeClassifier(random_state=42))
                ],
                final_estimator=LogisticRegression()
            )

            # Train ensemble models
            bagging_metrics = builder.train_ensemble(bagging, X_train, y_train, X_val, y_val)
            boosting_metrics = builder.train_ensemble(boosting, X_train, y_train, X_val, y_val)
            stacking_metrics = builder.train_ensemble(stacking, X_train, y_train, X_val, y_val)

            # Save models
            builder.save_model(bagging, 'bagging_iris', 'models/bagging_iris.joblib')
            builder.save_model(boosting, 'boosting_iris', 'models/boosting_iris.joblib')
            builder.save_model(stacking, 'stacking_iris', 'models/stacking_iris.joblib')

            # Load models
            loaded_bagging = builder.load_model('models/bagging_iris.joblib')
            loaded_boosting = builder.load_model('models/boosting_iris.joblib')
            loaded_stacking = builder.load_model('models/stacking_iris.joblib')

            # Evaluate models
            if loaded_bagging:
                bagging_eval = builder.evaluate_ensemble(loaded_bagging, X_test, y_test)
                print("Bagging Ensemble Evaluation:")
                print(bagging_eval)

            if loaded_boosting:
                boosting_eval = builder.evaluate_ensemble(loaded_boosting, X_test, y_test)
                print("\nBoosting Ensemble Evaluation:")
                print(boosting_eval)

            if loaded_stacking:
                stacking_eval = builder.evaluate_ensemble(loaded_stacking, X_test, y_test)
                print("\nStacking Ensemble Evaluation:")
                print(stacking_eval)

            # Generate feature importance for the bagging model
            feature_importance = builder.generate_feature_importance(loaded_bagging, X.columns.tolist())
            if feature_importance:
                print("\nFeature Importances for Bagging Ensemble:")
                for feature, importance in feature_importance.items():
                    print(f"{feature}: {importance:.4f}")

        except Exception as e:
            logger.exception(f"Error in example usage: {e}")


    # --------------------- Main Execution --------------------- #

    if __name__ == "__main__":
        # Run the model ensemble builder example
        example_usage()
