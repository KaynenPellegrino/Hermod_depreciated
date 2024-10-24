# src/modules/self_optimization/automl_manager.py

import logging
import os
import json
from typing import Dict, Any, Optional, List

from src.utils.logger import get_logger
from src.utils.configuration_manager import ConfigurationManager
from src.modules.data_management.data_ingestor import DataIngestor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.base import BaseEstimator
import joblib


class AutoMLManager:
    """
    Automates model selection, hyperparameter tuning, and benchmarking of models to improve AI performance.
    """

    def __init__(self, project_id: str):
        """
        Initializes the AutoMLManager with necessary components.

        Args:
            project_id (str): Unique identifier for the project.
        """
        self.logger = get_logger(__name__)
        self.project_id = project_id
        self.config_manager = ConfigurationManager()
        self.config = self.config_manager.get_configuration(project_id)

        self.data_ingestor = DataIngestor(project_id)
        self.automl_results_path = self.config.get('automl_results_path',
                                                   f'data/processed/{project_id}_automl_results.json')
        self.model_save_path = self.config.get('model_save_path', f'models/{project_id}_best_model.joblib')

        self.logger.info(f"AutoMLManager initialized for project '{project_id}'.")

    def load_data(self) -> Optional[Dict[str, Any]]:
        """
        Loads data for model training and evaluation.

        Returns:
            Optional[Dict[str, Any]]: Dictionary containing training and testing data.
        """
        self.logger.info("Loading data for AutoML.")
        try:
            # Example: Load dataset from data_ingestor
            data = self.data_ingestor.load_processed_data('training')
            X = data.get('features')
            y = data.get('labels')
            if X is None or y is None:
                self.logger.error("Features or labels not found in the loaded data.")
                return None
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.logger.debug(f"Data split into train and test sets.")
            return {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            }
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}", exc_info=True)
            return None

    def select_models(self) -> List[BaseEstimator]:
        """
        Selects a list of candidate models for evaluation.

        Returns:
            List[BaseEstimator]: List of instantiated scikit-learn models.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier

        self.logger.info("Selecting candidate models for AutoML.")
        models = [
            LogisticRegression(max_iter=1000),
            RandomForestClassifier(),
            SVC(),
            KNeighborsClassifier()
        ]
        self.logger.debug(f"Selected models: {[type(model).__name__ for model in models]}")
        return models

    def tune_hyperparameters(self, model: BaseEstimator, param_grid: Dict[str, Any], X_train, y_train) -> GridSearchCV:
        """
        Performs hyperparameter tuning using GridSearchCV.

        Args:
            model (BaseEstimator): The model to tune.
            param_grid (Dict[str, Any]): The hyperparameter grid.
            X_train: Training features.
            y_train: Training labels.

        Returns:
            GridSearchCV: Fitted GridSearchCV object.
        """
        self.logger.info(f"Starting hyperparameter tuning for {type(model).__name__}.")
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        self.logger.info(
            f"Completed hyperparameter tuning for {type(model).__name__}. Best params: {grid_search.best_params_}")
        return grid_search

    def benchmark_models(self, models: List[BaseEstimator], X_test, y_test) -> Dict[str, Any]:
        """
        Benchmarks the models on the test set and records performance metrics.

        Args:
            models (List[BaseEstimator]): List of trained models.
            X_test: Testing features.
            y_test: Testing labels.

        Returns:
            Dict[str, Any]: Dictionary containing performance metrics for each model.
        """
        self.logger.info("Benchmarking models on the test set.")
        metrics = {}
        for model in models:
            model_name = type(model).__name__
            self.logger.debug(f"Benchmarking model: {model_name}")
            predictions = model.predict(X_test)
            metrics[model_name] = {
                'accuracy': accuracy_score(y_test, predictions),
                'f1_score': f1_score(y_test, predictions, average='weighted'),
                'precision': precision_score(y_test, predictions, average='weighted'),
                'recall': recall_score(y_test, predictions, average='weighted')
            }
            self.logger.debug(f"Metrics for {model_name}: {metrics[model_name]}")
        return metrics

    def select_best_model(self, metrics: Dict[str, Any]) -> Optional[str]:
        """
        Selects the best model based on F1 score.

        Args:
            metrics (Dict[str, Any]): Performance metrics for each model.

        Returns:
            Optional[str]: The name of the best model, or None if no models are present.
        """
        if not metrics:
            self.logger.error("No metrics available to select the best model.")
            return None
        best_model = max(metrics.items(), key=lambda item: item[1]['f1_score'])[0]
        self.logger.info(f"Selected best model: {best_model} with F1 score: {metrics[best_model]['f1_score']}")
        return best_model

    def save_model(self, model: BaseEstimator) -> bool:
        """
        Saves the trained model to disk.

        Args:
            model (BaseEstimator): The trained model to save.

        Returns:
            bool: True if the model was saved successfully, False otherwise.
        """
        self.logger.info(f"Saving model '{type(model).__name__}' to '{self.model_save_path}'.")
        try:
            os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
            joblib.dump(model, self.model_save_path)
            self.logger.info(f"Model saved successfully to '{self.model_save_path}'.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}", exc_info=True)
            return False

    def save_automl_results(self, metrics: Dict[str, Any]) -> bool:
        """
        Saves the AutoML benchmarking results to a JSON file.

        Args:
            metrics (Dict[str, Any]): Performance metrics for each model.

        Returns:
            bool: True if the results were saved successfully, False otherwise.
        """
        self.logger.info(f"Saving AutoML results to '{self.automl_results_path}'.")
        try:
            os.makedirs(os.path.dirname(self.automl_results_path), exist_ok=True)
            with open(self.automl_results_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            self.logger.info(f"AutoML results saved successfully to '{self.automl_results_path}'.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save AutoML results: {e}", exc_info=True)
            return False

    def run_automl_pipeline(self) -> bool:
        """
        Runs the full AutoML pipeline: model selection, hyperparameter tuning, benchmarking, and saving results.

        Returns:
            bool: True if the AutoML pipeline was successful, False otherwise.
        """
        self.logger.info("Running the AutoML pipeline.")
        try:
            data = self.load_data()
            if data is None:
                self.logger.error("Data loading failed. AutoML pipeline aborted.")
                return False

            models = self.select_models()
            tuned_models = []
            param_grids = {
                'LogisticRegression': {'C': [0.1, 1, 10], 'solver': ['lbfgs', 'liblinear']},
                'RandomForestClassifier': {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]},
                'SVC': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
                'KNeighborsClassifier': {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
            }

            for model in models:
                model_name = type(model).__name__
                param_grid = param_grids.get(model_name, {})
                if param_grid:
                    grid_search = self.tune_hyperparameters(model, param_grid, data['X_train'], data['y_train'])
                    tuned_models.append(grid_search.best_estimator_)
                else:
                    self.logger.warning(
                        f"No hyperparameter grid defined for model '{model_name}'. Using default parameters.")
                    model.fit(data['X_train'], data['y_train'])
                    tuned_models.append(model)

            metrics = self.benchmark_models(tuned_models, data['X_test'], data['y_test'])
            best_model_name = self.select_best_model(metrics)
            if best_model_name is None:
                self.logger.error("Best model selection failed. AutoML pipeline aborted.")
                return False

            # Find the best model
            best_model = next((model for model in tuned_models if type(model).__name__ == best_model_name), None)
            if best_model is None:
                self.logger.error(f"Best model '{best_model_name}' not found among tuned models.")
                return False

            model_saved = self.save_model(best_model)
            results_saved = self.save_automl_results(metrics)

            if model_saved and results_saved:
                self.logger.info("AutoML pipeline completed successfully.")
                return True
            else:
                self.logger.warning("AutoML pipeline completed with some issues.")
                return False

        except Exception as e:
            self.logger.error(f"AutoML pipeline failed: {e}", exc_info=True)
            return False


# Example Usage and Test Cases
if __name__ == "__main__":
    # Initialize AutoMLManager
    project_id = "proj_12345"  # Replace with your actual project ID
    automl_manager = AutoMLManager(project_id)

    # Run the AutoML pipeline
    success = automl_manager.run_automl_pipeline()
    print(f"AutoML Pipeline Successful: {success}")
