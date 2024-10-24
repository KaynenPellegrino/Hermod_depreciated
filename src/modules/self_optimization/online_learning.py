# src/modules/self_optimization/online_learning.py

import os
import logging
import time
from typing import Dict, Any, Optional, List

import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report

from utils.logger import get_logger
from utils.configuration_manager import ConfigurationManager
from src.modules.data_management.data_ingestor import DataIngestor


class OnlineLearner:
    """
    Implements online learning capabilities to update models in real-time as new data arrives.
    """

    def __init__(self, project_id: str, model_registry, data_ingestor: DataIngestor):
        """
        Initializes the OnlineLearner with necessary configurations and dependencies.

        Args:
            project_id (str): Unique identifier for the project.
            model_registry: Instance of ModelRegistry to manage models.
            data_ingestor (DataIngestor): Instance of DataIngestor to fetch new data.
        """
        self.logger = get_logger(__name__)
        self.project_id = project_id
        self.config_manager = ConfigurationManager()
        self.config = self.config_manager.get_configuration(project_id)

        self.model_registry = model_registry
        self.data_ingestor = data_ingestor

        self.online_model_path = self.config.get('online_model_path', f'models/{project_id}/online_model.joblib')
        self.model = self.load_or_initialize_model()

        self.logger.info(f"OnlineLearner initialized for project '{project_id}'.")

    def load_or_initialize_model(self):
        """
        Loads an existing online model or initializes a new one.

        Returns:
            sklearn.base.BaseEstimator: The loaded or initialized model.
        """
        if os.path.exists(self.online_model_path):
            self.logger.info(f"Loading existing online model from '{self.online_model_path}'.")
            model = joblib.load(self.online_model_path)
            return model
        else:
            self.logger.info("No existing online model found. Initializing a new SGDClassifier.")
            model = SGDClassifier(loss='log', max_iter=1000, tol=1e-3)
            return model

    def partial_fit(self, X, y, classes: Optional[List[Any]] = None):
        """
        Performs a partial fit on the model with new data.

        Args:
            X (array-like): Feature matrix.
            y (array-like): Labels.
            classes (Optional[List[Any]]): List of all possible classes.
        """
        self.logger.info("Performing partial fit on the online model.")
        if classes is None:
            # Initialize classes if not provided
            classes = list(set(y))
            self.logger.debug(f"Classes initialized: {classes}")
        self.model.partial_fit(X, y, classes=classes)
        self.logger.info("Partial fit completed.")

    def update_model(self, data_source: str, params: Dict[str, Any]):
        """
        Fetches new data and updates the online model.

        Args:
            data_source (str): Type of data source ('database', 'api', 'file').
            params (Dict[str, Any]): Parameters for data ingestion.
        """
        self.logger.info(f"Updating model with data from source '{data_source}'.")
        try:
            # Ingest new data
            new_data = self.data_ingestor.ingest(
                source_type=data_source,
                source=params.get('source'),
                params=params
            )
            if new_data is None or new_data.empty:
                self.logger.warning("No new data available for online learning.")
                return

            # Assume that the last column is the label
            X = new_data.iloc[:, :-1].values
            y = new_data.iloc[:, -1].values

            # Perform partial fit
            self.partial_fit(X, y)

            # Evaluate performance on the new data
            predictions = self.model.predict(X)
            accuracy = accuracy_score(y, predictions)
            self.logger.info(f"Model updated with new data. Accuracy on new data: {accuracy:.4f}")
            self.logger.debug(classification_report(y, predictions))

            # Save the updated model
            self.save_model()

            # Update the model registry with performance
            self.model_registry.update_performance(
                model_name='online_model',
                version='v1.0',  # Assuming a single version for online model
                performance=accuracy,
                metrics={'accuracy': accuracy}
            )
        except Exception as e:
            self.logger.error(f"Failed to update the online model: {e}", exc_info=True)

    def save_model(self):
        """
        Saves the current state of the online model to disk.
        """
        try:
            os.makedirs(os.path.dirname(self.online_model_path), exist_ok=True)
            joblib.dump(self.model, self.online_model_path)
            self.logger.info(f"Online model saved to '{self.online_model_path}'.")
        except Exception as e:
            self.logger.error(f"Failed to save online model: {e}", exc_info=True)

    def run_online_learning_pipeline(self):
        """
        Runs the online learning pipeline by periodically fetching new data and updating the model.
        """
        self.logger.info("Starting the online learning pipeline.")
        try:
            while True:
                # Define how to fetch new data
                # For demonstration, assume a database source
                data_source = 'database'
                params = {
                    'source': 'postgresql',
                    'db_type': 'postgresql',
                    'query': 'SELECT * FROM new_training_data;'  # Replace with your actual query
                }

                # Update the model with new data
                self.update_model(data_source, params)

                # Sleep for a defined interval before next update
                update_interval = self.config.get('update_interval_seconds', 300)  # Default: 5 minutes
                self.logger.info(f"Sleeping for {update_interval} seconds before next update.")
                time.sleep(update_interval)
        except KeyboardInterrupt:
            self.logger.info("Online learning pipeline terminated by user.")
        except Exception as e:
            self.logger.error(f"An error occurred in the online learning pipeline: {e}", exc_info=True)

    def run_sample_operations(self):
        """
        Runs sample operations to demonstrate usage of OnlineLearner.
        """
        self.logger.info("Running sample operations on OnlineLearner.")

        # Example: Update model with sample data
        sample_params = {
            'source': 'postgresql',
            'db_type': 'postgresql',
            'query': 'SELECT feature1, feature2, label FROM sample_training_data;'  # Replace with actual query
        }
        self.update_model('database', sample_params)


# Example Usage and Test Cases
if __name__ == "__main__":
    from model_registry import ModelRegistry

    # Initialize ModelRegistry
    project_id = os.getenv('PROJECT_ID', 'default_project')  # Ensure PROJECT_ID is set
    model_registry = ModelRegistry(project_id=project_id)

    # Initialize DataIngestor
    from src.modules.data_management.data_ingestor import DataIngestor, DatabaseIngestor, APIIngestor, FileIngestor

    try:
        db_ingestor = DatabaseIngestor()
    except Exception as e:
        logging.error(f"Failed to initialize DatabaseIngestor: {e}")
        db_ingestor = None

    api_ingestor = APIIngestor()
    file_ingestor = FileIngestor()

    data_ingestor = DataIngestor(
        db_ingestor=db_ingestor,
        api_ingestor=api_ingestor,
        file_ingestor=file_ingestor
    )

    # Initialize OnlineLearner
    online_learner = OnlineLearner(
        project_id=project_id,
        model_registry=model_registry,
        data_ingestor=data_ingestor
    )

    # Run sample operations
    online_learner.run_sample_operations()
