# src/modules/self_optimization/federated_learning.py

import logging
import os
import json
from typing import Dict, Any, List, Optional

import tensorflow as tf
import tensorflow_federated as tff

from src.utils.logger import get_logger
from src.utils.configuration_manager import ConfigurationManager
from src.modules.data_management.data_ingestor import DataIngestor


class FederatedLearningManager:
    """
    Enables Hermod to train models across decentralized data sets (e.g., multiple machines or client data)
    without compromising privacy, using TensorFlow Federated.
    """

    def __init__(self, project_id: str):
        """
        Initializes the FederatedLearningManager with necessary components.

        Args:
            project_id (str): Unique identifier for the project.
        """
        self.logger = get_logger(__name__)
        self.project_id = project_id
        self.config_manager = ConfigurationManager()
        self.config = self.config_manager.get_configuration(project_id)

        self.data_ingestor = DataIngestor(project_id)
        self.model_save_path = self.config.get('federated_model_save_path', f'models/{project_id}_federated_model.tf')

        self.logger.info(f"FederatedLearningManager initialized for project '{project_id}'.")

    def load_federated_data(self) -> Optional[List[Dict[str, Any]]]:
        """
        Loads decentralized data sets for federated training.

        Returns:
            Optional[List[Dict[str, Any]]]: List of client data dictionaries.
        """
        self.logger.info("Loading federated data.")
        try:
            # Placeholder: Assume data_ingestor can load data for multiple clients
            # Each client's data is a dict with 'features' and 'labels'
            client_ids = self.config.get('federated_clients', ['client_1', 'client_2'])
            federated_data = []
            for client_id in client_ids:
                data = self.data_ingestor.load_processed_data(client_id)
                if data:
                    federated_data.append({
                        'client_id': client_id,
                        'features': data.get('features'),
                        'labels': data.get('labels')
                    })
            self.logger.debug(
                f"Loaded federated data for clients: {[client['client_id'] for client in federated_data]}")
            return federated_data
        except Exception as e:
            self.logger.error(f"Failed to load federated data: {e}", exc_info=True)
            return None

    def preprocess_data(self, client_data: List[Dict[str, Any]]) -> tff.simulation.ClientData:
        """
        Preprocesses data for federated learning.

        Args:
            client_data (List[Dict[str, Any]]): List of client data dictionaries.

        Returns:
            tff.simulation.ClientData: Preprocessed federated client data.
        """
        self.logger.info("Preprocessing federated data for training.")

        def create_tf_dataset(features, labels):
            return tf.data.Dataset.from_tensor_slices((tf.constant(features, dtype=tf.float32),
                                                       tf.constant(labels, dtype=tf.int32))).batch(20)

        # Create a mapping from client IDs to their datasets
        client_datasets = {client['client_id']: create_tf_dataset(client['features'], client['labels'])
                           for client in client_data}

        # Convert to tff.simulation.ClientData
        federated_data = tff.simulation.FromTensorSlicesClientData(client_datasets)
        self.logger.debug("Preprocessing completed.")
        return federated_data

    def build_model_fn(self):
        """
        Builds the model function for federated learning.

        Returns:
            Callable: A model function compatible with TFF.
        """
        self.logger.info("Building model function for federated learning.")

        def model_fn():
            model = tf.keras.models.Sequential([
                tf.keras.layers.Dense(10, activation='relu',
                                      input_shape=(self.config.get('input_shape', [None, 10])[1],)),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            return tff.learning.from_keras_model(
                model,
                input_spec=self.input_spec,
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[tf.keras.metrics.BinaryAccuracy()]
            )

        return model_fn

    def train_federated_model(self, federated_data: tff.simulation.ClientData) -> bool:
        """
        Trains the federated model using the provided client data.

        Args:
            federated_data (tff.simulation.ClientData): Preprocessed federated client data.

        Returns:
            bool: True if training was successful, False otherwise.
        """
        self.logger.info("Starting federated model training.")
        try:
            # Define model function
            model_fn = self.build_model_fn()

            # Define iterative process
            iterative_process = tff.learning.build_federated_averaging_process(model_fn)

            # Initialize the process
            state = iterative_process.initialize()
            self.logger.debug("Federated training initialized.")

            # Train for a specified number of rounds
            num_rounds = self.config.get('federated_training_rounds', 10)
            for round_num in range(1, num_rounds + 1):
                self.logger.info(f"Federated training round {round_num}/{num_rounds}.")
                state, metrics = iterative_process.next(state, federated_data.client_ids[:5])  # Select first 5 clients
                self.logger.info(f"Round {round_num}: {metrics}")

            # Save the final model
            self.save_federated_model(state.model)
            self.logger.info("Federated model training completed successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Federated model training failed: {e}", exc_info=True)
            return False

    def save_federated_model(self, model_weights: Any) -> bool:
        """
        Saves the trained federated model weights to disk.

        Args:
            model_weights (Any): The model weights to save.

        Returns:
            bool: True if the model was saved successfully, False otherwise.
        """
        self.logger.info(f"Saving federated model to '{self.model_save_path}'.")
        try:
            # Extract the underlying keras model
            keras_model = model_weights._keras_model
            os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
            keras_model.save(self.model_save_path)
            self.logger.info(f"Federated model saved successfully to '{self.model_save_path}'.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save federated model: {e}", exc_info=True)
            return False

    def run_federated_learning_pipeline(self) -> bool:
        """
        Runs the full federated learning pipeline: data loading, preprocessing, training, and saving.

        Returns:
            bool: True if the pipeline was successful, False otherwise.
        """
        self.logger.info("Running the federated learning pipeline.")
        try:
            client_data = self.load_federated_data()
            if client_data is None or not client_data:
                self.logger.error("No federated data available. Federated learning pipeline aborted.")
                return False

            federated_data = self.preprocess_data(client_data)
            training_success = self.train_federated_model(federated_data)

            if training_success:
                self.logger.info("Federated learning pipeline completed successfully.")
                return True
            else:
                self.logger.warning("Federated learning pipeline encountered issues.")
                return False
        except Exception as e:
            self.logger.error(f"Federated learning pipeline failed: {e}", exc_info=True)
            return False


# Example Usage and Test Cases
if __name__ == "__main__":
    # Initialize FederatedLearningManager
    project_id = "proj_12345"  # Replace with your actual project ID
    federated_manager = FederatedLearningManager(project_id)

    # Run the federated learning pipeline
    success = federated_manager.run_federated_learning_pipeline()
    print(f"Federated Learning Pipeline Successful: {success}")
