# data_management/federated_pipeline.py

import logging
import os
import sys
from typing import Dict, Any, Optional, List, Tuple
import requests
import json
import time
from threading import Thread
from concurrent.futures import ThreadPoolExecutor

# Configure logging with RotatingFileHandler to prevent log files from becoming too large
from logging.handlers import RotatingFileHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a rotating file handler
handler = RotatingFileHandler('logs/hermod_federated_pipeline.log', maxBytes=10**6, backupCount=5)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)


class NodeClient:
    """
    Represents a federated node with which the central orchestrator communicates.
    """

    def __init__(self, node_id: str, base_url: str, api_key: str):
        """
        Initializes the NodeClient.

        :param node_id: Unique identifier for the node
        :param base_url: Base URL of the node's API
        :param api_key: API key or token for authenticating with the node
        """
        self.node_id = node_id
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        logger.info(f"Initialized NodeClient for node '{self.node_id}' at '{self.base_url}'.")

    def send_task(self, task: str, data: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Sends a task to the node.

        :param task: The task to be performed (e.g., 'train_model', 'process_data')
        :param data: Optional data payload for the task
        :return: Response from the node if successful, else None
        """
        url = f"{self.base_url}/tasks/{task}"
        payload = data or {}
        try:
            logger.debug(f"Sending task '{task}' to node '{self.node_id}'. Payload: {payload}")
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            logger.info(f"Task '{task}' sent successfully to node '{self.node_id}'.")
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send task '{task}' to node '{self.node_id}': {e}")
            return None

    def get_model_update(self) -> Optional[Dict[str, Any]]:
        """
        Retrieves the latest model update from the node.

        :return: Model parameters if successful, else None
        """
        url = f"{self.base_url}/model/update"
        try:
            logger.debug(f"Retrieving model update from node '{self.node_id}'.")
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            logger.info(f"Model update retrieved successfully from node '{self.node_id}'.")
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to retrieve model update from node '{self.node_id}': {e}")
            return None

    def status(self) -> Optional[Dict[str, Any]]:
        """
        Checks the status of the node.

        :return: Status information if successful, else None
        """
        url = f"{self.base_url}/status"
        try:
            logger.debug(f"Checking status of node '{self.node_id}'.")
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            logger.info(f"Status of node '{self.node_id}': {response.json()}")
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to check status of node '{self.node_id}': {e}")
            return None


class FederatedPipeline:
    """
    Orchestrates data flow and model training across federated nodes.
    Ensures data privacy by coordinating distributed data processing tasks without sharing raw data.
    """

    def __init__(self, nodes: List[NodeClient]):
        """
        Initializes the FederatedPipeline with a list of federated nodes.

        :param nodes: List of NodeClient instances representing federated nodes
        """
        self.nodes = {node.node_id: node for node in nodes}
        self.global_model = None
        logger.info(f"FederatedPipeline initialized with {len(self.nodes)} nodes.")

    def register_node(self, node: NodeClient):
        """
        Registers a new federated node.

        :param node: NodeClient instance to be registered
        """
        if node.node_id in self.nodes:
            logger.warning(f"Node '{node.node_id}' is already registered.")
        else:
            self.nodes[node.node_id] = node
            logger.info(f"Node '{node.node_id}' registered successfully.")

    def deregister_node(self, node_id: str):
        """
        Deregisters a federated node.

        :param node_id: Unique identifier of the node to be deregistered
        """
        if node_id in self.nodes:
            del self.nodes[node_id]
            logger.info(f"Node '{node_id}' deregistered successfully.")
        else:
            logger.warning(f"Node '{node_id}' not found in the registry.")

    def distribute_task(self, task: str, data: Optional[Dict[str, Any]] = None):
        """
        Distributes a specific task to all registered nodes.

        :param task: The task to be distributed (e.g., 'train_model', 'process_data')
        :param data: Optional data payload for the task
        """
        logger.info(f"Distributing task '{task}' to all nodes.")
        with ThreadPoolExecutor(max_workers=len(self.nodes)) as executor:
            futures = {executor.submit(node.send_task, task, data): node_id for node_id, node in self.nodes.items()}
            for future in futures:
                node_id = futures[future]
                try:
                    result = future.result()
                    if result:
                        logger.debug(f"Task '{task}' completed by node '{node_id}'. Result: {result}")
                    else:
                        logger.warning(f"Task '{task}' failed on node '{node_id}'.")
                except Exception as e:
                    logger.error(f"Exception during task '{task}' on node '{node_id}': {e}")

    def collect_model_updates(self) -> List[Dict[str, Any]]:
        """
        Collects model updates from all nodes.

        :return: List of model parameter dictionaries
        """
        logger.info("Collecting model updates from all nodes.")
        model_updates = []
        with ThreadPoolExecutor(max_workers=len(self.nodes)) as executor:
            futures = {executor.submit(node.get_model_update): node_id for node_id, node in self.nodes.items()}
            for future in futures:
                node_id = futures[future]
                try:
                    update = future.result()
                    if update and 'model_parameters' in update:
                        model_updates.append(update['model_parameters'])
                        logger.debug(f"Model update received from node '{node_id}'.")
                    else:
                        logger.warning(f"No model update received from node '{node_id}'.")
                except Exception as e:
                    logger.error(f"Exception while collecting model update from node '{node_id}': {e}")
        logger.info(f"Collected {len(model_updates)} model updates.")
        return model_updates

    def aggregate_models(self, model_updates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Aggregates model updates using Federated Averaging.

        :param model_updates: List of model parameter dictionaries from nodes
        :return: Aggregated model parameters if successful, else None
        """
        if not model_updates:
            logger.error("No model updates to aggregate.")
            return None

        logger.info("Aggregating model updates using Federated Averaging.")
        try:
            # Initialize aggregated parameters with zeros
            aggregated_params = {}
            num_updates = len(model_updates)

            # Assuming all models have the same structure
            for key in model_updates[0]:
                aggregated_params[key] = sum(update.get(key, 0) for update in model_updates) / num_updates

            logger.info("Model aggregation completed successfully.")
            return aggregated_params
        except Exception as e:
            logger.error(f"Error during model aggregation: {e}")
            return None

    def update_global_model(self, aggregated_params: Dict[str, Any], model_save_path: str = 'model/global_model.json') -> bool:
        """
        Updates the global model with aggregated parameters.

        :param aggregated_params: Aggregated model parameters
        :param model_save_path: Path to save the global model parameters
        :return: True if successful, else False
        """
        try:
            logger.info(f"Updating global model at '{model_save_path}'.")
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            with open(model_save_path, 'w') as f:
                json.dump(aggregated_params, f)
            self.global_model = aggregated_params
            logger.info("Global model updated successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to update global model: {e}")
            return False

    def broadcast_global_model(self, model_path: str):
        """
        Broadcasts the updated global model to all nodes.

        :param model_path: Path to the global model parameters file
        """
        logger.info("Broadcasting global model to all nodes.")
        try:
            with open(model_path, 'r') as f:
                model_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load global model from '{model_path}': {e}")
            return

        with ThreadPoolExecutor(max_workers=len(self.nodes)) as executor:
            futures = {executor.submit(node.send_task, 'update_model', {'model_parameters': model_data}): node_id for node_id, node in self.nodes.items()}
            for future in futures:
                node_id = futures[future]
                try:
                    result = future.result()
                    if result and result.get('status') == 'success':
                        logger.debug(f"Global model updated on node '{node_id}'.")
                    else:
                        logger.warning(f"Failed to update global model on node '{node_id}'.")
                except Exception as e:
                    logger.error(f"Exception while broadcasting model to node '{node_id}': {e}")

    def synchronize_models(self, model_save_path: str = 'model/global_model.json'):
        """
        Synchronizes models across all nodes by aggregating updates and broadcasting the global model.

        :param model_save_path: Path to save the global model parameters
        """
        logger.info("Synchronizing models across all nodes.")
        model_updates = self.collect_model_updates()
        aggregated_params = self.aggregate_models(model_updates)
        if aggregated_params:
            if self.update_global_model(aggregated_params, model_save_path=model_save_path):
                self.broadcast_global_model(model_save_path=model_save_path)
            else:
                logger.error("Failed to update the global model.")
        else:
            logger.error("Model synchronization aborted due to aggregation failure.")

    def run_federated_training(self, rounds: int = 5, model_save_path: str = 'model/global_model.json'):
        """
        Executes the federated training process over a specified number of rounds.

        :param rounds: Number of federated training rounds
        :param model_save_path: Path to save the global model parameters
        """
        logger.info(f"Starting federated training for {rounds} rounds.")
        for round_num in range(1, rounds + 1):
            logger.info(f"--- Federated Training Round {round_num} ---")
            # Step 1: Distribute training task
            self.distribute_task('train_model')

            # Allow some time for nodes to train and send updates
            logger.info("Waiting for nodes to complete training...")
            time.sleep(60)  # Adjust sleep time based on expected node training duration

            # Step 2: Synchronize models
            self.synchronize_models(model_save_path=model_save_path)

            logger.info(f"--- Federated Training Round {round_num} Completed ---\n")

        logger.info("Federated training process completed.")

    def get_global_model(self) -> Optional[Dict[str, Any]]:
        """
        Retrieves the current global model parameters.

        :return: Global model parameters if available, else None
        """
        if self.global_model:
            return self.global_model
        else:
            logger.warning("Global model is not yet available.")
            return None


# Example usage and test cases
if __name__ == "__main__":
    # Example NodeClient instances (replace with actual node details)
    node1 = NodeClient(
        node_id='node_1',
        base_url='http://localhost:5001',
        api_key='node1_api_key'
    )
    node2 = NodeClient(
        node_id='node_2',
        base_url='http://localhost:5002',
        api_key='node2_api_key'
    )
    node3 = NodeClient(
        node_id='node_3',
        base_url='http://localhost:5003',
        api_key='node3_api_key'
    )

    # Initialize FederatedPipeline with nodes
    federated_pipeline = FederatedPipeline(nodes=[node1, node2, node3])

    # Optionally, register additional nodes
    # new_node = NodeClient(node_id='node_4', base_url='http://localhost:5004', api_key='node4_api_key')
    # federated_pipeline.register_node(new_node)

    # Run federated training for 3 rounds
    federated_pipeline.run_federated_training(rounds=3, model_save_path='model/global_model.json')

    # Retrieve the final global model
    global_model = federated_pipeline.get_global_model()
    if global_model:
        print("Final Global Model Parameters:")
        print(global_model)
    else:
        print("Global model is not available.")
