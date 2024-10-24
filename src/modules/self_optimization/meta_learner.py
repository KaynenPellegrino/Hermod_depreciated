# src/modules/self_optimization/meta_learner.py

import logging
import os
import copy
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from src.utils.logger import get_logger
from src.utils.configuration_manager import ConfigurationManager
from src.modules.data_management.data_ingestor import DataIngestor


class MetaDataset(Dataset):
    """
    A simple dataset class for meta-learning tasks.
    """

    def __init__(self, features: List[List[float]], labels: List[int]):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)


class SimpleClassifier(nn.Module):
    """
    A simple neural network classifier for demonstration.
    """

    def __init__(self, input_size: int = 10, hidden_size: int = 40, num_classes: int = 2):
        super(SimpleClassifier, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        return out


class MetaLearner:
    """
    Implements meta-learning techniques (e.g., MAML) to enable Hermod to adapt to new tasks with minimal data.
    """

    def __init__(self, project_id: str):
        """
        Initializes the MetaLearner with necessary configurations.

        Args:
            project_id (str): Unique identifier for the project.
        """
        self.logger = get_logger(__name__)
        self.project_id = project_id
        self.config_manager = ConfigurationManager()
        self.config = self.config_manager.get_configuration(project_id)

        self.meta_model_save_path = self.config.get('meta_model_save_path', f'models/{project_id}_meta_model.pth')
        self.meta_lr = self.config.get('meta_learning_rate', 0.001)
        self.inner_lr = self.config.get('inner_learning_rate', 0.01)
        self.meta_epochs = self.config.get('meta_epochs', 100)
        self.task_batch_size = self.config.get('task_batch_size', 5)
        self.support_set_size = self.config.get('support_set_size', 10)
        self.query_set_size = self.config.get('query_set_size', 10)

        self.data_ingestor = DataIngestor(project_id)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"MetaLearner initialized on device '{self.device}'.")

    def load_tasks(self) -> List[Dict[str, Any]]:
        """
        Loads a list of tasks for meta-training.

        Returns:
            List[Dict[str, Any]]: List of task dictionaries containing 'features' and 'labels'.
        """
        self.logger.info("Loading meta-learning tasks.")
        try:
            # Placeholder: Assume data_ingestor can load multiple tasks
            tasks = self.data_ingestor.load_meta_tasks()
            self.logger.debug(f"Loaded {len(tasks)} tasks for meta-learning.")
            return tasks
        except Exception as e:
            self.logger.error(f"Failed to load meta-learning tasks: {e}", exc_info=True)
            return []

    def create_dataloaders(self, task: Dict[str, Any]) -> Dict[str, DataLoader]:
        """
        Creates support and query dataloaders for a given task.

        Args:
            task (Dict[str, Any]): A task dictionary containing 'features' and 'labels'.

        Returns:
            Dict[str, DataLoader]: Dictionary with 'support' and 'query' DataLoaders.
        """
        support_features = task['support']['features']
        support_labels = task['support']['labels']
        query_features = task['query']['features']
        query_labels = task['query']['labels']

        support_dataset = MetaDataset(support_features, support_labels)
        query_dataset = MetaDataset(query_features, query_labels)

        support_loader = DataLoader(support_dataset, batch_size=self.support_set_size, shuffle=True)
        query_loader = DataLoader(query_dataset, batch_size=self.query_set_size, shuffle=True)

        return {'support': support_loader, 'query': query_loader}

    def train_on_task(self, model: nn.Module, support_loader: DataLoader) -> nn.Module:
        """
        Performs inner-loop training on a single task's support set.

        Args:
            model (nn.Module): The model to train.
            support_loader (DataLoader): Support set DataLoader.

        Returns:
            nn.Module: The adapted model after inner-loop training.
        """
        model = copy.deepcopy(model)
        model.to(self.device)
        optimizer = optim.SGD(model.parameters(), lr=self.inner_lr)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for batch in support_loader:
            features, labels = batch
            features, labels = features.to(self.device), labels.to(self.device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        return model

    def evaluate_on_task(self, model: nn.Module, query_loader: DataLoader) -> float:
        """
        Evaluates the adapted model on a single task's query set.

        Args:
            model (nn.Module): The adapted model.
            query_loader (DataLoader): Query set DataLoader.

        Returns:
            float: Accuracy on the query set.
        """
        model.to(self.device)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in query_loader:
                features, labels = batch
                features, labels = features.to(self.device), labels.to(self.device)
                outputs = model(features)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        return accuracy

    def meta_train(self):
        """
        Performs meta-training using MAML.
        """
        self.logger.info("Starting meta-training.")
        tasks = self.load_tasks()
        if not tasks:
            self.logger.error("No tasks available for meta-training.")
            return

        model = SimpleClassifier(input_size=self.config.get('input_size', 10)).to(self.device)
        meta_optimizer = optim.Adam(model.parameters(), lr=self.meta_lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(1, self.meta_epochs + 1):
            meta_loss = 0.0
            meta_accuracy = 0.0
            for task in tasks:
                dataloaders = self.create_dataloaders(task)
                support_loader = dataloaders['support']
                query_loader = dataloaders['query']

                # Inner loop: Adapt to task
                adapted_model = self.train_on_task(model, support_loader)

                # Outer loop: Update meta-model based on query performance
                adapted_model.eval()
                query_loss = 0.0
                correct = 0
                total = 0
                for batch in query_loader:
                    features, labels = batch
                    features, labels = features.to(self.device), labels.to(self.device)
                    outputs = adapted_model(features)
                    loss = criterion(outputs, labels)
                    query_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                loss = query_loss / len(query_loader)
                accuracy = correct / total

                meta_loss += loss
                meta_accuracy += accuracy

                # Compute gradients w.r.t. meta-model parameters
                meta_loss_tensor = torch.tensor(loss, requires_grad=True).to(self.device)
                meta_optimizer.zero_grad()
                meta_loss_tensor.backward()
                meta_optimizer.step()

            avg_meta_loss = meta_loss / len(tasks)
            avg_meta_accuracy = meta_accuracy / len(tasks)
            self.logger.info(
                f"Epoch {epoch}/{self.meta_epochs}: Meta Loss={avg_meta_loss:.4f}, Meta Accuracy={avg_meta_accuracy:.4f}")

        # Save the meta-trained model
        self.save_meta_model(model)
        self.logger.info("Meta-training completed and model saved successfully.")

    def save_meta_model(self, model: nn.Module) -> bool:
        """
        Saves the meta-trained model to disk.

        Args:
            model (nn.Module): The meta-trained model.

        Returns:
            bool: True if the model was saved successfully, False otherwise.
        """
        self.logger.info(f"Saving meta-trained model to '{self.meta_model_save_path}'.")
        try:
            os.makedirs(os.path.dirname(self.meta_model_save_path), exist_ok=True)
            torch.save(model.state_dict(), self.meta_model_save_path)
            self.logger.info(f"Meta-trained model saved successfully to '{self.meta_model_save_path}'.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save meta-trained model: {e}", exc_info=True)
            return False

    def run_meta_learning_pipeline(self) -> bool:
        """
        Runs the full meta-learning pipeline: meta-training and model saving.

        Returns:
            bool: True if the pipeline was successful, False otherwise.
        """
        self.logger.info("Running the meta-learning pipeline.")
        try:
            self.meta_train()
            self.logger.info("Meta-learning pipeline completed successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Meta-learning pipeline failed: {e}", exc_info=True)
            return False


# Example Usage and Test Cases
if __name__ == "__main__":
    # Initialize MetaLearner
    project_id = "proj_12345"  # Replace with your actual project ID
    meta_learner = MetaLearner(project_id)

    # Run the meta-learning pipeline
    success = meta_learner.run_meta_learning_pipeline()
    print(f"Meta-Learning Pipeline Successful: {success}")
