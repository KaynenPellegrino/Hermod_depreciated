# src/modules/self_optimization/model_registry.py

import os
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

import joblib
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from utils.logger import get_logger
from utils.configuration_manager import ConfigurationManager

Base = declarative_base()


class Model(Base):
    """
    SQLAlchemy model for storing model metadata.
    """
    __tablename__ = 'models'

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String, nullable=False)
    version = Column(String, nullable=False)
    performance = Column(Float, nullable=True)
    metrics = Column(Text, nullable=True)  # JSON string of various metrics
    description = Column(Text, nullable=True)
    path = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class ModelRegistry:
    """
    Manages the registration, retrieval, and lifecycle of AI models.
    """

    def __init__(self, project_id: str):
        """
        Initializes the ModelRegistry with necessary configurations.

        Args:
            project_id (str): Unique identifier for the project.
        """
        self.logger = get_logger(__name__)
        self.project_id = project_id
        self.config_manager = ConfigurationManager()
        self.config = self.config_manager.get_configuration(project_id)

        self.registry_dir = self.config.get('model_registry_dir', f'models/{project_id}/registry')
        os.makedirs(self.registry_dir, exist_ok=True)

        # Initialize SQLite database for metadata
        self.db_path = os.path.join(self.registry_dir, 'model_registry.db')
        self.engine = create_engine(f'sqlite:///{self.db_path}')
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.logger.info(f"ModelRegistry initialized for project '{project_id}'. Database at '{self.db_path}'.")

    def register_model(self, model, model_name: str, version: str, performance: Optional[float] = None,
                      metrics: Optional[Dict[str, Any]] = None, description: Optional[str] = None) -> bool:
        """
        Registers a new model in the registry.

        Args:
            model: The trained model object to be saved.
            model_name (str): Name of the model.
            version (str): Version identifier (e.g., 'v1.0', 'v1.1').
            performance (Optional[float]): Performance metric (e.g., accuracy).
            metrics (Optional[Dict[str, Any]]): Additional performance metrics.
            description (Optional[str]): Description of the model.

        Returns:
            bool: True if registration is successful, False otherwise.
        """
        self.logger.info(f"Registering model '{model_name}' version '{version}'.")
        try:
            # Define model file path
            model_filename = f"{model_name}_{version}.joblib"
            model_path = os.path.join(self.registry_dir, model_filename)

            # Save the model to the filesystem
            joblib.dump(model, model_path)
            self.logger.info(f"Model saved at '{model_path}'.")

            # Convert metrics dict to JSON string
            metrics_str = joblib.dumps(metrics) if metrics else None

            # Create a new Model instance
            new_model = Model(
                model_name=model_name,
                version=version,
                performance=performance,
                metrics=joblib.dumps(metrics) if metrics else None,
                description=description,
                path=model_path
            )

            # Add to the database
            session = self.Session()
            session.add(new_model)
            session.commit()
            session.close()

            self.logger.info(f"Model '{model_name}' version '{version}' registered successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to register model '{model_name}' version '{version}': {e}", exc_info=True)
            return False

    def get_model(self, model_name: str, version: Optional[str] = None):
        """
        Retrieves a model from the registry.

        Args:
            model_name (str): Name of the model.
            version (Optional[str]): Specific version to retrieve. If None, retrieves the latest version.

        Returns:
            Optional[Any]: The loaded model object, or None if not found.
        """
        self.logger.info(f"Retrieving model '{model_name}' version '{version}'.")
        try:
            session = self.Session()
            query = session.query(Model).filter(Model.model_name == model_name)
            if version:
                query = query.filter(Model.version == version)
            else:
                # Retrieve the latest version based on creation time
                query = query.order_by(Model.created_at.desc())
            model_entry = query.first()
            session.close()

            if not model_entry:
                self.logger.warning(f"Model '{model_name}' version '{version}' not found.")
                return None

            # Load the model from the filesystem
            loaded_model = joblib.load(model_entry.path)
            self.logger.info(f"Model '{model_name}' version '{model_entry.version}' loaded successfully.")
            return loaded_model
        except Exception as e:
            self.logger.error(f"Failed to retrieve model '{model_name}' version '{version}': {e}", exc_info=True)
            return None

    def update_performance(self, model_name: str, version: str, performance: float, metrics: Optional[Dict[str, Any]] = None) -> bool:
        """
        Updates the performance metrics of a registered model.

        Args:
            model_name (str): Name of the model.
            version (str): Version of the model.
            performance (float): Updated performance metric.
            metrics (Optional[Dict[str, Any]]): Updated additional metrics.

        Returns:
            bool: True if update is successful, False otherwise.
        """
        self.logger.info(f"Updating performance for model '{model_name}' version '{version}'.")
        try:
            session = self.Session()
            model_entry = session.query(Model).filter(Model.model_name == model_name, Model.version == version).first()
            if not model_entry:
                self.logger.warning(f"Model '{model_name}' version '{version}' not found for performance update.")
                session.close()
                return False

            model_entry.performance = performance
            model_entry.metrics = joblib.dumps(metrics) if metrics else model_entry.metrics
            session.commit()
            session.close()

            self.logger.info(f"Performance for model '{model_name}' version '{version}' updated successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to update performance for model '{model_name}' version '{version}': {e}", exc_info=True)
            return False

    def list_models(self, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Lists all registered models or filters by model name.

        Args:
            model_name (Optional[str]): Name of the model to filter by.

        Returns:
            List[Dict[str, Any]]: List of model metadata dictionaries.
        """
        self.logger.info(f"Listing models with name filter: '{model_name}'.")
        try:
            session = self.Session()
            if model_name:
                models = session.query(Model).filter(Model.model_name == model_name).all()
            else:
                models = session.query(Model).all()
            session.close()

            model_list = []
            for m in models:
                model_info = {
                    'model_name': m.model_name,
                    'version': m.version,
                    'performance': m.performance,
                    'metrics': joblib.loads(m.metrics) if m.metrics else None,
                    'description': m.description,
                    'path': m.path,
                    'created_at': m.created_at.isoformat()
                }
                model_list.append(model_info)

            self.logger.debug(f"Retrieved {len(model_list)} models from the registry.")
            return model_list
        except Exception as e:
            self.logger.error(f"Failed to list models: {e}", exc_info=True)
            return []

    def rollback_model(self, model_name: str, target_version: str) -> bool:
        """
        Rolls back to a specified version of a model.

        Args:
            model_name (str): Name of the model.
            target_version (str): The version to rollback to.

        Returns:
            bool: True if rollback is successful, False otherwise.
        """
        self.logger.info(f"Rolling back model '{model_name}' to version '{target_version}'.")
        try:
            target_model = self.get_model(model_name, target_version)
            if not target_model:
                self.logger.warning(f"Target model '{model_name}' version '{target_version}' not found for rollback.")
                return False

            # Optionally, set the target model as the current model by creating a new version
            # Here, we'll assume that the latest model is considered the current one
            latest_models = [m for m in self.list_models(model_name=model_name)]
            latest_versions = sorted([m['version'] for m in latest_models])
            new_version = f"{latest_versions[-1].rsplit('.', 1)[0]}.{int(latest_versions[-1].rsplit('.', 1)[1]) + 1}"

            # Register the target model as the new version
            success = self.register_model(
                model=target_model,
                model_name=model_name,
                version=new_version,
                performance=target_model.get('performance', None),
                metrics=target_model.get('metrics', None),
                description=f"Rollback to version {target_version}"
            )

            if success:
                self.logger.info(f"Model '{model_name}' rolled back to version '{target_version}' as new version '{new_version}'.")
            return success
        except Exception as e:
            self.logger.error(f"Failed to rollback model '{model_name}' to version '{target_version}': {e}", exc_info=True)
            return False

    def delete_model(self, model_name: str, version: str) -> bool:
        """
        Deletes a specific version of a model from the registry.

        Args:
            model_name (str): Name of the model.
            version (str): Version of the model to delete.

        Returns:
            bool: True if deletion is successful, False otherwise.
        """
        self.logger.info(f"Deleting model '{model_name}' version '{version}'.")
        try:
            session = self.Session()
            model_entry = session.query(Model).filter(Model.model_name == model_name, Model.version == version).first()
            if not model_entry:
                self.logger.warning(f"Model '{model_name}' version '{version}' not found for deletion.")
                session.close()
                return False

            # Delete the model file from the filesystem
            if os.path.exists(model_entry.path):
                os.remove(model_entry.path)
                self.logger.info(f"Model file '{model_entry.path}' deleted.")
            else:
                self.logger.warning(f"Model file '{model_entry.path}' does not exist.")

            # Delete the metadata entry from the database
            session.delete(model_entry)
            session.commit()
            session.close()

            self.logger.info(f"Model '{model_name}' version '{version}' deleted successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete model '{model_name}' version '{version}': {e}", exc_info=True)
            return False

    def run_sample_operations(self):
        """
        Runs sample operations to demonstrate usage of ModelRegistry.
        """
        self.logger.info("Running sample operations on ModelRegistry.")

        # Example: Register a new model
        try:
            sample_model = SimpleClassifier(input_size=10)  # Replace with your actual model
            sample_model.fit([[0]*10], [0])  # Dummy fit
            self.register_model(
                model=sample_model,
                model_name='sample_model',
                version='v1.0',
                performance=0.95,
                metrics={'accuracy': 0.95},
                description='Sample model for demonstration purposes.'
            )
        except Exception as e:
            self.logger.error(f"Sample model registration failed: {e}")

        # Example: List all models
        models = self.list_models()
        self.logger.info(f"Registered Models: {models}")

        # Example: Retrieve a model
        retrieved_model = self.get_model('sample_model', 'v1.0')
        self.logger.info(f"Retrieved Model: {retrieved_model}")

        # Example: Update performance
        self.update_performance('sample_model', 'v1.0', performance=0.96, metrics={'accuracy': 0.96})

        # Example: Rollback model
        self.rollback_model('sample_model', 'v1.0')

        # Example: Delete a model
        self.delete_model('sample_model', 'v1.0')


# Example SimpleClassifier for demonstration purposes
class SimpleClassifier:
    """
    A simple classifier model for demonstration purposes.
    Replace with your actual model implementation.
    """
    def __init__(self, input_size: int = 10):
        self.input_size = input_size
        self.model = None  # Replace with actual model (e.g., scikit-learn, PyTorch)

    def fit(self, X, y):
        # Dummy fit method
        self.model = "TrainedModel"

    def predict(self, X):
        return [0] * len(X)

    def __repr__(self):
        return f"SimpleClassifier(input_size={self.input_size})"
