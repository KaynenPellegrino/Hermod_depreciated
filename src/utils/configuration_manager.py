# src/utils/configuration_manager.py

import logging
from typing import Dict, Any, Optional

import yaml

from utils.logger import get_logger

# Configure logging
logging.basicConfig(
    filename='hermod_configuration_manager.log',
    level=logging.DEBUG,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

class ConfigurationManager:
    """
    Manages application configurations dynamically, allowing for runtime updates and retrieval of configuration values.
    Supports loading from and saving to a persistent storage (JSON file).
    """
    def __init__(self, config_file: str = 'configurations.json'):
        """
        Initializes the ConfigurationManager with default configurations.
        Loads existing configurations from a file if it exists.

        Args:
            config_file (str): Path to the configuration file.
        """
        self.logger = get_logger(__name__)
        self.config_file = config_file
        self.config_file = config_file
        self.configurations = {}
        self.configurations = self.load_configuration()
        logging.info("ConfigurationManager initialized.")

    def load_configuration(self) -> Dict[str, Any]:
        """
        Loads configurations from the YAML configuration file.

        Returns:
            Dict[str, Any]: Dictionary containing all configurations.
        """
        try:
            with open(self.config_file, 'r') as file:
                configurations = yaml.safe_load(file)
            self.logger.info(f"Loaded configurations from '{self.config_file}'.")
            return configurations
        except Exception as e:
            self.logger.error(f"Failed to load configuration file '{self.config_file}': {e}")
            raise e

    def get_configuration(self, project_id: str) -> Dict[str, Any]:
        """
        Retrieves configuration settings for a specific project.

        Args:
            project_id (str): Identifier for the project.

        Returns:
            Dict[str, Any]: Dictionary containing project-specific configurations.
        """
        try:
            project_config = self.configurations.get(project_id, {})
            if not project_config:
                self.logger.warning(f"No specific configurations found for project '{project_id}'. Using defaults.")
            return project_config
        except Exception as e:
            self.logger.error(f"Error retrieving configuration for project '{project_id}': {e}")
            return {}

    def update_configuration(self, project_id: str, configuration: Dict[str, Any]) -> None:
        """
        Updates the configuration for a specified project.

        Args:
            project_id (str): Unique identifier for the project.
            configuration (Dict[str, Any]): Dictionary containing configuration updates.
        """
        if project_id not in self.configurations:
            self.configurations[project_id] = configuration
            logging.info(f"Created new configuration for project_id='{project_id}': {configuration}")
        else:
            self.configurations[project_id].update(configuration)
            logging.info(f"Updated configuration for project_id='{project_id}': {configuration}")
        logging.debug(f"Current configuration for project_id='{project_id}': {self.configurations[project_id]}")

    def get_value(self, project_id: str, key: str) -> Optional[Any]:
        """
        Retrieves a specific configuration value for a project using dot-notation.

        Args:
            project_id (str): Unique identifier for the project.
            key (str): Key of the configuration value to retrieve, using dot-notation (e.g., 'intent_classifier.spacy_model').

        Returns:
            Optional[Any]: The configuration value, or None if the key does not exist.
        """
        configuration = self.get_configuration(project_id)
        keys = key.split('.')
        value = configuration
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                logging.debug(f"Key '{key}' not found in project_id='{project_id}'.")
                return None
        logging.debug(f"Retrieved value for key='{key}' in project_id='{project_id}': {value}")
        return value

    def set_value(self, project_id: str, key: str, value: Any):
        """
        Sets a configuration value for a specific project.

        Args:
            project_id (str): Identifier for the project.
            key (str): Configuration key to set.
            value (Any): Value to set for the configuration key.
        """
        try:
            if project_id not in self.configurations:
                self.configurations[project_id] = {}
            self.configurations[project_id][key] = value
            self.logger.info(f"Set configuration '{key}' to '{value}' for project '{project_id}'.")
        except Exception as e:
            self.logger.error(f"Failed to set configuration '{key}' for project '{project_id}': {e}")

    def save_configuration(self, project_id: str):
        """
        Saves the current configurations back to the YAML configuration file.

        Args:
            project_id (str): Identifier for the project.
        """
        try:
            with open(self.config_file, 'w') as file:
                yaml.dump(self.configurations, file)
            self.logger.info(f"Saved configurations for project '{project_id}' to '{self.config_file}'.")
        except Exception as e:
            self.logger.error(f"Failed to save configurations to '{self.config_file}': {e}")

# Example usage
if __name__ == "__main__":
    config_manager = ConfigurationManager()

    # Define project ID
    project_id = "proj_12345"

    # Set initial configuration
    initial_config = {
        "resource_allocation": {
            "cpu_cores": 4,
            "memory_gb": 8
        },
        "scaling_policies": {
            "auto_scale": True,
            "max_instances": 10
        }
    }
    config_manager.update_configuration(project_id, initial_config)

    # Save configurations to file
    config_manager.save_configuration()

    # Update a specific configuration value
    config_manager.set_value(project_id, "resource_allocation.cpu_cores", 8)
    config_manager.set_value(project_id, "resource_allocation.memory_gb", 16)

    # Save configurations after update
    config_manager.save_configuration()

    # Retrieve a specific configuration value
    value = config_manager.get_value(project_id, "resource_allocation")
    print(f"Resource allocation for project '{project_id}': {value}")

    # Retrieve full configuration
    full_config = config_manager.get_configuration(project_id)
    print(f"Full configuration for project '{project_id}': {full_config}")
