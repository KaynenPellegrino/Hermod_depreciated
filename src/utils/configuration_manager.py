import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    filename='hermod_configuration_manager.log',
    level=logging.DEBUG,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

class ConfigurationManager:
    """
    Manages application configurations dynamically, allowing for runtime updates and retrieval of configuration values.
    """
    def __init__(self):
        """
        Initializes the ConfigurationManager with default configurations.
        """
        self.configurations = {}
        logging.info("ConfigurationManager initialized.")

    def get_configuration(self, project_id: str) -> Dict[str, Any]:
        """
        Retrieves the configuration for a specified project.

        :param project_id: Unique identifier for the project
        :return: Configuration dictionary for the project
        """
        configuration = self.configurations.get(project_id, {})
        logging.debug(f"Retrieved configuration for project_id='{project_id}': {configuration}")
        return configuration

    def update_configuration(self, project_id: str, configuration: Dict[str, Any]) -> None:
        """
        Updates the configuration for a specified project.

        :param project_id: Unique identifier for the project
        :param configuration: Dictionary containing configuration updates
        """
        if project_id not in self.configurations:
            self.configurations[project_id] = configuration
            logging.info(f"Created new configuration for project_id='{project_id}': {configuration}")
        else:
            self.configurations[project_id].update(configuration)
            logging.info(f"Updated configuration for project_id='{project_id}': {configuration}")
        logging.debug(f"Current configuration for project_id='{project_id}': {self.configurations[project_id]}")

    def get_value(self, project_id: str, key: str) -> Any:
        """
        Retrieves a specific configuration value for a project using dot-notation.

        :param project_id: Unique identifier for the project
        :param key: Key of the configuration value to retrieve, using dot-notation (e.g., 'intent_classifier.spacy_model')
        :return: The configuration value, or None if the key does not exist
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

    def set_value(self, project_id: str, key: str, value: Any) -> None:
        """
        Sets a specific configuration value for a project using dot-notation.

        :param project_id: Unique identifier for the project
        :param key: Key of the configuration value to set, using dot-notation (e.g., 'intent_classifier.spacy_model')
        :param value: The value to set for the specified key
        """
        if project_id not in self.configurations:
            self.configurations[project_id] = {}
        keys = key.split('.')
        config = self.configurations[project_id]
        for k in keys[:-1]:
            if k not in config or not isinstance(config[k], dict):
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
        logging.info(f"Set value for key='{key}' in project_id='{project_id}' to '{value}'")
        logging.debug(f"Current configuration for project_id='{project_id}': {self.configurations[project_id]}")

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

    # Update a specific configuration value
    config_manager.set_value(project_id, "resource_allocation", {"cpu_cores": 8, "memory_gb": 16})

    # Retrieve a specific configuration value
    value = config_manager.get_value(project_id, "resource_allocation")
    print(f"Resource allocation for project '{project_id}': {value}")

    # Retrieve full configuration
    full_config = config_manager.get_configuration(project_id)
    print(f"Full configuration for project '{project_id}': {full_config}")