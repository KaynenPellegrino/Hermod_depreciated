# src/modules/deployment/environment_manager.py

import os
import logging
from typing import Optional, Dict, Any
from src.utils.configuration_manager import ConfigurationManager
from src.modules.notifications.notification_manager import NotificationManager

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('logs/environment_manager.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class EnvironmentManager:
    """
    Environment Configuration Management
    Manages different deployment environments (development, staging, production),
    handling environment-specific configurations, variables, and resources.
    """

    def __init__(self):
        """
        Initializes the EnvironmentManager with necessary configurations.
        """
        try:
            self.config_manager = ConfigurationManager()
            self.notification_manager = NotificationManager()
            self.load_environment_config()
            logger.info("EnvironmentManager initialized successfully.")
        except Exception as e:
            logger.exception(f"Failed to initialize EnvironmentManager: {e}")
            raise e

    def load_environment_config(self):
        """
        Loads environment configurations from the configuration manager or environment variables.
        """
        logger.info("Loading environment configurations.")
        try:
            self.environments = self.config_manager.get('ENVIRONMENTS', ['development', 'staging', 'production'])
            self.current_environment = self.config_manager.get('CURRENT_ENVIRONMENT', 'development')
            self.environment_configs = {}

            for env in self.environments:
                env_prefix = env.upper()
                self.environment_configs[env] = {
                    'database_url': self.config_manager.get(f'{env_prefix}_DATABASE_URL', ''),
                    'api_endpoint': self.config_manager.get(f'{env_prefix}_API_ENDPOINT', ''),
                    'secret_key': self.config_manager.get(f'{env_prefix}_SECRET_KEY', ''),
                    'other_settings': self.config_manager.get(f'{env_prefix}_OTHER_SETTINGS', {}),
                }

            logger.info(f"Environment configurations loaded for environments: {self.environments}")
        except Exception as e:
            logger.error(f"Failed to load environment configurations: {e}")
            raise e

    def get_current_config(self) -> Dict[str, Any]:
        """
        Retrieves the configuration for the current environment.

        :return: Dictionary of configuration settings.
        """
        logger.info(f"Retrieving configuration for environment: {self.current_environment}")
        try:
            config = self.environment_configs.get(self.current_environment)
            if not config:
                raise ValueError(f"No configuration found for environment '{self.current_environment}'")
            return config
        except Exception as e:
            logger.error(f"Failed to retrieve current environment configuration: {e}")
            raise e

    def switch_environment(self, environment: str):
        """
        Switches the current environment to the specified one.

        :param environment: The environment to switch to.
        """
        logger.info(f"Switching environment to: {environment}")
        try:
            if environment not in self.environments:
                raise ValueError(f"Environment '{environment}' is not defined.")
            self.current_environment = environment
            logger.info(f"Environment switched to '{environment}'.")
        except Exception as e:
            logger.error(f"Failed to switch environment: {e}")
            raise e

    def get_environment_variable(self, key: str) -> Any:
        """
        Retrieves an environment-specific variable.

        :param key: The key of the variable.
        :return: The value of the variable.
        """
        logger.info(f"Retrieving variable '{key}' for environment '{self.current_environment}'")
        try:
            config = self.get_current_config()
            value = config.get(key)
            if value is None:
                raise KeyError(f"Variable '{key}' not found in environment '{self.current_environment}'")
            return value
        except Exception as e:
            logger.error(f"Failed to retrieve environment variable: {e}")
            raise e

    def set_environment_variable(self, key: str, value: Any):
        """
        Sets an environment-specific variable.

        :param key: The key of the variable.
        :param value: The value to set.
        """
        logger.info(f"Setting variable '{key}' to '{value}' in environment '{self.current_environment}'")
        try:
            config = self.environment_configs.get(self.current_environment)
            if config is None:
                raise ValueError(f"Environment '{self.current_environment}' is not configured.")
            config[key] = value
            logger.info(f"Variable '{key}' set to '{value}' in environment '{self.current_environment}'.")
        except Exception as e:
            logger.error(f"Failed to set environment variable: {e}")
            raise e

    def apply_environment_settings(self):
        """
        Applies the current environment's settings to the application.
        """
        logger.info(f"Applying settings for environment '{self.current_environment}'")
        try:
            config = self.get_current_config()
            os.environ.update(config)
            logger.info(f"Environment settings applied for '{self.current_environment}'.")
        except Exception as e:
            logger.error(f"Failed to apply environment settings: {e}")
            raise e

    def notify_environment_change(self):
        """
        Sends a notification about the environment change.
        """
        logger.info(f"Notifying about environment change to '{self.current_environment}'")
        try:
            subject = f"Environment Switched to '{self.current_environment}'"
            message = f"The application environment has been switched to '{self.current_environment}'."
            recipients = self.config_manager.get('NOTIFICATION_RECIPIENTS', '').split(',')

            if recipients:
                self.notification_manager.send_notification(
                    recipients=recipients,
                    subject=subject,
                    message=message
                )
                logger.info("Environment change notification sent successfully.")
            else:
                logger.warning("No notification recipients configured.")
        except Exception as e:
            logger.error(f"Failed to send environment change notification: {e}")

    # --------------------- Example Usage --------------------- #

    def example_usage(self):
        """
        Demonstrates example usage of the EnvironmentManager class.
        """
        try:
            # Initialize EnvironmentManager
            env_manager = EnvironmentManager()

            # Get current environment configuration
            current_config = env_manager.get_current_config()
            logger.info(f"Current environment configuration: {current_config}")

            # Switch to a different environment
            env_manager.switch_environment('staging')
            env_manager.apply_environment_settings()
            env_manager.notify_environment_change()

            # Retrieve a specific variable
            api_endpoint = env_manager.get_environment_variable('api_endpoint')
            logger.info(f"API Endpoint for current environment: {api_endpoint}")

            # Set a new variable
            env_manager.set_environment_variable('debug_mode', True)

            # Get updated configuration
            updated_config = env_manager.get_current_config()
            logger.info(f"Updated environment configuration: {updated_config}")

        except Exception as e:
            logger.exception(f"Error in example usage: {e}")

    # --------------------- Main Execution --------------------- #

    if __name__ == "__main__":
        # Run the environment manager example
        example_usage()
