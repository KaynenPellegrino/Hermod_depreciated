# src/modules/ui_ux/customization_handler.py

import os
import json
import logging
from typing import Dict, Any
from src.utils.configuration_manager import ConfigurationManager
from src.modules.notifications.notification_manager import NotificationManager

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('logs/customization_handler.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class CustomizationHandler:
    """
    User Interface Customization
    Manages customization options for the user interface, allowing users to personalize their experience,
    such as themes, layouts, and preferences.
    """

    def __init__(self):
        """
        Initializes the CustomizationHandler with necessary configurations.
        """
        try:
            self.config_manager = ConfigurationManager()
            self.notification_manager = NotificationManager()
            self.load_customization_config()
            self.user_preferences: Dict[str, Dict[str, Any]] = {}
            self.preferences_file = self.customization_config['preferences_file']
            self.load_user_preferences()
            logger.info("CustomizationHandler initialized successfully.")
        except Exception as e:
            logger.exception(f"Failed to initialize CustomizationHandler: {e}")
            raise e

    def load_customization_config(self):
        """
        Loads customization configurations from the configuration manager or environment variables.
        """
        logger.info("Loading customization configurations.")
        try:
            self.customization_config = {
                'preferences_file': self.config_manager.get('PREFERENCES_FILE', 'data/user_preferences.json'),
                'default_theme': self.config_manager.get('DEFAULT_THEME', 'light'),
                'default_layout': self.config_manager.get('DEFAULT_LAYOUT', 'standard'),
                'notification_recipients': self.config_manager.get('NOTIFICATION_RECIPIENTS', '').split(','),
            }
            logger.info(f"Customization configurations loaded: {self.customization_config}")
        except Exception as e:
            logger.error(f"Failed to load customization configurations: {e}")
            raise e

    def load_user_preferences(self):
        """
        Loads user preferences from the preferences file.
        """
        logger.info("Loading user preferences.")
        try:
            if os.path.exists(self.preferences_file):
                with open(self.preferences_file, 'r') as f:
                    self.user_preferences = json.load(f)
                logger.info("User preferences loaded successfully.")
            else:
                logger.info("Preferences file not found. Using default preferences.")
                self.user_preferences = {}
        except Exception as e:
            logger.error(f"Failed to load user preferences: {e}")
            raise e

    def save_user_preferences(self):
        """
        Saves user preferences to the preferences file.
        """
        logger.info("Saving user preferences.")
        try:
            os.makedirs(os.path.dirname(self.preferences_file), exist_ok=True)
            with open(self.preferences_file, 'w') as f:
                json.dump(self.user_preferences, f, indent=4)
            logger.info("User preferences saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save user preferences: {e}")
            raise e

    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """
        Retrieves the user preferences for a given user ID.

        :param user_id: The user's unique identifier.
        :return: Dictionary of user preferences.
        """
        logger.info(f"Retrieving preferences for user '{user_id}'.")
        try:
            preferences = self.user_preferences.get(user_id, {
                'theme': self.customization_config['default_theme'],
                'layout': self.customization_config['default_layout'],
                'preferences': {}
            })
            return preferences
        except Exception as e:
            logger.error(f"Failed to retrieve preferences for user '{user_id}': {e}")
            raise e

    def set_user_preferences(self, user_id: str, preferences: Dict[str, Any]):
        """
        Sets the user preferences for a given user ID.

        :param user_id: The user's unique identifier.
        :param preferences: Dictionary of preferences to set.
        """
        logger.info(f"Setting preferences for user '{user_id}'.")
        try:
            self.user_preferences[user_id] = preferences
            self.save_user_preferences()
            logger.info(f"Preferences updated for user '{user_id}'.")
        except Exception as e:
            logger.error(f"Failed to set preferences for user '{user_id}': {e}")
            raise e

    def update_user_preferences(self, user_id: str, updates: Dict[str, Any]):
        """
        Updates the user preferences for a given user ID.

        :param user_id: The user's unique identifier.
        :param updates: Dictionary of preference updates.
        """
        logger.info(f"Updating preferences for user '{user_id}'.")
        try:
            preferences = self.get_user_preferences(user_id)
            preferences.update(updates)
            self.set_user_preferences(user_id, preferences)
            logger.info(f"Preferences updated for user '{user_id}'.")
        except Exception as e:
            logger.error(f"Failed to update preferences for user '{user_id}': {e}")
            raise e

    def reset_user_preferences(self, user_id: str):
        """
        Resets the user preferences to default for a given user ID.

        :param user_id: The user's unique identifier.
        """
        logger.info(f"Resetting preferences for user '{user_id}'.")
        try:
            default_preferences = {
                'theme': self.customization_config['default_theme'],
                'layout': self.customization_config['default_layout'],
                'preferences': {}
            }
            self.set_user_preferences(user_id, default_preferences)
            logger.info(f"Preferences reset to default for user '{user_id}'.")
        except Exception as e:
            logger.error(f"Failed to reset preferences for user '{user_id}': {e}")
            raise e

    def send_notification(self, user_id: str, subject: str, message: str):
        """
        Sends a notification to the user about preference changes.

        :param user_id: The user's unique identifier.
        :param subject: Subject of the notification.
        :param message: Body of the notification.
        """
        try:
            # Retrieve user's contact information (e.g., email) from a user management system
            # For this example, we'll assume the user's email is their user_id
            recipient_email = f"{user_id}@example.com"
            self.notification_manager.send_notification(
                recipients=[recipient_email],
                subject=subject,
                message=message
            )
            logger.info(f"Notification sent to user '{user_id}'.")
        except Exception as e:
            logger.error(f"Failed to send notification to user '{user_id}': {e}")

# --------------------- Example Usage --------------------- #

def example_usage():
    """
    Demonstrates example usage of the CustomizationHandler class.
    """
    try:
        # Initialize CustomizationHandler
        handler = CustomizationHandler()

        # User ID for the example
        user_id = 'john_doe'

        # Get current preferences
        current_preferences = handler.get_user_preferences(user_id)
        print(f"Current preferences for '{user_id}': {current_preferences}")

        # Update preferences
        updates = {
            'theme': 'dark',
            'layout': 'compact',
            'preferences': {
                'font_size': 'medium',
                'language': 'en-US'
            }
        }
        handler.update_user_preferences(user_id, updates)
        print(f"Updated preferences for '{user_id}': {handler.get_user_preferences(user_id)}")

        # Reset preferences to default
        handler.reset_user_preferences(user_id)
        print(f"Preferences after reset for '{user_id}': {handler.get_user_preferences(user_id)}")

    except Exception as e:
        logger.exception(f"Error in example usage: {e}")

# --------------------- Main Execution --------------------- #

if __name__ == "__main__":
    # Run the customization handler example
    example_usage()
