# src/modules/error_handling/error_alerts.py

import logging
import os
from typing import Optional, List
from src.utils.configuration_manager import ConfigurationManager
from src.modules.notifications.notification_manager import NotificationManager

class ErrorAlerts:
    """
    Error Notification System
    Integrates with the existing notification system to send alerts when critical errors or issues occur.
    Responsible for detecting significant issues and sending real-time notifications to users or administrators,
    enabling swift resolution.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        """
        Singleton pattern to ensure only one instance of ErrorAlerts exists.
        """
        if cls._instance is None:
            cls._instance = super(ErrorAlerts, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """
        Initializes the ErrorAlerts with necessary configurations.
        """
        if hasattr(self, 'initialized') and self.initialized:
            return

        self.config_manager = ConfigurationManager()
        self.notification_manager = NotificationManager()
        self.load_alert_config()
        self.setup_alert_handler()
        self.initialized = True

    def load_alert_config(self):
        """
        Loads alert configurations from the configuration manager or environment variables.
        """
        self.alert_config = {
            'alert_levels': self.config_manager.get('ALERT_LEVELS', ['ERROR', 'CRITICAL']),
            'alert_recipients': self.config_manager.get('ALERT_RECIPIENTS', '').split(','),
            'enable_alerts': self.config_manager.get('ENABLE_ALERTS', True),
            'log_format': self.config_manager.get(
                'LOG_FORMAT',
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ),
        }

    def setup_alert_handler(self):
        """
        Sets up the alert handler to send notifications when critical errors occur.
        """
        if not self.alert_config['enable_alerts']:
            return

        self.logger = logging.getLogger()
        alert_levels = [getattr(logging, level.upper(), logging.ERROR) for level in self.alert_config['alert_levels']]
        min_alert_level = min(alert_levels)

        alert_handler = AlertNotificationHandler(
            notification_manager=self.notification_manager,
            recipients=self.alert_config['alert_recipients'],
            level=min_alert_level
        )
        alert_formatter = logging.Formatter(self.alert_config['log_format'])
        alert_handler.setFormatter(alert_formatter)
        self.logger.addHandler(alert_handler)

    def add_custom_alert_handler(self, handler: logging.Handler):
        """
        Adds a custom alert handler to the logger.

        :param handler: A logging.Handler instance.
        """
        self.logger.addHandler(handler)

    def get_logger(self, name: Optional[str] = None) -> logging.Logger:
        """
        Returns a logger with the specified name.

        :param name: Name of the logger.
        :return: Logger instance.
        """
        return logging.getLogger(name)

class AlertNotificationHandler(logging.Handler):
    """
    Custom logging handler to send notifications for critical logs.
    """

    def __init__(self, notification_manager: NotificationManager, recipients: List[str], level=logging.ERROR):
        super().__init__(level)
        self.notification_manager = notification_manager
        self.recipients = recipients

    def emit(self, record):
        """
        Sends a notification when a log record is emitted.

        :param record: Log record.
        """
        try:
            log_entry = self.format(record)
            subject = f"Alert: {record.levelname} in {record.name}"
            message = f"A critical issue occurred:\n\n{log_entry}"
            self.notification_manager.send_notification(
                recipients=self.recipients,
                subject=subject,
                message=message
            )
        except Exception as e:
            print(f"Failed to send alert notification: {e}")

# --------------------- Example Usage --------------------- #

def example_usage():
    """
    Demonstrates example usage of the ErrorAlerts class.
    """
    try:
        # Initialize ErrorAlerts
        error_alerts = ErrorAlerts()
        logger = error_alerts.get_logger(__name__)

        # Log messages at different levels
        logger.debug("This is a debug message.")
        logger.info("This is an info message.")
        logger.warning("This is a warning message.")
        logger.error("This is an error message.")
        logger.critical("This is a critical message.")

        # Simulate an exception
        try:
            1 / 0
        except ZeroDivisionError as e:
            logger.exception("An exception occurred: %s", e)

    except Exception as e:
        logger.exception(f"Error in example usage: {e}")

# --------------------- Main Execution --------------------- #

if __name__ == "__main__":
    # Run the error alerts example
    example_usage()
