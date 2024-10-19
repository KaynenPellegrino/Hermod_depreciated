# src/modules/error_handling/error_logger.py

import logging
import logging.handlers
import os
from typing import Optional, Dict, Any
from datetime import datetime
from src.utils.configuration_manager import ConfigurationManager
from src.modules.notifications.notification_manager import NotificationManager

class ErrorLogger:
    """
    Centralized Error Logging
    Handles logging of all errors and warnings across the system.
    Aggregates logs from various modules, formats them for consistency,
    and stores them for future analysis and debugging.
    Includes mechanisms for setting log levels to filter relevant information.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        """
        Singleton pattern to ensure only one instance of ErrorLogger exists.
        """
        if cls._instance is None:
            cls._instance = super(ErrorLogger, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """
        Initializes the ErrorLogger with necessary configurations.
        """
        if hasattr(self, 'initialized') and self.initialized:
            return

        self.config_manager = ConfigurationManager()
        self.notification_manager = NotificationManager()
        self.load_logging_config()
        self.setup_logger()
        self.initialized = True

    def load_logging_config(self):
        """
        Loads logging configurations from the configuration manager or environment variables.
        """
        self.logging_config = {
            'log_level': self.config_manager.get('LOG_LEVEL', 'INFO').upper(),
            'log_format': self.config_manager.get(
                'LOG_FORMAT',
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ),
            'log_file': self.config_manager.get('LOG_FILE', 'logs/hermod.log'),
            'max_bytes': int(self.config_manager.get('LOG_MAX_BYTES', 5 * 1024 * 1024)),  # 5 MB
            'backup_count': int(self.config_manager.get('LOG_BACKUP_COUNT', 5)),
            'email_notifications': self.config_manager.get('EMAIL_NOTIFICATIONS', False),
            'alert_recipients': self.config_manager.get('ALERT_RECIPIENTS', '').split(','),
            'critical_log_level': self.config_manager.get('CRITICAL_LOG_LEVEL', 'ERROR').upper(),
        }

    def setup_logger(self):
        """
        Sets up the root logger with RotatingFileHandler and appropriate formatting.
        """
        self.logger = logging.getLogger()
        self.logger.setLevel(getattr(logging, self.logging_config['log_level'], logging.INFO))

        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(self.logging_config['log_file'])
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            self.logging_config['log_file'],
            maxBytes=self.logging_config['max_bytes'],
            backupCount=self.logging_config['backup_count']
        )
        file_formatter = logging.Formatter(self.logging_config['log_format'])
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(self.logging_config['log_format'])
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # Set up email notifications for critical logs
        if self.logging_config['email_notifications']:
            self.setup_email_notifications()

    def setup_email_notifications(self):
        """
        Sets up email notifications for logs at or above a certain level.
        """
        log_level = getattr(logging, self.logging_config['critical_log_level'], logging.ERROR)
        email_handler = EmailNotificationHandler(
            notification_manager=self.notification_manager,
            recipients=self.logging_config['alert_recipients'],
            level=log_level
        )
        email_formatter = logging.Formatter(self.logging_config['log_format'])
        email_handler.setFormatter(email_formatter)
        self.logger.addHandler(email_handler)

    def get_logger(self, name: Optional[str] = None) -> logging.Logger:
        """
        Returns a logger with the specified name.

        :param name: Name of the logger.
        :return: Logger instance.
        """
        return logging.getLogger(name)

    def set_log_level(self, level: str):
        """
        Sets the log level for the root logger.

        :param level: Log level as a string (e.g., 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
        """
        self.logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        for handler in self.logger.handlers:
            handler.setLevel(getattr(logging, level.upper(), logging.INFO))

class EmailNotificationHandler(logging.Handler):
    """
    Custom logging handler to send email notifications for critical logs.
    """

    def __init__(self, notification_manager: NotificationManager, recipients: list, level=logging.ERROR):
        super().__init__(level)
        self.notification_manager = notification_manager
        self.recipients = recipients

    def emit(self, record):
        """
        Sends an email notification when a log record is emitted.

        :param record: Log record.
        """
        try:
            log_entry = self.format(record)
            subject = f"Critical Alert: {record.levelname}"
            message = f"A critical error occurred:\n\n{log_entry}"
            self.notification_manager.send_notification(
                recipients=self.recipients,
                subject=subject,
                message=message
            )
        except Exception as e:
            print(f"Failed to send email notification: {e}")

# --------------------- Example Usage --------------------- #

def example_usage():
    """
    Demonstrates example usage of the ErrorLogger class.
    """
    try:
        # Initialize ErrorLogger
        error_logger = ErrorLogger()
        logger = error_logger.get_logger(__name__)

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

        # Change log level
        error_logger.set_log_level('DEBUG')
        logger.debug("This debug message should now appear.")

    except Exception as e:
        logger.exception(f"Error in example usage: {e}")

# --------------------- Main Execution --------------------- #

if __name__ == "__main__":
    # Run the error logger example
    example_usage()
