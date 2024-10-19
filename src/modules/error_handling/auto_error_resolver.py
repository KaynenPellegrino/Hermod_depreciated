# src/modules/error_handling/auto_error_resolver.py

import logging
import time
from typing import Callable, Any, Optional, Dict, List
from functools import wraps
from src.utils.configuration_manager import ConfigurationManager
from src.modules.notifications.notification_manager import NotificationManager

# Initialize logger
logger = logging.getLogger(__name__)

class AutoErrorResolver:
    """
    Automatic Error Handling
    Includes logic to handle predictable errors automatically, such as retrying failed processes,
    switching to fallback options, or recovering from common issues.
    Integrated with various system components to enhance robustness and resilience to failures.
    """

    def __init__(self):
        """
        Initializes the AutoErrorResolver with necessary configurations.
        """
        self.config_manager = ConfigurationManager()
        self.notification_manager = NotificationManager()
        self.load_resolver_config()
        logger.info("AutoErrorResolver initialized successfully.")

    def load_resolver_config(self):
        """
        Loads resolver configurations from the configuration manager or environment variables.
        """
        self.resolver_config = {
            'max_retries': int(self.config_manager.get('MAX_RETRIES', 3)),
            'retry_delay': float(self.config_manager.get('RETRY_DELAY', 2.0)),  # seconds
            'fallback_functions': {},  # Function name mapping to their fallback functions
            'alert_recipients': self.config_manager.get('ALERT_RECIPIENTS', '').split(','),
            'enable_notifications': self.config_manager.get('ENABLE_ERROR_NOTIFICATIONS', False),
        }

    def retry(self, max_retries: Optional[int] = None, retry_delay: Optional[float] = None):
        """
        Decorator to retry a function if it raises an exception.

        :param max_retries: Maximum number of retries.
        :param retry_delay: Delay between retries in seconds.
        :return: Decorated function.
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                retries = max_retries if max_retries is not None else self.resolver_config['max_retries']
                delay = retry_delay if retry_delay is not None else self.resolver_config['retry_delay']
                last_exception = None

                for attempt in range(1, retries + 1):
                    try:
                        logger.debug(f"Attempt {attempt} for function '{func.__name__}'")
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        logger.warning(f"Error in '{func.__name__}' on attempt {attempt}: {e}")
                        time.sleep(delay)

                logger.error(f"Function '{func.__name__}' failed after {retries} retries.")
                self.handle_failure(func, last_exception)
                raise last_exception

            return wrapper
        return decorator

    def fallback(self, fallback_func: Callable):
        """
        Decorator to switch to a fallback function if the original function fails.

        :param fallback_func: The fallback function to call upon failure.
        :return: Decorated function.
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    logger.debug(f"Executing function '{func.__name__}'")
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Error in '{func.__name__}': {e}. Switching to fallback '{fallback_func.__name__}'")
                    return fallback_func(*args, **kwargs)
            return wrapper
        return decorator

    def handle_failure(self, func: Callable, exception: Exception):
        """
        Handles a function failure by sending notifications if enabled.

        :param func: The function that failed.
        :param exception: The exception that was raised.
        """
        logger.debug(f"Handling failure for function '{func.__name__}'")
        if self.resolver_config['enable_notifications']:
            subject = f"Error in function '{func.__name__}'"
            message = f"An error occurred in function '{func.__name__}':\n\n{exception}"
            self.notification_manager.send_notification(
                recipients=self.resolver_config['alert_recipients'],
                subject=subject,
                message=message
            )

    def recover(self, recovery_func: Callable):
        """
        Decorator to attempt recovery using a recovery function upon failure.

        :param recovery_func: The recovery function to call upon failure.
        :return: Decorated function.
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    logger.debug(f"Executing function '{func.__name__}'")
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Error in '{func.__name__}': {e}. Attempting recovery with '{recovery_func.__name__}'")
                    recovery_successful = recovery_func(*args, **kwargs)
                    if recovery_successful:
                        logger.info(f"Recovery successful for function '{func.__name__}'. Retrying...")
                        return func(*args, **kwargs)
                    else:
                        logger.error(f"Recovery failed for function '{func.__name__}'")
                        self.handle_failure(func, e)
                        raise e
            return wrapper
        return decorator

    # --------------------- Example Usage --------------------- #

    def example_usage(self):
        """
        Demonstrates example usage of the AutoErrorResolver class.
        """
        # Initialize AutoErrorResolver
        resolver = AutoErrorResolver()

        # Define a function that may fail
        @resolver.retry(max_retries=3, retry_delay=1)
        def unreliable_function(x):
            logger.info(f"Attempting to process x={x}")
            if x < 0:
                raise ValueError("Negative value not allowed")
            return x * 2

        # Define a fallback function
        def fallback_function(x):
            logger.info(f"Fallback function called with x={x}")
            return abs(x) * 2

        # Define a function with fallback
        @resolver.fallback(fallback_func=fallback_function)
        def function_with_fallback(x):
            logger.info(f"Processing x={x} in function_with_fallback")
            if x < 0:
                raise ValueError("Negative value not allowed")
            return x * 3

        # Define a recovery function
        def recovery_function(x):
            logger.info(f"Attempting recovery for x={x}")
            # Perform recovery steps here
            return True  # Return True if recovery was successful

        # Define a function with recovery
        @resolver.recover(recovery_func=recovery_function)
        def function_with_recovery(x):
            logger.info(f"Processing x={x} in function_with_recovery")
            if x < 0:
                raise ValueError("Negative value not allowed")
            return x * 4

        # Test the functions
        try:
            result = unreliable_function(-1)
            logger.info(f"Result from unreliable_function: {result}")
        except Exception as e:
            logger.error(f"unreliable_function failed: {e}")

        result = function_with_fallback(-2)
        logger.info(f"Result from function_with_fallback: {result}")

        result = function_with_recovery(-3)
        logger.info(f"Result from function_with_recovery: {result}")

    # --------------------- Main Execution --------------------- #

    if __name__ == "__main__":
        # Run the auto error resolver example
        example_usage()
