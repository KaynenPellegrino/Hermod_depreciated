# hermod/utils/logger.py

"""
Module: logger.py

Provides a logger setup function for the Hermod application.
"""

import logging
import os

import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logger(log_file='logs/hermod.log', log_level_file=logging.DEBUG, log_level_console=logging.INFO):
    """
    Sets up the logger for the application.

    Args:
        log_file (str): The log file path.
        log_level_file (int): The logging level for the file handler.
        log_level_console (int): The logging level for the console handler.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create logs directory if it doesn't exist
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))

    # Create a logger
    logger = logging.getLogger('hermod')
    logger.setLevel(logging.DEBUG)  # Set to the lowest level to capture all messages

    # Avoid adding handlers multiple times
    if not logger.handlers:
        # File handler with log rotation
        file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3)  # 5MB per file, 3 backups
        file_handler.setLevel(log_level_file)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level_console)

        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Add formatter to handlers
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        # Prevent double logging by stopping propagation
        logger.propagate = False

    return logger
#initialize and export logger
logger = setup_logger()


def log_rollback(module_path, reason):
    with open('rollback_log.txt', 'a') as f:
        f.write(f"Rolled back {module_path} due to {reason}\n")
    logger.info(f"Rollback log updated for {module_path}")

    return logger

def log_refactor_results(module_path, complexity_before, complexity_after, performance_metrics, test_passed):
    """
    Logs the results of the refactor process including complexity, performance, and test results.

    Args:
        module_path (str): The path to the module that was refactored.
        complexity_before (dict): Complexity metrics before the refactor.
        complexity_after (dict): Complexity metrics after the refactor.
        performance_metrics (dict): Performance metrics after refactor.
        test_passed (bool): Whether the tests passed after refactor.
    """
    logger.info(f"Refactor results for {module_path}:")
    logger.info(f"Complexity before: {complexity_before}, after: {complexity_after}")
    logger.info(f"Performance metrics: {performance_metrics}")
    logger.info(f"Tests passed: {test_passed}")

def log_performance_profiling(module_path):
    profiler = cProfile.Profile()
    profiler.enable()
    exec(open(module_path).read())
    profiler.disable()
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats()
    with open(f'logs/{module_path}_profile.txt', 'w') as f:
        f.write(s.getvalue())
    logger.info(f"Profiling data saved for {module_path}")
