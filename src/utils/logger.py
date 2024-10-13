# src/modules/utils/logger.py
import logging
from logging.handlers import RotatingFileHandler
import os


def get_logger(name: str, log_file: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Prevent adding multiple handlers if the logger already has handlers
    if not logger.handlers:
        os.makedirs('logs', exist_ok=True)
        handler = RotatingFileHandler(log_file, maxBytes=10 ** 6, backupCount=5)
        formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
