# hermod/utils/config.py

"""
Module: config.py

Loads and provides access to application configuration settings.
"""

import os
import yaml

def load_config(config_file='config.yaml'):
    """
    Loads the configuration from a YAML file.

    Args:
        config_file (str): The path to the configuration file.

    Returns:
        dict: The configuration settings.
    """
    if not os.path.exists(config_file):
        default_config = {
            'app_name': 'Hermod',
            'version': '0.1.0',
            'author': 'Kaynen Pellegrino',
            'description': 'Autonomous AI-powered development assistant',
        }
        with open(config_file, 'w', encoding='utf-8') as file:
            yaml.dump(default_config, file)
        return default_config
    else:
        with open(config_file, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config
