# src/utils/config_loader.py

import os
import json
import yaml
import configparser
from typing import Any, Dict, Optional, Union
from threading import Lock

class ConfigurationManager:
    """
    Loads configuration files from various formats (e.g., YAML, JSON, INI),
    parsing them into usable data structures within the application.
    Implements the Singleton pattern to ensure only one instance exists.
    """

    _instance = None
    _lock = Lock()

    def __new__(cls, config_file: Optional[str] = None):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ConfigurationManager, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_file: Optional[str] = None):
        if self._initialized:
            return
        self._config: Dict[str, Any] = {}
        self._load_config(config_file)
        self._initialized = True

    def _load_config(self, config_file: Optional[str] = None):
        """
        Loads the configuration from the specified file or the default locations.
        """
        if config_file is None:
            config_file = self._find_config_file()

        if not config_file or not os.path.exists(config_file):
            raise FileNotFoundError("Configuration file not found.")

        _, ext = os.path.splitext(config_file)
        ext = ext.lower()

        if ext in ['.yaml', '.yml']:
            self._load_yaml_config(config_file)
        elif ext == '.json':
            self._load_json_config(config_file)
        elif ext in ['.ini', '.cfg']:
            self._load_ini_config(config_file)
        else:
            raise ValueError(f"Unsupported configuration file format: {ext}")

        # Override with environment variables
        self._override_with_env_variables()

    def _find_config_file(self) -> Optional[str]:
        """
        Attempts to find a configuration file in the default locations.
        """
        default_files = [
            'config/config.yaml',
            'config/config.yml',
            'config/config.json',
            'config/config.ini',
        ]
        for file in default_files:
            if os.path.exists(file):
                return file
        return None

    def _load_yaml_config(self, config_file: str):
        """
        Loads configuration from a YAML file.
        """
        with open(config_file, 'r') as f:
            self._config = yaml.safe_load(f) or {}

    def _load_json_config(self, config_file: str):
        """
        Loads configuration from a JSON file.
        """
        with open(config_file, 'r') as f:
            self._config = json.load(f)

    def _load_ini_config(self, config_file: str):
        """
        Loads configuration from an INI file.
        """
        parser = configparser.ConfigParser()
        parser.read(config_file)
        self._config = self._configparser_to_dict(parser)

    def _configparser_to_dict(self, parser: configparser.ConfigParser) -> Dict[str, Any]:
        """
        Converts a ConfigParser object to a nested dictionary.
        """
        config_dict = {}
        for section in parser.sections():
            config_dict[section] = {}
            for key, value in parser.items(section):
                config_dict[section][key] = self._cast_value(value)
        return config_dict

    def _cast_value(self, value: str) -> Any:
        """
        Attempts to cast a string value to int, float, bool, or leaves it as a string.
        """
        for cast in (self._to_bool, int, float):
            try:
                return cast(value)
            except ValueError:
                continue
        return value

    def _to_bool(self, value: str) -> bool:
        """
        Converts a string to a boolean if possible.
        """
        if value.lower() in ['true', 'yes', '1']:
            return True
        if value.lower() in ['false', 'no', '0']:
            return False
        raise ValueError()

    def _override_with_env_variables(self):
        """
        Overrides configuration values with environment variables if they are set.
        Environment variables should be in uppercase and use underscores as separators.
        """
        for key in self._flatten_dict(self._config):
            env_var = key.upper()
            if env_var in os.environ:
                self._set_by_flat_key(key, os.environ[env_var])

    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """
        Flattens a nested dictionary into a flat dictionary with keys representing the path.
        """
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(self._flatten_dict(v, new_key, sep=sep))
            else:
                items[new_key] = v
        return items

    def _set_by_flat_key(self, flat_key: str, value: Any, sep: str = '_'):
        """
        Sets a value in the nested configuration dictionary using a flat key.
        """
        keys = flat_key.lower().split(sep)
        d = self._config
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = self._cast_value(value)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieves a configuration value using a dot-separated key.
        """
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def set(self, key: str, value: Any):
        """
        Sets a configuration value using a dot-separated key.
        """
        keys = key.split('.')
        d = self._config
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value

    def as_dict(self) -> Dict[str, Any]:
        """
        Returns the entire configuration as a dictionary.
        """
        return self._config
