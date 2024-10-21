# src/utils/utils.py

import os
import json
import logging
import threading
import functools
from datetime import datetime, time
from typing import Any, Dict, Optional
import validators
from dotenv import load_dotenv

# Load environment variables from a .env file (if it exists)
load_dotenv()

def setup_logging(
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    format_string: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
):
    """
    Sets up logging for the application.
    """
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=handlers
    )

def singleton(cls):
    """
    Decorator to make a class a Singleton.
    """
    instances = {}
    lock = threading.Lock()

    @functools.wraps(cls)
    def get_instance(*args, **kwargs):
        with lock:
            if cls not in instances:
                instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

def is_valid_email(email: str) -> bool:
    """
    Validates an email address.
    """
    return validators.email(email)

def is_valid_url(url: str) -> bool:
    """
    Validates a URL.
    """
    return validators.url(url)

def dict_merge(a: dict, b: dict) -> dict:
    """
    Recursively merges dictionary b into dictionary a.
    """
    for key in b:
        if key in a and isinstance(a[key], dict) and isinstance(b[key], dict):
            dict_merge(a[key], b[key])
        else:
            a[key] = b[key]
    return a

def json_dumps(obj: Any, indent: Optional[int] = 2) -> str:
    """
    Serializes an object to a JSON-formatted string.
    """
    return json.dumps(obj, indent=indent, default=str)

def get_env_variable(key: str, default: Optional[Any] = None) -> Any:
    """
    Retrieves an environment variable, returning a default value if not found.
    """
    return os.getenv(key, default)

def to_camel_case(snake_str: str) -> str:
    """
    Converts snake_case string to camelCase.
    """
    components = snake_str.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])

def to_snake_case(camel_str: str) -> str:
    """
    Converts CamelCase string to snake_case.
    """
    import re
    snake_str = re.sub('([A-Z])', r'_\1', camel_str).lower()
    return snake_str.lstrip('_')

def get_current_datetime_iso() -> str:
    """
    Returns the current UTC datetime in ISO 8601 format.
    """
    return datetime.utcnow().isoformat() + 'Z'

def chunk_list(lst: list, chunk_size: int) -> list:
    """
    Splits a list into smaller lists (chunks) of a specified size.
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def retry(
    exception_to_check: Exception,
    tries: int = 3,
    delay: int = 1,
    backoff: int = 2
):
    """
    Decorator for retrying a function call with exponential backoff.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return func(*args, **kwargs)
                except exception_to_check as e:
                    logging.warning(f"{e}, Retrying in {mdelay} seconds...")
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return func(*args, **kwargs)
        return wrapper
    return decorator

def flatten_list(nested_list: list) -> list:
    """
    Flattens a nested list into a single list.
    """
    result = []
    for item in nested_list:
        if isinstance(item, list):
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result

def deep_get(dictionary: Dict, keys: list, default: Any = None) -> Any:
    """
    Safely gets a value from a nested dictionary using a list of keys.
    """
    for key in keys:
        if isinstance(dictionary, dict):
            dictionary = dictionary.get(key, default)
        else:
            return default
    return dictionary

def sizeof_fmt(num: float, suffix: str = 'B') -> str:
    """
    Converts a file size to human-readable form.
    """
    for unit in ['','K','M','G','T','P','E','Z']:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Y{suffix}"
