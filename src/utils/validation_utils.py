# src/utils/validation_utils.py
import json
import re
from typing import Any, Dict, List, Optional, Type, TypeVar
from pydantic import BaseModel, ValidationError, validator, EmailStr, HttpUrl
from validators import url as validate_url
from src.utils.config_loader import ConfigurationManager

# Load configuration
config_manager = ConfigurationManager()

# Type variable for Pydantic models
T = TypeVar('T', bound=BaseModel)

class UserRegistrationModel(BaseModel):
    username: str
    email: EmailStr
    password: str
    confirm_password: str
    phone_number: Optional[str] = None
    website: Optional[HttpUrl] = None

    @validator('password')
    def password_strength(cls, v):
        """
        Validates the strength of the password.
        Password must be at least 8 characters long, contain uppercase, lowercase, digit, and special character.
        """
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one digit')
        if not re.search(r'[^\w\s]', v):
            raise ValueError('Password must contain at least one special character')
        return v

    @validator('confirm_password')
    def passwords_match(cls, v, values, **kwargs):
        """
        Validates that confirm_password matches password.
        """
        if 'password' in values and v != values['password']:
            raise ValueError('Passwords do not match')
        return v

    @validator('phone_number')
    def validate_phone_number(cls, v):
        """
        Validates the phone number format.
        Accepts formats like +1234567890, 123-456-7890, (123) 456-7890, etc.
        """
        if v:
            phone_pattern = re.compile(r'^(\+\d{1,3}[- ]?)?\d{10}$')
            if not phone_pattern.match(v):
                raise ValueError('Invalid phone number format')
        return v

def validate_user_registration(data: Dict[str, Any]) -> UserRegistrationModel:
    """
    Validates user registration data against the UserRegistrationModel.
    Raises ValidationError if validation fails.
    """
    try:
        user = UserRegistrationModel(**data)
        return user
    except ValidationError as e:
        raise e

class ConfigModel(BaseModel):
    database: Dict[str, Any]
    app: Dict[str, Any]
    security: Dict[str, Any]
    email: Dict[str, Any]

    @validator('database')
    def validate_database(cls, v):
        required_fields = ['host', 'port', 'user', 'password', 'name', 'type']
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Database configuration missing '{field}' field")
        return v

    @validator('app')
    def validate_app(cls, v):
        required_fields = ['debug', 'secret_key']
        for field in required_fields:
            if field not in v:
                raise ValueError(f"App configuration missing '{field}' field")
        return v

    @validator('security')
    def validate_security(cls, v):
        required_fields = ['secret_key', 'algorithm', 'access_token_expire_minutes']
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Security configuration missing '{field}' field")
        return v

    @validator('email')
    def validate_email_config(cls, v):
        required_fields = ['host', 'port', 'user', 'password', 'use_tls', 'use_ssl', 'default_from']
        for field in required_fields:
            if field not in v:
                raise ValueError(f"Email configuration missing '{field}' field")
        return v

def validate_config(config: Dict[str, Any]) -> ConfigModel:
    """
    Validates the entire configuration dictionary against the ConfigModel.
    Raises ValidationError if validation fails.
    """
    try:
        validated_config = ConfigModel(**config)
        return validated_config
    except ValidationError as e:
        raise e

def validate_url_custom(url: str) -> bool:
    """
    Validates a URL using the validators library.
    Returns True if valid, False otherwise.
    """
    return validate_url(url)

def validate_email_custom(email: str) -> bool:
    """
    Validates an email address using Pydantic's EmailStr.
    Returns True if valid, False otherwise.
    """
    try:
        EmailStr.validate(email)
        return True
    except ValidationError:
        return False

def validate_phone_number_custom(phone: str) -> bool:
    """
    Validates a phone number format.
    Accepts formats like +1234567890, 123-456-7890, (123) 456-7890, etc.
    """
    phone_pattern = re.compile(r'^(\+\d{1,3}[- ]?)?\d{10}$')
    return bool(phone_pattern.match(phone))

def validate_password_strength(password: str) -> bool:
    """
    Validates the strength of a password.
    Password must be at least 8 characters long, contain uppercase, lowercase, digit, and special character.
    """
    if len(password) < 8:
        return False
    if not re.search(r'[A-Z]', password):
        return False
    if not re.search(r'[a-z]', password):
        return False
    if not re.search(r'\d', password):
        return False
    if not re.search(r'[^\w\s]', password):
        return False
    return True

def validate_json(json_str: str) -> bool:
    """
    Validates whether a string is a valid JSON.
    """
    try:
        json.loads(json_str)
        return True
    except json.JSONDecodeError:
        return False

def validate_yaml(yaml_str: str) -> bool:
    """
    Validates whether a string is a valid YAML.
    """
    try:
        import yaml
        yaml.safe_load(yaml_str)
        return True
    except yaml.YAMLError:
        return False

def validate_integer(value: Any) -> bool:
    """
    Validates whether a value is an integer.
    """
    return isinstance(value, int)

def validate_float(value: Any) -> bool:
    """
    Validates whether a value is a float.
    """
    return isinstance(value, float)

def validate_boolean(value: Any) -> bool:
    """
    Validates whether a value is a boolean.
    """
    return isinstance(value, bool)

def validate_non_empty_string(value: Any) -> bool:
    """
    Validates whether a value is a non-empty string.
    """
    return isinstance(value, str) and bool(value.strip())

def validate_list_of_strings(lst: Any) -> bool:
    """
    Validates whether a value is a list of strings.
    """
    if not isinstance(lst, list):
        return False
    return all(isinstance(item, str) for item in lst)
