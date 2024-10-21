# src/utils/security.py

from datetime import datetime, timedelta
from typing import Optional, Union
from passlib.context import CryptContext
import jwt
from jwt import PyJWTError
from src.utils.config_loader import ConfigurationManager
from src.app.models import User
from src.utils.database import SessionLocal
from sqlalchemy.orm import Session
import logging

# Configure the logger for this module
logger = logging.getLogger(__name__)

# Load configuration
config_manager = ConfigurationManager()
SECRET_KEY = config_manager.get('security.secret_key', 'your-secret-key')
ALGORITHM = config_manager.get('security.algorithm', 'HS256')
ACCESS_TOKEN_EXPIRE_MINUTES = config_manager.get('security.access_token_expire_minutes', 30)

# Password context for hashing and verification
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verifies a plain-text password against a hashed password.
    """
    return pwd_context.verify(plain_password, hashed_password)

def hash_password(password: str) -> str:
    """
    Hashes a plain-text password using bcrypt.
    """
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Creates a JWT access token.
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def decode_access_token(token: str) -> Optional[dict]:
    """
    Decodes a JWT access token.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except PyJWTError as e:
        logger.exception("Failed to decode JWT token")
        return None

async def authenticate_user(username: str, password: str) -> Optional[User]:
    """
    Authenticates a user by username and password.
    """
    db: Session = SessionLocal()
    try:
        user = db.query(User).filter(User.username == username).first()
        if user and verify_password(password, user.hashed_password):
            return user
        return None
    finally:
        db.close()

async def verify_token(token: str) -> Optional[User]:
    """
    Verifies a JWT token and returns the associated user.
    """
    payload = decode_access_token(token)
    if payload is None:
        return None
    username: str = payload.get("sub")
    if username is None:
        return None
    db: Session = SessionLocal()
    try:
        user = db.query(User).filter(User.username == username).first()
        return user
    finally:
        db.close()
