# src/modules/security_management/authentication_manager.py

import logging
import os
import bcrypt
import jwt
import pyotp
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean
from sqlalchemy.orm import sessionmaker, declarative_base

# Import MetadataStorage from data_management module
from src.modules.data_management.metadata_storage import MetadataStorage

# Import NotificationManager from notifications module
from src.modules.notifications.notification_manager import NotificationManager

# Import AdvancedSecurityManager from advanced_security module
# Assuming advanced_security is another module that handles behavioral authentication
from src.modules.advanced_security.behavioral_authentication import AdvancedSecurityManager

# Load environment variables
load_dotenv()

# Configure logging with RotatingFileHandler to prevent log files from becoming too large
from logging.handlers import RotatingFileHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = RotatingFileHandler('logs/authentication_manager.log', maxBytes=5*1024*1024, backupCount=5)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# SQLAlchemy setup
Base = declarative_base()


class User(Base):
    """
    Represents a user in the system.
    """
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    username = Column(String(150), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(50), default='user')  # roles: user, admin, etc.
    mfa_enabled = Column(Boolean, default=False)
    mfa_secret = Column(String(16))  # 16-character base32 secret for TOTP
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AuthenticationManager:
    """
    Manages user authentication, multi-factor authentication (MFA), token generation, and access control.
    Integrates behavioral authentication mechanisms for enhanced security.
    """

    def __init__(self):
        """
        Initializes the AuthenticationManager with necessary configurations.
        """
        # Initialize Metadata Storage
        self.metadata_storage = MetadataStorage()

        # Initialize Notification Manager
        self.notification_manager = NotificationManager()

        # Initialize Advanced Security Manager for behavioral authentication
        self.advanced_security_manager = AdvancedSecurityManager()

        # Database configuration
        self.db_url = os.getenv('AUTH_DB_URL', 'sqlite:///auth.db')  # Example using SQLite
        self.engine = create_engine(self.db_url, echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

        # JWT configuration
        self.jwt_secret = os.getenv('JWT_SECRET_KEY', 'your_jwt_secret_key')
        self.jwt_algorithm = os.getenv('JWT_ALGORITHM', 'HS256')
        self.jwt_exp_delta_seconds = int(os.getenv('JWT_EXP_DELTA_SECONDS', '3600'))  # 1 hour

        logger.info("AuthenticationManager initialized successfully.")

    def register_user(self, username: str, email: str, password: str, role: str = 'user') -> bool:
        """
        Registers a new user with the provided credentials.

        :param username: Desired username.
        :param email: User's email address.
        :param password: User's password.
        :param role: User role (default: 'user').
        :return: True if registration is successful, False otherwise.
        """
        session = self.Session()
        try:
            if session.query(User).filter((User.username == username) | (User.email == email)).first():
                logger.warning(f"Registration failed: Username '{username}' or email '{email}' already exists.")
                return False

            # Hash the password
            salt = bcrypt.gensalt()
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)

            # Generate MFA secret
            mfa_secret = pyotp.random_base32()

            # Create new user
            new_user = User(
                username=username,
                email=email,
                password_hash=hashed_password.decode('utf-8'),
                role=role,
                mfa_secret=mfa_secret  # Store MFA secret
            )
            session.add(new_user)
            session.commit()

            logger.info(f"User '{username}' registered successfully.")

            # Log the registration event
            self.metadata_storage.save_metadata({
                'event': 'user_registration',
                'username': username,
                'email': email,
                'role': role,
                'timestamp': datetime.utcnow().isoformat()
            }, storage_type='auth_event')

            # Send MFA setup instructions
            mfa_uri = pyotp.totp.TOTP(mfa_secret).provisioning_uri(name=email, issuer_name="Hermod")
            mfa_qr_code_link = f"https://api.qrserver.com/v1/create-qr-code/?data={mfa_uri}&size=200x200"
            subject = "MFA Setup Instructions for Hermod"
            message = f"""Hello {username},

Thank you for registering with Hermod. To enhance the security of your account, we have enabled Multi-Factor Authentication (MFA).

Please follow these steps to set up MFA:

1. Download an authenticator app like Google Authenticator or Authy on your mobile device.
2. Scan the QR code below using the app:
   {mfa_qr_code_link}
3. Enter the generated code in the Hermod application when prompted.

If you encounter any issues, please contact our support team.

Best regards,
Hermod Security Team
"""
            self.notification_manager.notify(channel='email', subject=subject, message=message, recipients=[email])

            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error during user registration: {e}")
            return False
        finally:
            session.close()

    def authenticate_user(self, username: str, password: str, mfa_code: Optional[str] = None) -> Optional[str]:
        """
        Authenticates a user with username, password, and optional MFA code. Returns JWT token if successful.

        :param username: User's username.
        :param password: User's password.
        :param mfa_code: MFA code from authenticator app.
        :return: JWT token if authentication is successful, None otherwise.
        """
        session = self.Session()
        try:
            user = session.query(User).filter(User.username == username).first()
            if not user:
                logger.warning(f"Authentication failed: User '{username}' not found.")
                return None

            # Verify password
            if not bcrypt.checkpw(password.encode('utf-8'), user.password_hash.encode('utf-8')):
                logger.warning(f"Authentication failed: Incorrect password for user '{username}'.")
                return None

            # Check if MFA is enabled
            if user.mfa_enabled:
                if not mfa_code:
                    logger.warning(f"Authentication failed: MFA code required for user '{username}'.")
                    return None
                totp = pyotp.TOTP(user.mfa_secret)
                if not totp.verify(mfa_code, valid_window=1):
                    logger.warning(f"Authentication failed: Invalid MFA code for user '{username}'.")
                    return None

            # Integrate behavioral authentication
            behavioral_score = self.advanced_security_manager.assess_behavior(username)
            if behavioral_score < self.advanced_security_manager.threshold:
                logger.warning(f"Authentication failed: Behavioral authentication score {behavioral_score} below threshold for user '{username}'.")
                self.notification_manager.notify(
                    channel='email',
                    subject="Unusual Login Attempt Detected",
                    message=f"Hello {username},\n\nAn unusual login attempt was detected from your account. If this wasn't you, please secure your account immediately.",
                    recipients=[user.email]
                )
                return None

            # Generate JWT token
            payload = {
                'user_id': user.id,
                'username': user.username,
                'role': user.role,
                'exp': datetime.utcnow() + timedelta(seconds=self.jwt_exp_delta_seconds)
            }
            token = jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)

            logger.info(f"User '{username}' authenticated successfully.")

            # Log the authentication event
            self.metadata_storage.save_metadata({
                'event': 'user_authentication',
                'username': username,
                'user_id': user.id,
                'behavioral_score': behavioral_score,
                'timestamp': datetime.utcnow().isoformat()
            }, storage_type='auth_event')

            return token
        except Exception as e:
            logger.error(f"Error during user authentication: {e}")
            return None
        finally:
            session.close()

    def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Validates a JWT token and returns the payload if valid.

        :param token: JWT token to validate.
        :return: Payload dictionary if token is valid, None otherwise.
        """
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            logger.info(f"Token validated successfully for user '{payload.get('username')}'.")
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token validation failed: Token has expired.")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Token validation failed: Invalid token.")
            return None

    def enable_mfa(self, username: str) -> bool:
        """
        Enables MFA for a user by generating a new MFA secret.

        :param username: Username of the user.
        :return: True if MFA is enabled successfully, False otherwise.
        """
        session = self.Session()
        try:
            user = session.query(User).filter(User.username == username).first()
            if not user:
                logger.warning(f"MFA enable failed: User '{username}' not found.")
                return False

            # Generate new MFA secret
            mfa_secret = pyotp.random_base32()
            user.mfa_secret = mfa_secret
            user.mfa_enabled = True
            session.commit()

            logger.info(f"MFA enabled for user '{username}'.")

            # Log the MFA enable event
            self.metadata_storage.save_metadata({
                'event': 'mfa_enabled',
                'username': username,
                'timestamp': datetime.utcnow().isoformat()
            }, storage_type='auth_event')

            # Send MFA setup instructions
            mfa_uri = pyotp.totp.TOTP(mfa_secret).provisioning_uri(name=user.email, issuer_name="Hermod")
            mfa_qr_code_link = f"https://api.qrserver.com/v1/create-qr-code/?data={mfa_uri}&size=200x200"
            subject = "MFA Setup Instructions for Hermod"
            message = f"""Hello {username},

We have enabled Multi-Factor Authentication (MFA) for your Hermod account.

Please follow these steps to set up MFA:

1. Download an authenticator app like Google Authenticator or Authy on your mobile device.
2. Scan the QR code below using the app:
   {mfa_qr_code_link}
3. Enter the generated code in the Hermod application when prompted.

If you encounter any issues, please contact our support team.

Best regards,
Hermod Security Team
"""
            self.notification_manager.notify(channel='email', subject=subject, message=message, recipients=[user.email])

            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error enabling MFA for user '{username}': {e}")
            return False
        finally:
            session.close()

    def disable_mfa(self, username: str) -> bool:
        """
        Disables MFA for a user.

        :param username: Username of the user.
        :return: True if MFA is disabled successfully, False otherwise.
        """
        session = self.Session()
        try:
            user = session.query(User).filter(User.username == username).first()
            if not user:
                logger.warning(f"MFA disable failed: User '{username}' not found.")
                return False

            user.mfa_enabled = False
            user.mfa_secret = None
            session.commit()

            logger.info(f"MFA disabled for user '{username}'.")

            # Log the MFA disable event
            self.metadata_storage.save_metadata({
                'event': 'mfa_disabled',
                'username': username,
                'timestamp': datetime.utcnow().isoformat()
            }, storage_type='auth_event')

            # Notify the user about MFA disable
            subject = "MFA Disabled for Your Hermod Account"
            message = f"""Hello {username},

Multi-Factor Authentication (MFA) has been disabled for your Hermod account. If you did not request this change, please contact our support team immediately.

Best regards,
Hermod Security Team
"""
            self.notification_manager.notify(channel='email', subject=subject, message=message, recipients=[user.email])

            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error disabling MFA for user '{username}': {e}")
            return False
        finally:
            session.close()

    def assign_role(self, username: str, new_role: str) -> bool:
        """
        Assigns a new role to a user.

        :param username: Username of the user.
        :param new_role: New role to assign.
        :return: True if role assignment is successful, False otherwise.
        """
        session = self.Session()
        try:
            user = session.query(User).filter(User.username == username).first()
            if not user:
                logger.warning(f"Role assignment failed: User '{username}' not found.")
                return False

            old_role = user.role
            user.role = new_role
            session.commit()

            logger.info(f"User '{username}' role changed from '{old_role}' to '{new_role}'.")

            # Log the role assignment event
            self.metadata_storage.save_metadata({
                'event': 'role_assignment',
                'username': username,
                'old_role': old_role,
                'new_role': new_role,
                'timestamp': datetime.utcnow().isoformat()
            }, storage_type='auth_event')

            # Optionally, notify the user about role change
            subject = "Your Role Has Been Updated"
            message = f"""Hello {username},

Your role has been updated from '{old_role}' to '{new_role}'.

If you did not request this change, please contact our support team immediately.

Best regards,
Hermod Security Team
"""
            self.notification_manager.notify(channel='email', subject=subject, message=message, recipients=[user.email])

            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error during role assignment: {e}")
            return False
        finally:
            session.close()

    def change_password(self, username: str, old_password: str, new_password: str) -> bool:
        """
        Changes a user's password after verifying the old password.

        :param username: Username of the user.
        :param old_password: Current password.
        :param new_password: New password to set.
        :return: True if password change is successful, False otherwise.
        """
        session = self.Session()
        try:
            user = session.query(User).filter(User.username == username).first()
            if not user:
                logger.warning(f"Password change failed: User '{username}' not found.")
                return False

            # Verify old password
            if not bcrypt.checkpw(old_password.encode('utf-8'), user.password_hash.encode('utf-8')):
                logger.warning(f"Password change failed: Incorrect old password for user '{username}'.")
                return False

            # Hash new password
            salt = bcrypt.gensalt()
            hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), salt)
            user.password_hash = hashed_password.decode('utf-8')
            session.commit()

            logger.info(f"Password changed successfully for user '{username}'.")

            # Log the password change event
            self.metadata_storage.save_metadata({
                'event': 'password_change',
                'username': username,
                'timestamp': datetime.utcnow().isoformat()
            }, storage_type='auth_event')

            # Optionally, notify the user about password change
            subject = "Your Password Has Been Changed"
            message = f"""Hello {username},

Your password has been successfully changed. If you did not initiate this change, please contact our support team immediately.

Best regards,
Hermod Security Team
"""
            self.notification_manager.notify(channel='email', subject=subject, message=message, recipients=[user.email])

            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error during password change: {e}")
            return False
        finally:
            session.close()

    def get_user_details(self, username: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves user details excluding sensitive information like password hashes.

        :param username: Username of the user.
        :return: Dictionary containing user details if found, None otherwise.
        """
        session = self.Session()
        try:
            user = session.query(User).filter(User.username == username).first()
            if not user:
                logger.warning(f"User details retrieval failed: User '{username}' not found.")
                return None

            user_details = {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'role': user.role,
                'mfa_enabled': user.mfa_enabled,
                'created_at': user.created_at.isoformat(),
                'updated_at': user.updated_at.isoformat()
            }

            logger.info(f"User details retrieved for '{username}'.")

            return user_details
        except Exception as e:
            logger.error(f"Error retrieving user details for '{username}': {e}")
            return None
        finally:
            session.close()

    def delete_user(self, username: str) -> bool:
        """
        Deletes a user from the system.

        :param username: Username of the user to delete.
        :return: True if deletion is successful, False otherwise.
        """
        session = self.Session()
        try:
            user = session.query(User).filter(User.username == username).first()
            if not user:
                logger.warning(f"User deletion failed: User '{username}' not found.")
                return False

            session.delete(user)
            session.commit()

            logger.info(f"User '{username}' deleted successfully.")

            # Log the deletion event
            self.metadata_storage.save_metadata({
                'event': 'user_deletion',
                'username': username,
                'timestamp': datetime.utcnow().isoformat()
            }, storage_type='auth_event')

            # Optionally, notify the user about account deletion
            subject = "Your Account Has Been Deleted"
            message = f"""Hello {username},

Your account has been successfully deleted from our system. If you did not request this deletion, please contact our support team immediately.

Best regards,
Hermod Security Team
"""
            self.notification_manager.notify(channel='email', subject=subject, message=message, recipients=[user.email])

            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error deleting user '{username}': {e}")
            return False
        finally:
            session.close()

    def disable_mfa(self, username: str) -> bool:
        """
        Disables MFA for a user.

        :param username: Username of the user.
        :return: True if MFA is disabled successfully, False otherwise.
        """
        session = self.Session()
        try:
            user = session.query(User).filter(User.username == username).first()
            if not user:
                logger.warning(f"MFA disable failed: User '{username}' not found.")
                return False

            user.mfa_enabled = False
            user.mfa_secret = None
            session.commit()

            logger.info(f"MFA disabled for user '{username}'.")

            # Log the MFA disable event
            self.metadata_storage.save_metadata({
                'event': 'mfa_disabled',
                'username': username,
                'timestamp': datetime.utcnow().isoformat()
            }, storage_type='auth_event')

            # Notify the user about MFA disable
            subject = "MFA Disabled for Your Hermod Account"
            message = f"""Hello {username},

Multi-Factor Authentication (MFA) has been disabled for your Hermod account. If you did not request this change, please contact our support team immediately.

Best regards,
Hermod Security Team
"""
            self.notification_manager.notify(channel='email', subject=subject, message=message, recipients=[user.email])

            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error disabling MFA for user '{username}': {e}")
            return False
        finally:
            session.close()


# Example usage:
if __name__ == "__main__":
    # Initialize Authentication Manager
    auth_manager = AuthenticationManager()

    # Register a new user
    success = auth_manager.register_user(
        username='john_doe',
        email='john@example.com',
        password='SecurePassword123!'
    )
    print("Registration successful:", success)

    # Authenticate the user (with MFA if enabled)
    token = auth_manager.authenticate_user(
        username='john_doe',
        password='SecurePassword123!',
        mfa_code='123456'  # Replace with actual TOTP code if MFA is enabled
    )
    print("Authentication token:", token)

    # Validate the token
    if token:
        payload = auth_manager.validate_token(token)
        print("Token payload:", payload)

    # Enable MFA for the user
    mfa_enabled = auth_manager.enable_mfa(username='john_doe')
    print("MFA enabled:", mfa_enabled)

    # Change the user's password
    password_changed = auth_manager.change_password(
        username='john_doe',
        old_password='SecurePassword123!',
        new_password='NewSecurePassword456!'
    )
    print("Password changed:", password_changed)
