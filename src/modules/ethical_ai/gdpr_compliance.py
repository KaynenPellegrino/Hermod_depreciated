# src/modules/ethical_ai/gdpr_compliance.py

import os
import logging
from typing import Any, Dict, Optional
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from cryptography.fernet import Fernet
from src.utils.configuration_manager import ConfigurationManager
from src.modules.notifications.notification_manager import NotificationManager

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('logs/gdpr_compliance.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# SQLAlchemy setup
Base = declarative_base()


class UserConsent(Base):
    __tablename__ = 'user_consent'
    id = Column(Integer, primary_key=True)
    user_id = Column(String, unique=True, nullable=False)
    consent_given = Column(Boolean, default=False)
    consent_timestamp = Column(DateTime, default=datetime.utcnow)
    data = Column(Text)  # Encrypted personal data


class GDPRCompliance:
    """
    Data Protection Compliance
    Ensures that the system complies with GDPR regulations regarding data privacy and protection.
    Handles consent management, data anonymization, and user data requests.
    """

    def __init__(self):
        """
        Initializes the GDPRCompliance with necessary configurations.
        """
        try:
            self.config_manager = ConfigurationManager()
            self.notification_manager = NotificationManager()
            self.load_gdpr_config()
            self.setup_database()
            self.setup_encryption()
            logger.info("GDPRCompliance initialized successfully.")
        except Exception as e:
            logger.exception(f"Failed to initialize GDPRCompliance: {e}")
            raise e

    def load_gdpr_config(self):
        """
        Loads GDPR configurations from the configuration manager or environment variables.
        """
        logger.info("Loading GDPR configurations.")
        try:
            self.gdpr_config = {
                'database_url': self.config_manager.get('DATABASE_URL', 'sqlite:///data/gdpr_compliance.db'),
                'encryption_key': self.config_manager.get('ENCRYPTION_KEY', Fernet.generate_key().decode()),
                'notification_recipients': self.config_manager.get('NOTIFICATION_RECIPIENTS', '').split(','),
            }
            logger.info(f"GDPR configurations loaded: {self.gdpr_config}")
        except Exception as e:
            logger.error(f"Failed to load GDPR configurations: {e}")
            raise e

    def setup_database(self):
        """
        Sets up the database connection and tables.
        """
        logger.info("Setting up the database.")
        try:
            self.engine = create_engine(self.gdpr_config['database_url'])
            Base.metadata.create_all(self.engine)
            self.Session = sessionmaker(bind=self.engine)
            logger.info("Database setup complete.")
        except Exception as e:
            logger.error(f"Failed to set up the database: {e}")
            raise e

    def setup_encryption(self):
        """
        Sets up the encryption key and cipher.
        """
        logger.info("Setting up encryption.")
        try:
            self.encryption_key = self.gdpr_config['encryption_key'].encode()
            self.cipher = Fernet(self.encryption_key)
            logger.info("Encryption setup complete.")
        except Exception as e:
            logger.error(f"Failed to set up encryption: {e}")
            raise e

    # --------------------- Consent Management --------------------- #

    def obtain_consent(self, user_id: str) -> bool:
        """
        Obtains consent from the user for data processing.

        :param user_id: Unique identifier of the user.
        :return: True if consent is obtained, False otherwise.
        """
        logger.info(f"Obtaining consent from user '{user_id}'.")
        try:
            session = self.Session()
            user_consent = session.query(UserConsent).filter_by(user_id=user_id).first()
            if user_consent:
                logger.info(f"User '{user_id}' has already given consent.")
                return user_consent.consent_given
            else:
                # Simulate obtaining consent (e.g., through a user interface)
                consent_given = True  # In real implementation, get actual consent
                new_consent = UserConsent(
                    user_id=user_id,
                    consent_given=consent_given,
                    consent_timestamp=datetime.utcnow()
                )
                session.add(new_consent)
                session.commit()
                logger.info(f"Consent obtained from user '{user_id}': {consent_given}")
                return consent_given
        except Exception as e:
            logger.error(f"Failed to obtain consent from user '{user_id}': {e}")
            raise e
        finally:
            session.close()

    def withdraw_consent(self, user_id: str) -> bool:
        """
        Withdraws consent for data processing from the user.

        :param user_id: Unique identifier of the user.
        :return: True if consent is withdrawn, False otherwise.
        """
        logger.info(f"Withdrawing consent from user '{user_id}'.")
        try:
            session = self.Session()
            user_consent = session.query(UserConsent).filter_by(user_id=user_id).first()
            if user_consent:
                user_consent.consent_given = False
                user_consent.consent_timestamp = datetime.utcnow()
                session.commit()
                logger.info(f"Consent withdrawn from user '{user_id}'.")
                return True
            else:
                logger.warning(f"No consent record found for user '{user_id}'.")
                return False
        except Exception as e:
            logger.error(f"Failed to withdraw consent from user '{user_id}': {e}")
            raise e
        finally:
            session.close()

    # --------------------- Data Anonymization --------------------- #

    def store_personal_data(self, user_id: str, personal_data: Dict[str, Any]):
        """
        Stores encrypted personal data for the user.

        :param user_id: Unique identifier of the user.
        :param personal_data: Dictionary of personal data to store.
        """
        logger.info(f"Storing personal data for user '{user_id}'.")
        try:
            session = self.Session()
            user_consent = session.query(UserConsent).filter_by(user_id=user_id).first()
            if user_consent and user_consent.consent_given:
                # Encrypt personal data
                data_str = str(personal_data)
                encrypted_data = self.cipher.encrypt(data_str.encode()).decode()
                user_consent.data = encrypted_data
                session.commit()
                logger.info(f"Personal data stored for user '{user_id}'.")
            else:
                logger.warning(f"Consent not given by user '{user_id}'. Cannot store data.")
        except Exception as e:
            logger.error(f"Failed to store personal data for user '{user_id}': {e}")
            raise e
        finally:
            session.close()

    def anonymize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Anonymizes personal data to protect user identities.

        :param data: Dictionary of data to anonymize.
        :return: Anonymized data dictionary.
        """
        logger.info("Anonymizing data.")
        try:
            anonymized_data = data.copy()
            # Remove or mask personally identifiable information
            pii_fields = ['name', 'email', 'phone', 'address', 'user_id']
            for field in pii_fields:
                if field in anonymized_data:
                    anonymized_data[field] = 'REDACTED'
            logger.info("Data anonymization complete.")
            return anonymized_data
        except Exception as e:
            logger.error(f"Failed to anonymize data: {e}")
            raise e

    # --------------------- User Data Requests --------------------- #

    def get_user_data(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the user's personal data.

        :param user_id: Unique identifier of the user.
        :return: Dictionary of personal data, or None if not found.
        """
        logger.info(f"Retrieving personal data for user '{user_id}'.")
        try:
            session = self.Session()
            user_consent = session.query(UserConsent).filter_by(user_id=user_id).first()
            if user_consent and user_consent.data:
                # Decrypt personal data
                encrypted_data = user_consent.data.encode()
                decrypted_data = self.cipher.decrypt(encrypted_data).decode()
                personal_data = eval(decrypted_data)
                logger.info(f"Personal data retrieved for user '{user_id}'.")
                return personal_data
            else:
                logger.warning(f"No personal data found for user '{user_id}'.")
                return None
        except Exception as e:
            logger.error(f"Failed to retrieve personal data for user '{user_id}': {e}")
            raise e
        finally:
            session.close()

    def delete_user_data(self, user_id: str) -> bool:
        """
        Deletes the user's personal data upon request.

        :param user_id: Unique identifier of the user.
        :return: True if data is deleted, False otherwise.
        """
        logger.info(f"Deleting personal data for user '{user_id}'.")
        try:
            session = self.Session()
            user_consent = session.query(UserConsent).filter_by(user_id=user_id).first()
            if user_consent:
                session.delete(user_consent)
                session.commit()
                logger.info(f"Personal data deleted for user '{user_id}'.")
                return True
            else:
                logger.warning(f"No personal data found for user '{user_id}'.")
                return False
        except Exception as e:
            logger.error(f"Failed to delete personal data for user '{user_id}': {e}")
            raise e
        finally:
            session.close()

    # --------------------- Notification Method --------------------- #

    def send_notification(self, subject: str, message: str):
        """
        Sends a notification to the configured recipients.

        :param subject: Subject of the notification.
        :param message: Body of the notification.
        """
        try:
            recipients = self.gdpr_config['notification_recipients']
            if recipients:
                self.notification_manager.send_notification(
                    recipients=recipients,
                    subject=subject,
                    message=message
                )
                logger.info("Notification sent successfully.")
            else:
                logger.warning("No notification recipients configured.")
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")

# --------------------- Example Usage --------------------- #

def example_usage():
    """
    Demonstrates example usage of the GDPRCompliance class.
    """
    try:
        # Initialize GDPRCompliance
        gdpr = GDPRCompliance()

        # User ID
        user_id = 'user123'

        # Obtain consent
        consent_obtained = gdpr.obtain_consent(user_id)
        if consent_obtained:
            # Store personal data
            personal_data = {
                'name': 'Alice Smith',
                'email': 'alice@example.com',
                'phone': '123-456-7890',
                'address': '123 Main St, Anytown, USA',
                'preferences': {'newsletter': True}
            }
            gdpr.store_personal_data(user_id, personal_data)

            # Retrieve personal data
            retrieved_data = gdpr.get_user_data(user_id)
            print("Retrieved Personal Data:")
            print(retrieved_data)

            # Anonymize data for processing
            anonymized_data = gdpr.anonymize_data(retrieved_data)
            print("Anonymized Data:")
            print(anonymized_data)

            # Handle user data deletion request
            data_deleted = gdpr.delete_user_data(user_id)
            if data_deleted:
                print(f"User data for '{user_id}' has been deleted.")
            else:
                print(f"Failed to delete user data for '{user_id}'.")

        else:
            print(f"Consent not obtained from user '{user_id}'.")

    except Exception as e:
        logger.exception(f"Error in example usage: {e}")

# --------------------- Main Execution --------------------- #

if __name__ == "__main__":
    # Run the GDPR compliance example
    example_usage()
