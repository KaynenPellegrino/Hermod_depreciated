# src/modules/advanced_security/behavioral_authentication.py

import logging
import os
from datetime import datetime
from typing import Optional, Dict, Any

from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from sqlalchemy.exc import SQLAlchemyError

# Import MetadataStorage from data_management module
from src.modules.data_management.metadata_storage import MetadataStorage

# Import NotificationManager from notifications module
from src.modules.notifications.notification_manager import NotificationManager

# For machine learning model handling
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Load environment variables
load_dotenv()

# Configure logging with RotatingFileHandler to prevent log files from becoming too large
from logging.handlers import RotatingFileHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = RotatingFileHandler('logs/behavioral_authentication.log', maxBytes=5*1024*1024, backupCount=5)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# SQLAlchemy setup
Base = declarative_base()


class BehavioralProfile(Base):
    """
    Represents a user's behavioral biometric profile.
    """
    __tablename__ = 'behavioral_profiles'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    typing_speed = Column(Float, nullable=False)                   # Characters per minute
    typing_pattern_similarity = Column(Float, nullable=False)      # Similarity score with stored pattern
    mouse_movement_similarity = Column(Float, nullable=False)      # Similarity score with stored pattern
    login_time_variance = Column(Float, nullable=False)            # Variance in login times
    device_fingerprint = Column(String(255), nullable=False)       # Device fingerprint hash
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="behavioral_profile")


class BehavioralAuthenticationManager:
    """
    Manages behavior-based authentication using behavioral biometrics.
    """

    def __init__(self):
        """
        Initializes the BehavioralAuthenticationManager with necessary configurations.
        """
        # Initialize Metadata Storage
        self.metadata_storage = MetadataStorage()

        # Initialize Notification Manager
        self.notification_manager = NotificationManager()

        # Database configuration
        self.db_url = os.getenv('BEHAVIORAL_AUTH_DB_URL', 'sqlite:///behavioral_auth.db')  # Example using SQLite
        self.engine = create_engine(self.db_url, echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

        # Initialize machine learning model and scaler
        try:
            # Define paths to model and scaler
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, '../../data/models/cybersecurity_models/behavioral_authentication_model.h5')
            scaler_path = os.path.join(current_dir, '../../data/models/cybersecurity_models/behavioral_scaler.pkl')

            # Load the trained model
            if not os.path.exists(model_path):
                logger.error(f"Behavioral authentication model not found at {model_path}.")
                raise FileNotFoundError(f"Model file not found: {model_path}")
            self.model = load_model(model_path)
            logger.info("Behavioral authentication model loaded successfully.")

            # Load the pre-fitted scaler
            if not os.path.exists(scaler_path):
                logger.error(f"Behavioral scaler not found at {scaler_path}.")
                raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            logger.info("Behavioral scaler loaded successfully.")

        except Exception as e:
            logger.exception(f"Failed to initialize machine learning components: {e}")
            raise e  # Re-raise exception after logging

    def register_behavior_profile(self, user_id: int, behavior_data: Dict[str, Any]) -> bool:
        """
        Registers or updates a user's behavioral biometric profile.

        :param user_id: ID of the user.
        :param behavior_data: Dictionary containing behavioral metrics.
        :return: True if registration is successful, False otherwise.
        """
        session = self.Session()
        try:
            profile = session.query(BehavioralProfile).filter(BehavioralProfile.user_id == user_id).first()
            if profile:
                # Update existing profile
                profile.typing_speed = behavior_data.get('typing_speed', profile.typing_speed)
                profile.typing_pattern_similarity = behavior_data.get('typing_pattern_similarity', profile.typing_pattern_similarity)
                profile.mouse_movement_similarity = behavior_data.get('mouse_movement_similarity', profile.mouse_movement_similarity)
                profile.login_time_variance = behavior_data.get('login_time_variance', profile.login_time_variance)
                profile.device_fingerprint = behavior_data.get('device_fingerprint', profile.device_fingerprint)
                logger.info(f"Behavioral profile updated for user ID {user_id}.")
            else:
                # Create new profile
                new_profile = BehavioralProfile(
                    user_id=user_id,
                    typing_speed=behavior_data['typing_speed'],
                    typing_pattern_similarity=behavior_data['typing_pattern_similarity'],
                    mouse_movement_similarity=behavior_data['mouse_movement_similarity'],
                    login_time_variance=behavior_data['login_time_variance'],
                    device_fingerprint=behavior_data['device_fingerprint']
                )
                session.add(new_profile)
                logger.info(f"Behavioral profile created for user ID {user_id}.")

            session.commit()

            # Log the registration/update event
            self.metadata_storage.save_metadata({
                'event': 'behavioral_profile_registration',
                'user_id': user_id,
                'timestamp': datetime.utcnow().isoformat()
            }, storage_type='behavioral_auth_event')

            return True
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Database error during behavioral profile registration for user ID {user_id}: {e}")
            return False
        except KeyError as e:
            logger.error(f"Missing behavioral data key: {e}")
            return False
        except Exception as e:
            session.rollback()
            logger.error(f"Unexpected error during behavioral profile registration for user ID {user_id}: {e}")
            return False
        finally:
            session.close()

    def assess_behavior(self, user_id: int, current_behavior: Dict[str, Any]) -> float:
        """
        Assesses the user's current behavior against their stored profile and returns a similarity score.

        :param user_id: ID of the user.
        :param current_behavior: Dictionary containing current behavioral metrics.
        :return: Similarity score between 0 and 1. Higher scores indicate higher similarity.
        """
        session = self.Session()
        try:
            profile = session.query(BehavioralProfile).filter(BehavioralProfile.user_id == user_id).first()
            if not profile:
                logger.warning(f"No behavioral profile found for user ID {user_id}.")
                return 0.0  # Lowest score if no profile exists

            # Extract stored metrics
            stored_metrics = np.array([
                profile.typing_speed,
                profile.typing_pattern_similarity,
                profile.mouse_movement_similarity,
                profile.login_time_variance
            ]).reshape(1, -1)

            # Extract current metrics
            current_metrics = np.array([
                current_behavior.get('typing_speed', 0.0),
                current_behavior.get('typing_pattern_similarity', 0.0),
                current_behavior.get('mouse_movement_similarity', 0.0),
                current_behavior.get('login_time_variance', 0.0)
            ]).reshape(1, -1)

            # Normalize metrics using the pre-fitted scaler
            stored_normalized = self.scaler.transform(stored_metrics)
            current_normalized = self.scaler.transform(current_metrics)

            # Predict similarity score using the loaded model
            similarity_score = self.model.predict(current_normalized)[0][0]  # Assuming model outputs a single value

            # Ensure the similarity score is within [0,1]
            similarity_score = max(0.0, min(1.0, similarity_score))

            logger.info(f"Behavioral assessment for user ID {user_id}: Similarity Score = {similarity_score:.4f}")

            return similarity_score
        except Exception as e:
            logger.error(f"Error during behavioral assessment for user ID {user_id}: {e}")
            return 0.0  # Return lowest score on error
        finally:
            session.close()

    def generate_device_fingerprint(self, device_info: Dict[str, Any]) -> str:
        """
        Generates a device fingerprint based on device information.

        :param device_info: Dictionary containing device attributes.
        :return: A hashed string representing the device fingerprint.
        """
        import hashlib
        fingerprint_string = ''.join([str(value) for value in device_info.values()])
        fingerprint_hash = hashlib.sha256(fingerprint_string.encode('utf-8')).hexdigest()
        logger.debug(f"Generated device fingerprint: {fingerprint_hash}")
        return fingerprint_hash

    def train_behavioral_model(self):
        """
        Trains a behavioral authentication model using stored profiles.
        Integrates machine learning models for enhanced assessments.
        """
        session = self.Session()
        try:
            # Retrieve all behavioral profiles
            profiles = session.query(BehavioralProfile).all()
            if not profiles:
                logger.warning("No behavioral profiles available for training.")
                return

            # Extract features and labels
            X = np.array([
                [
                    profile.typing_speed,
                    profile.typing_pattern_similarity,
                    profile.mouse_movement_similarity,
                    profile.login_time_variance
                ] for profile in profiles
            ])

            # Retrieve labels from the User model or another source
            # Assuming there's a 'is_anomalous' attribute in the User model
            y = np.array([
                profile.user.is_anomalous if hasattr(profile.user, 'is_anomalous') else 1
                for profile in profiles
            ])

            # Check if there are both classes present
            unique_labels = np.unique(y)
            if len(unique_labels) < 2:
                logger.warning("Insufficient classes for training. Need both legitimate and anomalous samples.")
                return

            # Example: Creating synthetic anomalies by adding noise
            num_anomalies = int(0.2 * X.shape[0])  # 20% anomalies
            if num_anomalies > 0:
                anomaly_data = X[:num_anomalies] + np.random.normal(0, 0.5, X[:num_anomalies].shape)
                X = np.vstack((X, anomaly_data))
                y = np.hstack((y, np.zeros(num_anomalies)))  # Label anomalies as 0

                # Split data into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Initialize and fit the scaler on training data
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Define the model architecture
                model = Sequential([
                    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
                    Dense(32, activation='relu'),
                    Dense(1, activation='sigmoid')  # Output layer for binary classification
                ])

                # Compile the model
                model.compile(optimizer=Adam(learning_rate=0.001),
                              loss='binary_crossentropy',
                              metrics=['accuracy'])

                # Define early stopping to prevent overfitting
                early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

                # Train the model
                history = model.fit(
                    X_train_scaled, y_train,
                    epochs=100,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[early_stop],
                    verbose=1
                )

                # Evaluate the model on test data
                y_pred = (model.predict(X_test_scaled) > 0.5).astype(int).flatten()
                report = classification_report(y_test, y_pred)
                cm = confusion_matrix(y_test, y_pred)
                logger.info(f"Model Evaluation Report:\n{report}")
                logger.info(f"Confusion Matrix:\n{cm}")

                # Save the trained model
                current_dir = os.path.dirname(os.path.abspath(__file__))
                model_save_path = os.path.join(current_dir,
                                               '../../data/models/cybersecurity_models/behavioral_authentication_model.h5')
                model.save(model_save_path)
                logger.info(f"Trained behavioral authentication model saved at {model_save_path}.")

                # Save the fitted scaler
                scaler_save_path = os.path.join(current_dir,
                                                '../../data/models/cybersecurity_models/behavioral_scaler.pkl')
                with open(scaler_save_path, 'wb') as f:
                    pickle.dump(scaler, f)
                logger.info(f"Fitted scaler saved at {scaler_save_path}.")

                # Optionally, notify admin about the successful training
                self.notification_manager.notify(
                    channel='email',
                    subject="Behavioral Authentication Model Trained Successfully",
                    message=f"The behavioral authentication model has been trained and saved successfully on {datetime.utcnow().isoformat()} UTC.",
                    recipients=[os.getenv('ALERT_RECIPIENT')]
                )

        except Exception as e:
            logger.exception(f"Error during behavioral model training: {e}")
        finally:
            session.close()

    def save_behavioral_profile(self, user_id: int, behavior_data: Dict[str, Any]) -> bool:
        """
        Saves or updates the user's behavioral profile.

        :param user_id: ID of the user.
        :param behavior_data: Dictionary containing behavioral metrics.
        :return: True if successful, False otherwise.
        """
        return self.register_behavior_profile(user_id, behavior_data)

    # Additional methods can be added here for advanced functionalities


# Example usage:
if __name__ == "__main__":
    # Initialize Behavioral Authentication Manager
    behavior_manager = BehavioralAuthenticationManager()

    # Example: Register a new behavioral profile for user ID 1
    user_id = 1
    behavior_data = {
        'typing_speed': 60.0,  # Characters per minute
        'typing_pattern_similarity': 0.85,  # Similarity score with stored pattern
        'mouse_movement_similarity': 0.80,  # Similarity score with stored pattern
        'login_time_variance': 15.0,  # Variance in login times (minutes)
        'device_fingerprint': behavior_manager.generate_device_fingerprint({
            'browser': 'Chrome',
            'os': 'Windows',
            'ip_address': '192.168.1.1'
        })
    }
    success = behavior_manager.save_behavioral_profile(user_id, behavior_data)
    print("Behavioral profile registration successful:", success)

    # Example: Assess behavior during login attempt
    current_behavior = {
        'typing_speed': 58.0,
        'typing_pattern_similarity': 0.80,
        'mouse_movement_similarity': 0.78,
        'login_time_variance': 12.0
    }
    similarity_score = behavior_manager.assess_behavior(user_id, current_behavior)
    print(f"Similarity Score for user ID {user_id}: {similarity_score:.4f}")

    # Example: Train the behavioral authentication model
    behavior_manager.train_behavioral_model()
    print("Behavioral authentication model training initiated.")
