# src/modules/advanced_security/behavioral_authentication.py

import logging
import os
import pickle
from datetime import datetime
from typing import Dict, Any

import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker, declarative_base, relationship

# Import PyTorch modules
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Import MetadataStorage from data_management module
from src.modules.data_management.staging import MetadataStorage
# Import NotificationManager from notifications module
from src.modules.notifications.staging import NotificationManager

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


class BehavioralAuthenticationModel(nn.Module):
    """
    PyTorch model for behavioral authentication.
    """
    def __init__(self, input_size: int):
        super(BehavioralAuthenticationModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


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

        # PyTorch model setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BehavioralAuthenticationModel(input_size=4).to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Load model and scaler if available
        self.model_path = os.path.join(os.path.dirname(__file__), '../../data/models/cybersecurity_models/behavioral_authentication_model.pth')
        self.scaler_path = os.path.join(os.path.dirname(__file__), '../../data/models/cybersecurity_models/behavioral_scaler.pkl')
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path))
            self.model.eval()
            logger.info("Behavioral authentication model loaded successfully.")

        if os.path.exists(self.scaler_path):
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            logger.info("Behavioral scaler loaded successfully.")
        else:
            self.scaler = StandardScaler()

    def register_behavior_profile(self, user_id: int, behavior_data: Dict[str, Any]) -> bool:
        """
        Registers or updates a user's behavioral biometric profile.
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
        """
        session = self.Session()
        try:
            profile = session.query(BehavioralProfile).filter(BehavioralProfile.user_id == user_id).first()
            if not profile:
                logger.warning(f"No behavioral profile found for user ID {user_id}.")
                return 0.0

            # Extract stored and current metrics
            stored_metrics = np.array([
                profile.typing_speed,
                profile.typing_pattern_similarity,
                profile.mouse_movement_similarity,
                profile.login_time_variance
            ]).reshape(1, -1)

            current_metrics = np.array([
                current_behavior.get('typing_speed', 0.0),
                current_behavior.get('typing_pattern_similarity', 0.0),
                current_behavior.get('mouse_movement_similarity', 0.0),
                current_behavior.get('login_time_variance', 0.0)
            ]).reshape(1, -1)

            # Normalize metrics using the pre-fitted scaler
            stored_normalized = self.scaler.transform(stored_metrics)
            current_normalized = self.scaler.transform(current_metrics)

            # Predict similarity score using the PyTorch model
            self.model.eval()
            with torch.no_grad():
                similarity_score = self.model(torch.tensor(current_normalized, dtype=torch.float32).to(self.device)).item()

            similarity_score = max(0.0, min(1.0, similarity_score))
            logger.info(f"Behavioral assessment for user ID {user_id}: Similarity Score = {similarity_score:.4f}")
            return similarity_score
        except Exception as e:
            logger.error(f"Error during behavioral assessment for user ID {user_id}: {e}")
            return 0.0
        finally:
            session.close()

    def train_behavioral_model(self):
        """
        Trains the behavioral authentication model using stored profiles.
        """
        session = self.Session()
        try:
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
            y = np.array([1] * len(X))  # Assuming all samples are legitimate for this example

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Normalize data
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Prepare data for PyTorch
            train_dataset = TensorDataset(
                torch.tensor(X_train_scaled, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32)
            )
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

            # Train the model
            self.model.train()
            for epoch in range(100):  # Adjust epochs as needed
                epoch_loss = 0.0
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    self.optimizer.zero_grad()
                    outputs = self.model(batch_X).squeeze()
                    loss = self.criterion(outputs, batch_y)
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item()
                logger.info(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")

            # Save the trained model and scaler
            torch.save(self.model.state_dict(), self.model_path)
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            logger.info("Model and scaler saved successfully.")
        except Exception as e:
            logger.exception(f"Error during behavioral model training: {e}")
        finally:
            session.close()
