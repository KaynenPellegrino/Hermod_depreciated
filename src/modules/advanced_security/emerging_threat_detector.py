# src/modules/advanced_security/emerging_threat_detector.py

import logging
import os
import json
from datetime import datetime
from typing import Optional, Dict, Any

from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from sqlalchemy.exc import SQLAlchemyError

# Import DataStorage from data_management module
from src.modules.data_management.data_storage import DataStorage

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
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import ipaddress
import pandas as pd

# Load environment variables
load_dotenv()

# Configure logging with RotatingFileHandler to prevent log files from becoming too large
from logging.handlers import RotatingFileHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = RotatingFileHandler('logs/emerging_threat_detector.log', maxBytes=5*1024*1024, backupCount=5)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)

# SQLAlchemy setup
Base = declarative_base()


class ThreatEvent(Base):
    """
    Represents a detected cybersecurity threat event.
    """
    __tablename__ = 'threat_events'

    id = Column(Integer, primary_key=True)
    threat_type = Column(String(255), nullable=False)
    description = Column(String(1024), nullable=False)
    severity = Column(Float, nullable=False)  # Severity score between 0 and 1
    detected_at = Column(DateTime, default=datetime.utcnow)
    source = Column(String(255), nullable=False)  # Data source identifier
    additional_info = Column(String(2048), nullable=True)  # JSON string with additional details

    def __repr__(self):
        return f"<ThreatEvent(threat_type='{self.threat_type}', severity={self.severity})>"


class EmergingThreatDetector:
    """
    Manages advanced threat detection using AI and machine learning techniques.
    """

    def __init__(self):
        """
        Initializes the EmergingThreatDetector with necessary configurations.
        """
        # Initialize Metadata Storage
        self.metadata_storage = DataStorage().metadata_storage  # Assuming DataStorage has metadata_storage attribute

        # Initialize Notification Manager
        self.notification_manager = NotificationManager()

        # Database configuration for storing threat events
        self.db_url = os.getenv('EMERGING_THREAT_DB_URL', 'sqlite:///emerging_threats.db')  # Example using SQLite
        self.engine = create_engine(self.db_url, echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

        # Initialize machine learning model and scaler
        try:
            # Define paths to model and scaler
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, '../../data/models/security_models/emerging_threat_model.h5')
            scaler_path = os.path.join(current_dir, '../../data/models/security_models/emerging_scaler.pkl')

            # Load the trained model
            if not os.path.exists(model_path):
                logger.error(f"Emerging threat detection model not found at {model_path}.")
                raise FileNotFoundError(f"Model file not found: {model_path}")
            self.model = load_model(model_path)
            logger.info("Emerging threat detection model loaded successfully.")

            # Load the pre-fitted scaler
            if not os.path.exists(scaler_path):
                logger.error(f"Emerging scaler not found at {scaler_path}.")
                raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            logger.info("Emerging scaler loaded successfully.")

        except Exception as e:
            logger.exception(f"Failed to initialize machine learning components: {e}")
            raise e  # Re-raise exception after logging

    def retrieve_data_for_detection(self) -> Optional[pd.DataFrame]:
        """
        Retrieves ingested data from storage systems for analysis.

        :return: DataFrame containing data for threat detection or None if retrieval fails.
        """
        try:
            # Example: Retrieve recent logs from a database or file storage
            # This should be customized based on actual data sources
            query = """
                SELECT 
                    id, 
                    event_type, 
                    source_ip, 
                    destination_ip, 
                    timestamp, 
                    details 
                FROM system_logs
                WHERE timestamp >= datetime('now', '-1 day');  -- Adjust based on your DBMS
            """
            df = DataStorage().load_from_sql(query, db_type='postgresql')  # Assuming PostgreSQL
            if df is None or df.empty:
                logger.warning("No data retrieved for threat detection.")
                return None
            logger.info(f"Retrieved {len(df)} records for threat detection.")
            return df
        except Exception as e:
            logger.error(f"Error retrieving data for threat detection: {e}")
            return None

    def preprocess_data(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Preprocesses the retrieved data to prepare it for the machine learning model.

        :param df: Raw DataFrame retrieved from data storage.
        :return: Numpy array of preprocessed features or None if preprocessing fails.
        """
        try:
            # Ensure required columns are present
            required_columns = ['event_type', 'source_ip', 'destination_ip', 'details']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Missing required columns for preprocessing: {missing_columns}")
                return None

            # Feature Engineering

            # Convert IP addresses to numerical format
            def ip_to_int(ip_str):
                try:
                    return int(ipaddress.IPv4Address(ip_str))
                except ipaddress.AddressValueError:
                    return 0  # Default value for invalid IPs

            df['source_ip_num'] = df['source_ip'].apply(ip_to_int)
            df['destination_ip_num'] = df['destination_ip'].apply(ip_to_int)

            # Process 'details' text to extract meaningful features
            # Example: Compute the length of the details string
            df['details_length'] = df['details'].astype(str).apply(len)

            # Drop original IP and details columns as we've extracted features
            df_processed = df.drop(['source_ip', 'destination_ip', 'details'], axis=1)

            # Define feature columns
            numerical_features = ['source_ip_num', 'destination_ip_num', 'details_length']
            categorical_features = ['event_type']

            # Define preprocessing pipelines for numerical and categorical data
            numerical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])

            categorical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

            # Combine preprocessing steps
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_pipeline, numerical_features),
                    ('cat', categorical_pipeline, categorical_features)
                ]
            )

            # Fit and transform the data
            X_processed = preprocessor.fit_transform(df_processed)

            # Optionally, save the preprocessor for future use (e.g., during inference)
            # preprocessor_save_path = os.path.join('path_to_save', 'emerging_preprocessor.pkl')
            # with open(preprocessor_save_path, 'wb') as f:
            #     pickle.dump(preprocessor, f)

            logger.info("Data preprocessing completed successfully.")
            return X_processed

        except Exception as e:
            logger.error(f"Error during preprocessing: {e}")
            return None

    def detect_threats(self, X: np.ndarray, original_df: pd.DataFrame) -> bool:
        """
        Applies the machine learning model to detect threats in the preprocessed data.

        :param X: Numpy array of preprocessed features.
        :param original_df: Original DataFrame corresponding to the preprocessed data.
        :return: True if threats are detected and handled, False otherwise.
        """
        try:
            # Predict threat probabilities
            predictions = self.model.predict(X)
            # Assume threshold of 0.5 for threat detection
            threat_indices = np.where(predictions.flatten() >= 0.5)[0]

            if len(threat_indices) == 0:
                logger.info("No threats detected in the current data batch.")
                return False

            # Iterate over detected threats and handle them
            for idx in threat_indices:
                threat_event = original_df.iloc[idx]
                threat_type = "Unknown"  # Placeholder; actual threat type determination may require additional logic
                description = "Suspicious activity detected based on system logs."
                severity = float(predictions[idx][0])  # Assuming prediction output is between 0 and 1
                source = "System Logs"  # Example source identifier
                additional_info = threat_event.to_json()

                # Create ThreatEvent record
                threat_record = ThreatEvent(
                    threat_type=threat_type,
                    description=description,
                    severity=severity,
                    source=source,
                    additional_info=additional_info
                )

                # Store the threat event in the database
                session = self.Session()
                try:
                    session.add(threat_record)
                    session.commit()
                    logger.info(f"Threat detected and recorded: {threat_type} with severity {severity:.2f}")

                    # Send notification
                    self.notification_manager.notify(
                        channel='email',
                        subject=f"Emerging Threat Detected: {threat_type}",
                        message=f"A threat of type '{threat_type}' with severity {severity:.2f} was detected.",
                        recipients=[os.getenv('ALERT_RECIPIENT')]
                    )

                    # Log the detection event
                    self.metadata_storage.save_metadata({
                        'event': 'threat_detection',
                        'threat_type': threat_type,
                        'severity': severity,
                        'timestamp': datetime.utcnow().isoformat()
                    }, storage_type='threat_detection_event')

                except SQLAlchemyError as e:
                    session.rollback()
                    logger.error(f"Database error while recording threat event: {e}")
                except Exception as e:
                    session.rollback()
                    logger.error(f"Unexpected error while recording threat event: {e}")
                finally:
                    session.close()

            logger.info(f"Total threats detected in current batch: {len(threat_indices)}")
            return True

        except Exception as e:
            logger.error(f"Error during threat detection: {e}")
            return False

    def train_threat_detection_model(self):
        """
        Trains the threat detection machine learning model using historical data.
        """
        session = self.Session()
        try:
            # Retrieve all threat events for training
            threats = session.query(ThreatEvent).all()
            if not threats:
                logger.warning("No threat events available for training.")
                return

            # Extract features and labels
            X = []
            y = []
            for threat in threats:
                # Example feature extraction from additional_info JSON
                details = json.loads(threat.additional_info)
                event_type = details.get('event_type', 'unknown')
                source_ip_num = int(details.get('source_ip_num', 0))
                destination_ip_num = int(details.get('destination_ip_num', 0))
                details_length = int(details.get('details_length', 0))

                # One-hot encoding for event_type
                event_types = ['login', 'file_access', 'network_scan', 'unknown']  # Example event types
                event_type_encoded = [1 if event_type == et else 0 for et in event_types]

                features = event_type_encoded + [source_ip_num, destination_ip_num, details_length]
                X.append(features)

                # Label: 1 for threat, 0 for non-threat
                y.append(1)  # All records in ThreatEvent are threats

            X = np.array(X)
            y = np.array(y)

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Initialize and fit the scaler
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
            model_save_path = os.path.join(current_dir, '../../data/models/security_models/emerging_threat_model.h5')
            model.save(model_save_path)
            logger.info(f"Trained emerging threat detection model saved at {model_save_path}.")

            # Save the fitted scaler
            scaler_save_path = os.path.join(current_dir, '../../data/models/security_models/emerging_scaler.pkl')
            with open(scaler_save_path, 'wb') as f:
                pickle.dump(scaler, f)
            logger.info(f"Fitted scaler saved at {scaler_save_path}.")

            # Optionally, notify admin about the successful training
            self.notification_manager.notify(
                channel='email',
                subject="Emerging Threat Detection Model Trained Successfully",
                message=f"The emerging threat detection model has been trained and saved successfully on {datetime.utcnow().isoformat()} UTC.",
                recipients=[os.getenv('ALERT_RECIPIENT')]
            )

        except Exception as e:
            logger.exception(f"Failed to train threat detection model: {e}")
        finally:
            session.close()

    def run_detection_pipeline(self):
        """
        Runs the complete detection pipeline: data retrieval, preprocessing, threat detection.
        """
        try:
            # Step 1: Retrieve data
            df = self.retrieve_data_for_detection()
            if df is None:
                logger.info("No data available for threat detection at this time.")
                return

            # Step 2: Preprocess data
            X = self.preprocess_data(df)
            if X is None:
                logger.error("Preprocessing failed. Threat detection aborted.")
                return

            # Step 3: Detect threats
            threats_detected = self.detect_threats(X, df)
            if threats_detected:
                logger.info("Threat detection process completed with detections.")
            else:
                logger.info("Threat detection process completed with no detections.")

        except Exception as e:
            logger.exception(f"Error running detection pipeline: {e}")

    def save_threat_event(self, threat_data: Dict[str, Any]) -> bool:
        """
        Saves a detected threat event to the database.

        :param threat_data: Dictionary containing threat event details.
        :return: True if successful, False otherwise.
        """
        session = self.Session()
        try:
            threat_record = ThreatEvent(
                threat_type=threat_data.get('threat_type', 'Unknown'),
                description=threat_data.get('description', ''),
                severity=threat_data.get('severity', 0.0),
                source=threat_data.get('source', 'Unknown'),
                additional_info=json.dumps(threat_data.get('additional_info', {}))
            )
            session.add(threat_record)
            session.commit()
            logger.info(f"Threat event saved: {threat_record}")
            return True
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Database error while saving threat event: {e}")
            return False
        except Exception as e:
            session.rollback()
            logger.error(f"Unexpected error while saving threat event: {e}")
            return False
        finally:
            session.close()

    # Additional methods can be added here for advanced functionalities


# Example usage and test cases
if __name__ == "__main__":
    # Initialize Emerging Threat Detector
    threat_detector = EmergingThreatDetector()

    # Example: Run the detection pipeline once
    threat_detector.run_detection_pipeline()

    # Example: Train the threat detection model (to be run periodically or manually)
    # threat_detector.train_threat_detection_model()
    # print("Threat detection model training initiated.")
