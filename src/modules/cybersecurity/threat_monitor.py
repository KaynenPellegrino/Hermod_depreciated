# src/modules/cybersecurity/threat_monitor.py

import os
import threading
import time
from datetime import datetime
from typing import Optional, List

import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import IsolationForest
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

# Import MetadataStorage from data_management module
from src.modules.data_management.staging import MetadataStorage
# Import NotificationManager from notifications module
from src.modules.notifications.staging import NotificationManager

# Load environment variables from .env file
load_dotenv()

# Configure logging with RotatingFileHandler to prevent log files from becoming too large

from src.utils.logger import get_logger

logger = get_logger(__name__, 'logs/threat_monitor.log')


class LogEventHandler(FileSystemEventHandler):
    """
    Custom event handler for monitoring log file changes.
    """

    def __init__(self, monitor):
        super().__init__()
        self.monitor = monitor

    def on_modified(self, event):
        if not event.is_directory and event.src_path == self.monitor.log_file_path:
            logger.debug(f"Detected modification in log file: {event.src_path}")
            self.monitor.process_log_file()


class ThreatMonitor:
    """
    Monitors system logs, network traffic, and system behaviors to detect security threats and anomalies.
    """

    def __init__(self):
        """
        Initializes the ThreatMonitor with necessary configurations.
        """
        # Initialize Metadata Storage
        self.metadata_storage = MetadataStorage()

        # Initialize Notification Manager
        self.notification_manager = NotificationManager()

        # Configuration parameters
        self.log_file_path = os.getenv('THREAT_MONITOR_LOG_FILE_PATH', '/var/log/hermod/hermod.log')
        self.network_interface = os.getenv('THREAT_MONITOR_NETWORK_INTERFACE', 'eth0')
        self.fetch_interval = int(os.getenv('THREAT_MONITOR_FETCH_INTERVAL', '60'))  # in seconds
        self.observer = Observer()
        self.model = self.initialize_anomaly_detection_model()

        logger.info("ThreatMonitor initialized successfully.")

    def initialize_anomaly_detection_model(self) -> Optional[IsolationForest]:
        """
        Initializes the anomaly detection model.

        :return: Trained IsolationForest model or None if training data is unavailable
        """
        try:
            # Load historical log data for training
            historical_data_path = os.getenv('THREAT_MONITOR_HISTORICAL_DATA_PATH', 'data/historical_logs.csv')
            if not os.path.exists(historical_data_path):
                logger.warning(f"Historical data file '{historical_data_path}' not found. Anomaly detection model not initialized.")
                return None

            data = pd.read_csv(historical_data_path)
            # Feature engineering can be performed here
            # For simplicity, assuming numerical features are present
            X = data.select_dtypes(include=['float64', 'int64'])

            model = IsolationForest(contamination=0.01, random_state=42)
            model.fit(X)
            logger.info("Anomaly detection model initialized and trained successfully.")
            return model
        except Exception as e:
            logger.error(f"Failed to initialize anomaly detection model: {e}")
            return None

    def start_log_monitoring(self):
        """
        Starts monitoring the specified log file for changes.
        """
        event_handler = LogEventHandler(self)
        log_dir = os.path.dirname(self.log_file_path)
        self.observer.schedule(event_handler, path=log_dir, recursive=False)
        self.observer.start()
        logger.info(f"Started monitoring log file: {self.log_file_path}")

    def process_log_file(self):
        """
        Processes the log file to detect anomalies or security threats.
        """
        try:
            with open(self.log_file_path, 'r') as f:
                lines = f.readlines()

            # Assuming the latest entries are at the end
            # Implement logic to parse and analyze log entries
            latest_logs = lines[-100:]  # Read last 100 lines as an example
            log_data = self.parse_logs(latest_logs)

            # Analyze logs for anomalies
            if self.model:
                anomalies = self.detect_anomalies(log_data)
                if anomalies:
                    self.handle_anomalies(anomalies)
            else:
                logger.warning("Anomaly detection model is not initialized. Skipping anomaly analysis.")

            # Save log data to Metadata Storage
            report = {
                'timestamp': datetime.utcnow().isoformat(),
                'log_entries': log_data.to_dict(orient='records')
            }
            self.metadata_storage.save_metadata(report, storage_type='log_monitoring')

        except Exception as e:
            logger.error(f"Failed to process log file '{self.log_file_path}': {e}")

    def parse_logs(self, log_lines: List[str]) -> pd.DataFrame:
        """
        Parses log lines into a structured DataFrame.

        :param log_lines: List of log lines
        :return: DataFrame containing parsed log data
        """
        try:
            # Example log format: "2024-10-12 12:34:56,789 - INFO - User login successful for user 'admin'"
            log_entries = []
            for line in log_lines:
                try:
                    parts = line.strip().split(' - ')
                    if len(parts) < 3:
                        continue
                    timestamp_str, level, message = parts[:3]
                    timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
                    log_entries.append({
                        'timestamp': timestamp,
                        'level': level,
                        'message': message
                    })
                except Exception as inner_e:
                    logger.debug(f"Failed to parse log line: {line.strip()} - {inner_e}")

            df = pd.DataFrame(log_entries)
            logger.debug(f"Parsed {len(df)} log entries.")
            return df
        except Exception as e:
            logger.error(f"Failed to parse logs: {e}")
            return pd.DataFrame()

    def detect_anomalies(self, log_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Detects anomalies in the log data using the anomaly detection model.

        :param log_data: DataFrame containing parsed log data
        :return: DataFrame containing anomalous log entries or None
        """
        try:
            # Feature engineering: Convert categorical data to numerical features
            # Example: Encode log levels
            log_data_encoded = log_data.copy()
            log_data_encoded['level'] = log_data_encoded['level'].astype('category').cat.codes

            # Assuming 'message' can be vectorized or transformed into numerical features
            # For simplicity, using length of the message as a feature
            log_data_encoded['message_length'] = log_data_encoded['message'].apply(len)

            X = log_data_encoded[['level', 'message_length']]

            predictions = self.model.predict(X)
            log_data_encoded['anomaly'] = predictions

            anomalies = log_data_encoded[log_data_encoded['anomaly'] == -1]

            if not anomalies.empty:
                logger.info(f"Detected {len(anomalies)} anomalies in log data.")
                return anomalies
            else:
                logger.info("No anomalies detected in log data.")
                return None

        except Exception as e:
            logger.error(f"Failed to detect anomalies: {e}")
            return None

    def handle_anomalies(self, anomalies: pd.DataFrame):
        """
        Handles detected anomalies by triggering alerts and logging them.

        :param anomalies: DataFrame containing anomalous log entries
        """
        try:
            # Save anomalies to Metadata Storage
            report = {
                'timestamp': datetime.utcnow().isoformat(),
                'anomalies': anomalies.to_dict(orient='records')
            }
            self.metadata_storage.save_metadata(report, storage_type='anomaly_detection')

            # Trigger alerts via Notification Manager
            subject = f"Security Alert: Detected {len(anomalies)} Anomalies"
            message = f"Anomalous activities have been detected in the system logs:\n\n{anomalies.to_string(index=False)}"
            self.notification_manager.send_email(subject, message)
            logger.info("Triggered security alert for detected anomalies.")

        except Exception as e:
            logger.error(f"Failed to handle anomalies: {e}")

    def monitor_network_traffic(self):
        """
        Monitors network traffic for suspicious activities.

        Note: This is a placeholder function. Implement network traffic monitoring as needed.
        """
        try:
            # Example: Capture packets using scapy (requires elevated permissions)
            from scapy.all import sniff

            def packet_callback(packet):
                # Implement packet analysis logic
                if packet.haslayer('TCP') and packet['TCP'].dport == 22:
                    # Example: Detect unusual SSH attempts
                    logger.debug(f"Detected SSH packet: {packet.summary()}")

            logger.info(f"Starting network traffic monitoring on interface {self.network_interface}.")
            sniff(iface=self.network_interface, prn=packet_callback, store=False, timeout=60)

        except Exception as e:
            logger.error(f"Failed to monitor network traffic: {e}")

    def run(self):
        """
        Starts the ThreatMonitor, initiating log and network monitoring concurrently.
        """
        logger.info("Starting ThreatMonitor.")

        # Start log monitoring in a separate thread
        log_thread = threading.Thread(target=self.start_log_monitoring)
        log_thread.start()

        # Start network traffic monitoring in a separate thread
        network_thread = threading.Thread(target=self.monitor_network_traffic)
        network_thread.start()

        try:
            while True:
                # Main thread can perform additional tasks or remain idle
                time.sleep(self.fetch_interval)
        except KeyboardInterrupt:
            logger.info("ThreatMonitor stopped manually.")
            self.observer.stop()
        except Exception as e:
            logger.exception(f"An unexpected error occurred: {e}")
            self.observer.stop()

        self.observer.join()
        log_thread.join()
        network_thread.join()
        logger.info("ThreatMonitor has been terminated.")


if __name__ == "__main__":
    try:
        monitor = ThreatMonitor()
        monitor.run()
    except Exception as e:
        logger.exception(f"Failed to start ThreatMonitor: {e}")
