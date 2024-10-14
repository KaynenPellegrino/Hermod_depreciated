# src/modules/analytics/system_health_monitor.py

import logging
import os
import time
from datetime import datetime
from typing import Optional, Dict, Any

import psutil
from dotenv import load_dotenv

# Import DataStorage from data_management module
from src.modules.data_management.data_storage import DataStorage

# Import NotificationManager from notifications module
from src.modules.notifications.notification_manager import NotificationManager

# Configure logging with RotatingFileHandler to prevent log files from becoming too large
from logging.handlers import RotatingFileHandler

# Load environment variables
load_dotenv()

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = RotatingFileHandler(
    os.path.join('logs', 'system_health_monitor.log'),
    maxBytes=5 * 1024 * 1024,  # 5 MB
    backupCount=5
)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)


class SystemHealthMonitor:
    """
    Continuously monitors the health of Hermod and its system components,
    including performance, memory usage, and system logs. Generates alerts
    if any anomalies are detected and sends notifications through the
    NotificationManager.
    """

    def __init__(self, check_interval: int = 60):
        """
        Initializes the SystemHealthMonitor with necessary configurations.

        :param check_interval: Time interval (in seconds) between health checks.
        """
        self.check_interval = check_interval  # Time between checks in seconds
        try:
            # Initialize DataStorage instance for storing health metrics
            self.data_storage = DataStorage()
            logger.info("SystemHealthMonitor initialized successfully.")
        except Exception as e:
            logger.exception(f"Failed to initialize SystemHealthMonitor: {e}")
            raise e

        try:
            # Initialize NotificationManager instance for sending alerts
            self.notification_manager = NotificationManager()
            logger.info("NotificationManager initialized successfully.")
        except Exception as e:
            logger.exception(f"Failed to initialize NotificationManager: {e}")
            raise e

        # Load health thresholds from environment variables or set defaults
        self.cpu_threshold = float(os.getenv('CPU_USAGE_THRESHOLD', 80.0))  # in percent
        self.memory_threshold = float(os.getenv('MEMORY_USAGE_THRESHOLD', 80.0))  # in percent
        self.disk_threshold = float(os.getenv('DISK_USAGE_THRESHOLD', 90.0))  # in percent

        logger.info(f"Health Thresholds - CPU: {self.cpu_threshold}%, "
                    f"Memory: {self.memory_threshold}%, Disk: {self.disk_threshold}%")

    def monitor(self):
        """
        Starts the continuous monitoring process.
        """
        logger.info("Starting system health monitoring...")
        try:
            while True:
                self.perform_health_check()
                time.sleep(self.check_interval)
        except KeyboardInterrupt:
            logger.info("System health monitoring stopped by user.")
        except Exception as e:
            logger.exception(f"Unexpected error in monitoring loop: {e}")

    def perform_health_check(self):
        """
        Performs a single health check, logging metrics, detecting anomalies,
        and sending alerts if necessary.
        """
        try:
            metrics = self.collect_metrics()
            self.log_metrics(metrics)
            anomalies = self.detect_anomalies(metrics)
            if anomalies:
                self.handle_anomalies(anomalies)
        except Exception as e:
            logger.error(f"Error during health check: {e}")

    def collect_metrics(self) -> Dict[str, Any]:
        """
        Collects current system performance metrics.

        :return: Dictionary containing CPU, Memory, and Disk usage.
        """
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        disk = psutil.disk_usage('/')
        disk_usage = disk.percent

        metrics = {
            'timestamp': datetime.utcnow().isoformat(),
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'disk_usage': disk_usage
        }

        logger.info(f"Collected Metrics: {metrics}")
        return metrics

    def log_metrics(self, metrics: Dict[str, Any]) -> bool:
        """
        Logs the collected metrics to the data storage.

        :param metrics: Dictionary containing system metrics.
        :return: True if logging is successful, False otherwise.
        """
        try:
            self.data_storage.save_data(table='system_health_metrics', data=metrics)
            logger.info(f"System metrics logged successfully: {metrics}")
            return True
        except Exception as e:
            logger.error(f"Failed to log system metrics: {metrics}. Error: {e}")
            return False

    def detect_anomalies(self, metrics: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """
        Detects anomalies by comparing current metrics against thresholds.

        :param metrics: Dictionary containing system metrics.
        :return: Dictionary of anomalies detected or None if no anomalies.
        """
        anomalies = {}
        if metrics['cpu_usage'] > self.cpu_threshold:
            anomalies['cpu_usage'] = metrics['cpu_usage']
        if metrics['memory_usage'] > self.memory_threshold:
            anomalies['memory_usage'] = metrics['memory_usage']
        if metrics['disk_usage'] > self.disk_threshold:
            anomalies['disk_usage'] = metrics['disk_usage']

        if anomalies:
            logger.warning(f"Anomalies detected: {anomalies}")
            return anomalies
        else:
            logger.info("No anomalies detected.")
            return None

    def handle_anomalies(self, anomalies: Dict[str, float]) -> bool:
        """
        Handles detected anomalies by sending alerts and logging the event.

        :param anomalies: Dictionary of anomalies detected.
        :return: True if handling is successful, False otherwise.
        """
        try:
            # Prepare alert message
            alert_message = "System Health Anomalies Detected:\n"
            for metric, value in anomalies.items():
                alert_message += f"- {metric.replace('_', ' ').title()}: {value}% exceeds threshold.\n"

            # Send notification
            recipients = os.getenv('ALERT_RECIPIENTS', '').split(',')
            if recipients:
                self.notification_manager.notify(
                    channel='email',
                    subject='Hermod System Health Alert',
                    message=alert_message,
                    recipients=recipients
                )
                logger.info(f"Sent alert notification to: {recipients}")
            else:
                logger.warning("No ALERT_RECIPIENTS configured for notifications.")

            # Log the anomaly event
            anomaly_event = {
                'timestamp': datetime.utcnow().isoformat(),
                'anomalies': anomalies
            }
            self.data_storage.save_data(table='system_anomalies', data=anomaly_event)
            logger.info(f"Anomaly event logged: {anomaly_event}")

            return True
        except Exception as e:
            logger.error(f"Failed to handle anomalies: {anomalies}. Error: {e}")
            return False
