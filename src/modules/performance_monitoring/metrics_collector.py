# src/modules/performance_monitoring/metrics_collector.py

import os
import logging
import time
import psutil
import threading
import json
from typing import Dict, Any, List
from datetime import datetime

from flask import Flask

from modules.ui_ux.realtime_updater import RealTimeUpdater
from src.utils.configuration_manager import ConfigurationManager
from src.modules.notifications.notification_manager import NotificationManager

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('logs/metrics_collector.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class MetricsCollector:
    """
    Performance Metrics Collection
    Collects metrics related to system and application performance, such as latency,
    throughput, and resource utilization. Provides data for performance analysis.
    """

    def __init__(self):
        """
        Initializes the MetricsCollector with necessary configurations.
        """
        try:
            self.config_manager = ConfigurationManager()
            self.notification_manager = NotificationManager()
            self.load_collector_config()
            self.collecting = False
            self.metrics_data: List[Dict[str, Any]] = []
            # Initialize Flask app and RealTimeUpdater
            self.app = Flask(__name__)
            self.realtime_updater = RealTimeUpdater(self.app)
            logger.info("MetricsCollector initialized successfully.")
        except Exception as e:
            logger.exception(f"Failed to initialize MetricsCollector: {e}")
            raise e

    def load_collector_config(self):
        """
        Loads collector configurations from the configuration manager or environment variables.
        """
        logger.info("Loading collector configurations.")
        try:
            self.collector_config = {
                'metrics_collection_interval': int(self.config_manager.get('METRICS_COLLECTION_INTERVAL', 5)),
                'metrics_data_file': self.config_manager.get('METRICS_DATA_FILE', 'data/metrics_data.json'),
                'notification_recipients': self.config_manager.get('NOTIFICATION_RECIPIENTS', '').split(','),
            }
            logger.info(f"Collector configurations loaded: {self.collector_config}")
        except Exception as e:
            logger.error(f"Failed to load collector configurations: {e}")
            raise e

    def start_collection(self):
        """
        Starts the metrics collection process in a separate thread.
        """
        logger.info("Starting metrics collection.")
        try:
            self.collection_thread = threading.Thread(target=self.collect_metrics)
            self.collection_thread.start()
        except Exception as e:
            logger.error(f"Failed to start metrics collection: {e}")
            self.send_notification(
                subject="Metrics Collection Failed to Start",
                message=f"Metrics collection failed to start with the following error:\n\n{e}"
            )
            raise e

    def stop_collection(self):
        """
        Stops the metrics collection process.
        """
        logger.info("Stopping metrics collection.")
        try:
            self.stop_event.set()
            self.collection_thread.join()
            self.save_metrics_data()
            logger.info("Metrics collection stopped successfully.")
        except Exception as e:
            logger.error(f"Failed to stop metrics collection: {e}")
            raise e

    def collect_metrics(self):
        """
        Collects metrics at regular intervals.
        """
        logger.info("Starting metrics collection.")
        try:
            while self.collecting:
                system_metrics = self.get_system_metrics()
                application_metrics = self.get_application_metrics()
                timestamp = datetime.utcnow().isoformat()
                metrics_record = {
                    'timestamp': timestamp,
                    'system_metrics': system_metrics,
                    'application_metrics': application_metrics
                }
                self.metrics_data.append(metrics_record)
                logger.info(f"Collected metrics at {timestamp}.")
                # Emit real-time update
                self.realtime_updater.emit_update('metrics_update', metrics_record)
                time.sleep(self.collection_interval)
        except Exception as e:
            logger.error(f"Metrics collection failed: {e}")
            self.send_notification(
                subject="Metrics Collection Failed",
                message=f"Metrics collection failed with the following error:\n\n{e}"
            )
            raise e

    def get_system_metrics(self) -> Dict[str, Any]:
        """
        Retrieves system performance metrics such as CPU usage, memory usage, and disk I/O.

        :return: Dictionary of system metrics.
        """
        try:
            cpu_usage = psutil.cpu_percent(interval=None)
            memory_info = psutil.virtual_memory()
            disk_io = psutil.disk_io_counters()
            network_io = psutil.net_io_counters()

            system_metrics = {
                'cpu_usage_percent': cpu_usage,
                'memory_used_mb': memory_info.used / (1024 * 1024),
                'memory_available_mb': memory_info.available / (1024 * 1024),
                'memory_usage_percent': memory_info.percent,
                'disk_read_mb': disk_io.read_bytes / (1024 * 1024),
                'disk_write_mb': disk_io.write_bytes / (1024 * 1024),
                'network_sent_mb': network_io.bytes_sent / (1024 * 1024),
                'network_received_mb': network_io.bytes_recv / (1024 * 1024)
            }
            return system_metrics
        except Exception as e:
            logger.error(f"Failed to retrieve system metrics: {e}")
            raise e

    def get_application_metrics(self) -> Dict[str, Any]:
        """
        Retrieves application-specific performance metrics such as latency and throughput.

        :return: Dictionary of application metrics.
        """
        try:
            # Placeholder implementation
            # Replace with actual logic to collect application metrics
            application_metrics = {
                'latency_ms': self.simulate_latency_measurement(),
                'throughput_rps': self.simulate_throughput_measurement()
            }
            return application_metrics
        except Exception as e:
            logger.error(f"Failed to retrieve application metrics: {e}")
            raise e

    def simulate_latency_measurement(self) -> float:
        """
        Simulates measuring application latency.

        :return: Simulated latency in milliseconds.
        """
        import random
        return random.uniform(50, 200)

    def simulate_throughput_measurement(self) -> float:
        """
        Simulates measuring application throughput.

        :return: Simulated throughput in requests per second.
        """
        import random
        return random.uniform(100, 500)

    def save_metrics_data(self):
        """
        Saves the collected metrics data to a file.
        """
        logger.info("Saving metrics data.")
        try:
            metrics_data_file = self.collector_config['metrics_data_file']
            os.makedirs(os.path.dirname(metrics_data_file), exist_ok=True)
            with open(metrics_data_file, 'w') as f:
                json.dump(self.metrics_data, f, indent=4)
            logger.info(f"Metrics data saved to '{metrics_data_file}'.")
        except Exception as e:
            logger.error(f"Failed to save metrics data: {e}")
            raise e

    def send_notification(self, subject: str, message: str):
        """
        Sends a notification to the configured recipients.

        :param subject: Subject of the notification.
        :param message: Body of the notification.
        """
        try:
            recipients = self.collector_config['notification_recipients']
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
    Demonstrates example usage of the MetricsCollector class.
    """
    try:
        # Initialize MetricsCollector
        collector = MetricsCollector()

        # Start metrics collection
        collector.start_collection()

        # Collect metrics for a certain duration
        collection_duration = 60  # seconds
        time.sleep(collection_duration)

        # Stop metrics collection
        collector.stop_collection()

        # Access the collected metrics data
        print("Collected Metrics Data:")
        print(json.dumps(collector.metrics_data, indent=4))

    except Exception as e:
        logger.exception(f"Error in example usage: {e}")

# --------------------- Main Execution --------------------- #

if __name__ == "__main__":
    # Run the metrics collector example
    example_usage()
