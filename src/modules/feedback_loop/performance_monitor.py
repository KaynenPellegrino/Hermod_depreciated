#!/usr/bin/env python3
"""
performance_monitor.py

Function: System Performance Monitoring
Purpose: Monitors system performance metrics like CPU usage, memory consumption, disk I/O, network usage,
         response times, and error rates. It provides data for analysis to maintain optimal system performance.
"""

import os
import sys
import yaml
import logging
from datetime import datetime
import time
import psutil
import requests
import pandas as pd
from sqlalchemy import create_engine


# ----------------------------
# Configuration and Logging
# ----------------------------

def load_config(config_path='config.yaml'):
    """
    Load configuration from a YAML file.
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logging.error(f"Failed to load configuration file: {e}")
        sys.exit(1)


def setup_logging(log_dir='logs'):
    """
    Setup logging configuration.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'performance_monitor_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


# ----------------------------
# Performance Monitor Class
# ----------------------------

class PerformanceMonitor:
    """
    Monitors system performance metrics and application performance.
    """

    def __init__(self, config):
        self.config = config
        self.monitor_interval = self.config.get('monitor_interval', 60)  # in seconds
        self.output_dir = self.config.get('output_dir', 'performance_data')
        os.makedirs(self.output_dir, exist_ok=True)

        # Database setup
        self.db_config = self.config.get('database')
        if self.db_config:
            self.engine = self.create_db_engine()
        else:
            self.engine = None

        # Application endpoints to monitor
        self.endpoints = self.config.get('endpoints', [])

        # Initialize data storage
        self.metrics = []
        self.app_metrics = []

    def create_db_engine(self):
        """
        Create a SQLAlchemy engine based on configuration.
        """
        try:
            db_url = f"{self.db_config['dialect']}://{self.db_config['username']}:{self.db_config['password']}@" \
                     f"{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
            engine = create_engine(db_url)
            logging.info("Database engine created successfully.")
            return engine
        except Exception as e:
            logging.error(f"Failed to create database engine: {e}")
            sys.exit(1)

    def collect_system_metrics(self):
        """
        Collect system performance metrics using psutil.
        """
        try:
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            network = psutil.net_io_counters()
            bytes_sent = network.bytes_sent
            bytes_recv = network.bytes_recv
            timestamp = datetime.now()

            metric = {
                'timestamp': timestamp,
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'disk_percent': disk_percent,
                'bytes_sent': bytes_sent,
                'bytes_recv': bytes_recv
            }
            self.metrics.append(metric)
            logging.info(f"Collected system metrics: {metric}")
        except Exception as e:
            logging.error(f"Error collecting system metrics: {e}")

    def collect_application_metrics(self):
        """
        Collect application performance metrics by pinging endpoints.
        """
        try:
            for endpoint in self.endpoints:
                url = endpoint.get('url')
                method = endpoint.get('method', 'GET').upper()
                headers = endpoint.get('headers', {})
                payload = endpoint.get('payload', {})
                timeout = endpoint.get('timeout', 5)
                expected_status = endpoint.get('expected_status', 200)

                timestamp = datetime.now()
                try:
                    response = requests.request(method, url, headers=headers, json=payload, timeout=timeout)
                    response_time = response.elapsed.total_seconds()
                    status_code = response.status_code
                    success = status_code == expected_status
                except requests.exceptions.RequestException as e:
                    response_time = None
                    status_code = None
                    success = False
                    logging.error(f"Error pinging {url}: {e}")

                app_metric = {
                    'timestamp': timestamp,
                    'endpoint': url,
                    'method': method,
                    'response_time_sec': response_time,
                    'status_code': status_code,
                    'success': success
                }
                self.app_metrics.append(app_metric)
                logging.info(f"Collected application metrics: {app_metric}")
        except Exception as e:
            logging.error(f"Error collecting application metrics: {e}")

    def save_metrics(self):
        """
        Save collected metrics to CSV files and/or database.
        """
        try:
            if self.metrics:
                df_metrics = pd.DataFrame(self.metrics)
                metrics_file = os.path.join(self.output_dir,
                                            f'system_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
                df_metrics.to_csv(metrics_file, index=False)
                logging.info(f"Saved system metrics to {metrics_file}.")

                if self.engine:
                    df_metrics.to_sql('system_metrics', con=self.engine, if_exists='append', index=False)
                    logging.info("Inserted system metrics into the database.")

                # Clear the metrics list after saving
                self.metrics = []

            if self.app_metrics:
                df_app_metrics = pd.DataFrame(self.app_metrics)
                app_metrics_file = os.path.join(self.output_dir,
                                                f'application_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
                df_app_metrics.to_csv(app_metrics_file, index=False)
                logging.info(f"Saved application metrics to {app_metrics_file}.")

                if self.engine:
                    df_app_metrics.to_sql('application_metrics', con=self.engine, if_exists='append', index=False)
                    logging.info("Inserted application metrics into the database.")

                # Clear the app_metrics list after saving
                self.app_metrics = []
        except Exception as e:
            logging.error(f"Error saving metrics: {e}")

    def generate_visualizations(self):
        """
        Generate visualizations for the collected metrics.
        """
        try:
            # Example: Generate CPU usage plot for the last N data points
            cpu_data = [m['cpu_percent'] for m in self.metrics]
            memory_data = [m['memory_percent'] for m in self.metrics]
            disk_data = [m['disk_percent'] for m in self.metrics]
            timestamps = [m['timestamp'] for m in self.metrics]

            if cpu_data:
                df = pd.DataFrame({
                    'timestamp': timestamps,
                    'cpu_percent': cpu_data,
                    'memory_percent': memory_data,
                    'disk_percent': disk_data
                })
                plt.figure(figsize=(10, 6))
                plt.plot(df['timestamp'], df['cpu_percent'], label='CPU Usage (%)')
                plt.plot(df['timestamp'], df['memory_percent'], label='Memory Usage (%)')
                plt.plot(df['timestamp'], df['disk_percent'], label='Disk Usage (%)')
                plt.xlabel('Timestamp')
                plt.ylabel('Usage (%)')
                plt.title('System Performance Metrics Over Time')
                plt.legend()
                plt.tight_layout()

                visualization_file = os.path.join(self.output_dir,
                                                  f'system_performance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
                plt.savefig(visualization_file)
                plt.close()
                logging.info(f"Saved system performance visualization to {visualization_file}.")
        except Exception as e:
            logging.error(f"Error generating visualizations: {e}")

    def run_monitoring(self):
        """
        Execute the monitoring process: collect metrics, save them, and generate visualizations.
        """
        logging.info("Starting system performance monitoring.")
        while True:
            self.collect_system_metrics()
            self.collect_application_metrics()
            self.save_metrics()
            self.generate_visualizations()
            logging.info(f"Sleeping for {self.monitor_interval} seconds.")
            time.sleep(self.monitor_interval)


# ----------------------------
# Main Function
# ----------------------------

def main():
    # Load configuration
    config = load_config()

    # Setup logging
    setup_logging(config.get('log_dir', 'logs'))

    logging.info("Initializing Performance Monitor.")

    # Initialize PerformanceMonitor
    monitor = PerformanceMonitor(config)

    # Start monitoring
    try:
        monitor.run_monitoring()
    except KeyboardInterrupt:
        logging.info("Performance monitoring stopped by user.")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
