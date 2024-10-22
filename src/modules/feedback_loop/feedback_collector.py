#!/usr/bin/env python3
"""
feedback_collector.py

Function: Feedback Data Collection
Purpose: Collects feedback from various sources, such as user interactions, logs, and performance metrics.
         It serves as an aggregator for data that feeds into the feedback loop.
"""

import os
import sys
import yaml
import logging
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


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
    log_file = os.path.join(log_dir, f'feedback_collector_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


# ----------------------------
# Data Collection Handlers
# ----------------------------

class LogFileHandler(FileSystemEventHandler):
    """
    Handler for monitoring log file changes.
    """

    def __init__(self, log_file_path, collector):
        super().__init__()
        self.log_file_path = log_file_path
        self.collector = collector

    def on_modified(self, event):
        if event.src_path == self.log_file_path:
            logging.info(f"Detected modification in log file: {self.log_file_path}")
            self.collector.collect_log_data(self.log_file_path)


class FeedbackCollector:
    """
    Collects feedback data from various sources.
    """

    def __init__(self, config):
        self.config = config
        self.engine = self.create_db_engine()
        self.data_dir = self.config.get('data_dir', 'data')
        os.makedirs(self.data_dir, exist_ok=True)

    def create_db_engine(self):
        """
        Create a SQLAlchemy engine based on configuration.
        """
        db_config = self.config.get('database')
        if not db_config:
            logging.error("Database configuration not found in config.yaml.")
            sys.exit(1)

        db_url = f"{db_config['dialect']}://{db_config['username']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        try:
            engine = create_engine(db_url)
            logging.info("Database engine created successfully.")
            return engine
        except Exception as e:
            logging.error(f"Failed to create database engine: {e}")
            sys.exit(1)

    def collect_user_feedback(self):
        """
        Collect user feedback data from the database.
        """
        try:
            feedback_table = self.config.get('feedback_table')
            query = f"SELECT * FROM {feedback_table};"
            df = pd.read_sql(query, self.engine)
            feedback_file = os.path.join(self.data_dir, f'user_feedback_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
            df.to_csv(feedback_file, index=False)
            logging.info(f"Collected user feedback data and saved to {feedback_file}.")
        except Exception as e:
            logging.error(f"Error collecting user feedback: {e}")

    def collect_performance_metrics(self):
        """
        Collect system performance metrics from the database.
        """
        try:
            perf_table = self.config.get('performance_table')
            query = f"SELECT * FROM {perf_table};"
            df = pd.read_sql(query, self.engine)
            perf_file = os.path.join(self.data_dir,
                                     f'performance_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
            df.to_csv(perf_file, index=False)
            logging.info(f"Collected performance metrics and saved to {perf_file}.")
        except Exception as e:
            logging.error(f"Error collecting performance metrics: {e}")

    def collect_log_data(self, log_file_path):
        """
        Collect feedback from log files.
        """
        try:
            df = pd.read_csv(log_file_path, sep='|', header=None, names=['timestamp', 'level', 'message'])
            log_file = os.path.join(self.data_dir, f'logs_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
            df.to_csv(log_file, index=False)
            logging.info(f"Collected log data and saved to {log_file}.")
        except Exception as e:
            logging.error(f"Error collecting log data: {e}")

    def aggregate_data(self):
        """
        Aggregate all collected data into a unified dataset.
        """
        try:
            feedback_files = [f for f in os.listdir(self.data_dir) if f.startswith('user_feedback')]
            perf_files = [f for f in os.listdir(self.data_dir) if f.startswith('performance_metrics')]
            log_files = [f for f in os.listdir(self.data_dir) if f.startswith('logs')]

            df_feedback = pd.concat([pd.read_csv(os.path.join(self.data_dir, f)) for f in feedback_files],
                                    ignore_index=True) if feedback_files else pd.DataFrame()
            df_perf = pd.concat([pd.read_csv(os.path.join(self.data_dir, f)) for f in perf_files],
                                ignore_index=True) if perf_files else pd.DataFrame()
            df_logs = pd.concat([pd.read_csv(os.path.join(self.data_dir, f)) for f in log_files],
                                ignore_index=True) if log_files else pd.DataFrame()

            # Example: Merge datasets on timestamp or relevant keys
            # This needs to be customized based on actual data schema
            # For demonstration, we'll simply concatenate them vertically
            aggregated_df = pd.concat([df_feedback, df_perf, df_logs], axis=0, ignore_index=True)

            aggregated_file = os.path.join(self.data_dir,
                                           f'aggregated_feedback_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
            aggregated_df.to_csv(aggregated_file, index=False)
            logging.info(f"Aggregated all data and saved to {aggregated_file}.")
        except Exception as e:
            logging.error(f"Error aggregating data: {e}")

    def start_log_monitoring(self):
        """
        Start monitoring log files for real-time data collection.
        """
        try:
            log_file_path = self.config.get('log_file_path')
            if not log_file_path:
                logging.warning("Log file path not specified in config.yaml. Skipping log monitoring.")
                return

            event_handler = LogFileHandler(log_file_path, self)
            observer = Observer()
            observer.schedule(event_handler, path=os.path.dirname(log_file_path) or '.', recursive=False)
            observer.start()
            logging.info(f"Started monitoring log file: {log_file_path}")
            return observer
        except Exception as e:
            logging.error(f"Error setting up log monitoring: {e}")

    def run_collection(self):
        """
        Execute the data collection process.
        """
        self.collect_user_feedback()
        self.collect_performance_metrics()
        self.aggregate_data()


# ----------------------------
# Main Function
# ----------------------------

def main():
    # Load configuration
    config = load_config()

    # Setup logging
    setup_logging(config.get('log_dir', 'logs'))

    logging.info("Starting Feedback Collector.")

    # Initialize FeedbackCollector
    collector = FeedbackCollector(config)

    # Start log file monitoring
    observer = collector.start_log_monitoring()

    # Run data collection
    collector.run_collection()

    # If log monitoring is active, keep the script running
    if observer:
        try:
            while True:
                pass
        except KeyboardInterrupt:
            observer.stop()
            logging.info("Stopping log file monitoring.")

        observer.join()

    logging.info("Feedback Collector completed successfully.")


if __name__ == "__main__":
    main()
