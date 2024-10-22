#!/usr/bin/env python3
"""
gui_interface.py

Function: GUI and Backend Interface
Purpose: Facilitates communication between the GUI and backend systems. Translates user actions
         from the GUI into backend processes, such as sending data to the feedback loop or
         retrieving real-time performance metrics. Handles data transfers between the user
         interface and core project modules.
"""

import sys
import os
import yaml
import logging
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QMessageBox,
    QTableWidgetItem,
    QFileDialog,
)
from PyQt5.QtCore import Qt, QObject, pyqtSignal
from PyQt5.QtGui import QKeySequence
import pandas as pd
import requests
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()


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
    log_file = os.path.join(log_dir, f'gui_interface_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


# ----------------------------
# Database Setup
# ----------------------------

class Database:
    """
    Database connection handler using SQLAlchemy.
    """

    def __init__(self, db_config):
        self.db_config = db_config
        self.engine = self.create_db_engine()

    def create_db_engine(self):
        """
        Create a SQLAlchemy engine based on configuration.
        """
        try:
            dialect = self.db_config['dialect']
            username = self.db_config['username']
            password = self.db_config['password']
            host = self.db_config['host']
            port = self.db_config['port']
            database = self.db_config['database']
            db_url = f"{dialect}://{username}:{password}@{host}:{port}/{database}"
            engine = create_engine(db_url, pool_pre_ping=True)
            logging.info("Database engine created successfully.")
            return engine
        except Exception as e:
            logging.error(f"Failed to create database engine: {e}")
            sys.exit(1)

    def fetch_data(self, query):
        """
        Execute a raw SQL query and return results as a DataFrame.
        """
        try:
            df = pd.read_sql_query(query, self.engine)
            logging.info(f"Fetched data successfully with query: {query}")
            return df
        except Exception as e:
            logging.error(f"Error fetching data: {e}")
            return pd.DataFrame()

    def insert_data(self, table, data):
        """
        Insert data into a specified table.
        """
        try:
            df = pd.DataFrame([data])
            df.to_sql(table, self.engine, if_exists='append', index=False)
            logging.info(f"Inserted data into {table} successfully.")
            return True
        except Exception as e:
            logging.error(f"Error inserting data into {table}: {e}")
            return False


# ----------------------------
# Backend API Interaction
# ----------------------------

class BackendAPI:
    """
    Handles interactions with backend APIs like real_time_feedback.py.
    """

    def __init__(self, api_config):
        self.api_base_url = api_config.get('base_url', 'http://localhost:8000')
        self.session = requests.Session()
        # Add authentication headers if needed
        self.api_key = api_config.get('api_key')
        if self.api_key:
            self.session.headers.update({'Authorization': f'Bearer {self.api_key}'})

    def submit_feedback(self, feedback_data):
        """
        Submit user feedback to the backend API.
        """
        try:
            response = self.session.post(f"{self.api_base_url}/submit_feedback", json=feedback_data)
            response.raise_for_status()
            logging.info("Feedback submitted successfully.")
            return True
        except Exception as e:
            logging.error(f"Error submitting feedback: {e}")
            return False

    def start_process(self, process_name, params=None):
        """
        Trigger a backend process based on user input.
        """
        try:
            payload = {'process_name': process_name}
            if params:
                payload.update(params)
            response = self.session.post(f"{self.api_base_url}/start_process", json=payload)
            response.raise_for_status()
            logging.info(f"Process '{process_name}' started successfully.")
            return True
        except Exception as e:
            logging.error(f"Error starting process '{process_name}': {e}")
            return False

    def authenticate(self, username, password):
        """
        Authenticate user via backend API.
        """
        try:
            payload = {
                'username': username,
                'password': password
            }
            response = self.session.post(f"{self.api_base_url}/authenticate", json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get('authenticated', False)
        except Exception as e:
            logging.error(f"Authentication error: {e}")
            return False

    def fetch_system_metrics(self, params=None):
        """
        Fetch system metrics from the backend API.
        """
        try:
            response = self.session.get(f"{self.api_base_url}/metrics/system", params=params)
            response.raise_for_status()
            metrics = response.json()
            logging.info("Fetched system metrics successfully.")
            return metrics
        except Exception as e:
            logging.error(f"Error fetching system metrics: {e}")
            return []

    def fetch_application_metrics(self, params=None):
        """
        Fetch application metrics from the backend API.
        """
        try:
            response = self.session.get(f"{self.api_base_url}/metrics/application", params=params)
            response.raise_for_status()
            metrics = response.json()
            logging.info("Fetched application metrics successfully.")
            return metrics
        except Exception as e:
            logging.error(f"Error fetching application metrics: {e}")
            return []


# ----------------------------
# Signals for Inter-Interface Communication
# ----------------------------

class Communicate(QObject):
    """
    Custom signals for communication between GUI Interface and other modules.
    """
    refresh_dashboard = pyqtSignal()
    refresh_feedback = pyqtSignal()
    update_overview = pyqtSignal()
    show_notification = pyqtSignal(str, str, str)  # title, message, type


# ----------------------------
# GUI Interface Class
# ----------------------------

class GUIInterface:
    """
    Facilitates communication between the GUI and backend systems.
    Translates user actions from the GUI into backend processes, sends and retrieves data,
    and ensures data integrity and synchronization between frontend and backend.
    """

    def __init__(self, db, backend_api, communicate, main_window):
        """
        Initialize the GUIInterface with references to the database, backend API,
        communication signals, and the main GUI window.

        :param db: Database instance for data operations.
        :param backend_api: BackendAPI instance for API interactions.
        :param communicate: Communicate instance for signal handling.
        :param main_window: Reference to the main GUI window.
        """
        self.db = db
        self.backend_api = backend_api
        self.communicate = communicate
        self.main_window = main_window
        self.setup_connections()

    def setup_connections(self):
        """
        Connect signals to their respective handler methods.
        """
        # Example: When dashboard needs to refresh
        self.communicate.refresh_dashboard.connect(self.refresh_dashboard)

        # Example: When feedback needs to refresh
        self.communicate.refresh_feedback.connect(self.refresh_feedback)

        # Example: When overview needs to update
        self.communicate.update_overview.connect(self.update_overview)

        # Example: When a notification needs to be shown
        self.communicate.show_notification.connect(self.show_notification)

    def refresh_dashboard(self):
        """
        Refresh the dashboard by fetching the latest system and application metrics.
        """
        logging.info("GUIInterface: Refreshing dashboard.")
        system_metrics = self.backend_api.fetch_system_metrics({'limit': 10})
        application_metrics = self.backend_api.fetch_application_metrics({'limit': 10})

        # Update Dashboard Widget
        self.main_window.dashboard_widget.populate_system_metrics(system_metrics)
        self.main_window.dashboard_widget.populate_application_metrics(application_metrics)

    def refresh_feedback(self):
        """
        Refresh the feedback section by fetching the latest feedback data.
        """
        logging.info("GUIInterface: Refreshing feedback.")
        try:
            response = self.backend_api.session.get(f"{self.backend_api.api_base_url}/processed_feedback",
                                                    params={'limit': 10})
            response.raise_for_status()
            feedback_data = response.json()
            self.main_window.feedback_widget.populate_feedback_table(feedback_data)
            self.main_window.feedback_widget.update_sentiment_chart(feedback_data)
        except Exception as e:
            logging.error(f"GUIInterface: Error refreshing feedback: {e}")
            self.show_notification("Data Fetch Error", "Failed to retrieve feedback data.", "error")

    def update_overview(self):
        """
        Update the project overview section with the latest summaries and metrics.
        """
        logging.info("GUIInterface: Updating project overview.")
        try:
            # Fetch total feedback count
            total_feedback_df = self.db.fetch_data("SELECT COUNT(*) as total FROM processed_feedback")
            total_feedback = total_feedback_df.iloc[0]['total'] if not total_feedback_df.empty else 0
            self.main_window.overview_widget.total_feedback_label.setText(f"Total Feedback: {total_feedback}")

            # Fetch average sentiment score
            avg_sentiment_df = self.db.fetch_data(
                "SELECT AVG(sentiment_score) as average_sentiment FROM processed_feedback")
            avg_sentiment = avg_sentiment_df.iloc[0]['average_sentiment'] if not avg_sentiment_df.empty else 0.0
            self.main_window.overview_widget.average_sentiment_label.setText(
                f"Average Sentiment Score: {avg_sentiment:.2f}")

            # Fetch top topics
            top_topics_df = self.db.fetch_data("""
                SELECT topic, COUNT(*) as count 
                FROM processed_feedback 
                GROUP BY topic 
                ORDER BY count DESC 
                LIMIT 5
            """)
            if not top_topics_df.empty:
                topics = ', '.join(top_topics_df['topic'].tolist())
            else:
                topics = 'N/A'
            self.main_window.overview_widget.top_topics_label.setText(f"Top Topics: {topics}")
        except Exception as e:
            logging.error(f"GUIInterface: Error updating project overview: {e}")
            self.show_notification("Data Fetch Error", "Failed to update project overview.", "error")

    def show_notification(self, title, message, notification_type="info"):
        """
        Display a pop-up notification to the user.

        :param title: Title of the notification.
        :param message: Message content of the notification.
        :param notification_type: Type of notification ('info', 'warning', 'error').
        """
        if notification_type == "info":
            QMessageBox.information(self.main_window, title, message)
        elif notification_type == "warning":
            QMessageBox.warning(self.main_window, title, message)
        elif notification_type == "error":
            QMessageBox.critical(self.main_window, title, message)
        else:
            QMessageBox.information(self.main_window, title, message)

    def submit_feedback(self, user_id, feedback_text):
        """
        Submit user feedback by interacting with the backend API.

        :param user_id: ID of the user submitting feedback.
        :param feedback_text: The feedback text provided by the user.
        :return: Boolean indicating success or failure.
        """
        feedback_data = {
            'user_id': int(user_id),
            'feedback_text': feedback_text
        }
        success = self.backend_api.submit_feedback(feedback_data)
        if success:
            self.show_notification("Success", "Feedback submitted successfully.", "info")
            self.refresh_feedback()
            return True
        else:
            self.show_notification("Submission Error", "Failed to submit feedback. Please try again later.", "error")
            return False

    def export_system_metrics(self, file_path):
        """
        Export system metrics to a specified CSV file.

        :param file_path: Path to save the exported CSV file.
        :return: Boolean indicating success or failure.
        """
        try:
            query = "SELECT * FROM system_metrics ORDER BY timestamp DESC"
            df = self.db.fetch_data(query)
            if not df.empty:
                df.to_csv(file_path, index=False)
                self.show_notification("Export Successful", f"System metrics exported to {file_path}.", "info")
                return True
            else:
                self.show_notification("No Data", "No system metrics available to export.", "warning")
                return False
        except Exception as e:
            logging.error(f"GUIInterface: Error exporting system metrics: {e}")
            self.show_notification("Export Failed", "Failed to export system metrics.", "error")
            return False

    def start_backend_process(self, process_name, params=None):
        """
        Start a backend process based on user input.

        :param process_name: Name of the process to start.
        :param params: Additional parameters for the process.
        :return: Boolean indicating success or failure.
        """
        success = self.backend_api.start_process(process_name, params)
        if success:
            self.show_notification("Process Started", f"Process '{process_name}' started successfully.", "info")
            return True
        else:
            self.show_notification("Process Error", f"Failed to start process '{process_name}'.", "error")
            return False
