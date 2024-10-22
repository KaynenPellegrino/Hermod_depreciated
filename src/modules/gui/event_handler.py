#!/usr/bin/env python3
"""
event_handler.py

Function: User Input Event Handler
Purpose: Processes all user interactions within the GUI. Handles events such as button clicks,
         form submissions, drag-and-drop actions, and keyboard inputs. Interacts with backend
         functionalities to trigger actions based on user inputs, such as starting a process
         or submitting feedback.
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
    QFileDialog, QShortcut,
)
from PyQt5.QtCore import Qt
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
    log_file = os.path.join(log_dir, f'event_handler_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

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


# ----------------------------
# Event Handler Class
# ----------------------------

class EventHandler:
    """
    Handles user input events within the GUI.
    Processes events such as button clicks, form submissions, drag-and-drop actions,
    and keyboard inputs. Interacts with backend functionalities to trigger actions
    based on user inputs.
    """

    def __init__(self, db, backend_api, main_window):
        """
        Initialize the EventHandler with database and backend API references.

        :param db: Database instance for data operations.
        :param backend_api: BackendAPI instance for API interactions.
        :param main_window: Reference to the main GUI window.
        """
        self.db = db
        self.backend_api = backend_api
        self.main_window = main_window  # Reference to the main window for accessing GUI components

    def handle_submit_feedback(self):
        """
        Handle the submission of user feedback.
        """
        user_id = self.main_window.feedback_widget.user_id_input.text().strip()
        feedback_text = self.main_window.feedback_widget.feedback_input.toPlainText().strip()

        if not user_id or not feedback_text:
            QMessageBox.warning(self.main_window, "Input Error", "Please provide both User ID and Feedback.")
            return

        # Validate User ID
        if not user_id.isdigit():
            QMessageBox.warning(self.main_window, "Input Error", "User ID must be a numeric value.")
            return

        feedback_data = {
            'user_id': int(user_id),
            'feedback_text': feedback_text
        }

        # Submit feedback via Backend API
        success = self.backend_api.submit_feedback(feedback_data)

        if success:
            QMessageBox.information(self.main_window, "Success", "Feedback submitted successfully.")
            self.main_window.feedback_widget.user_id_input.clear()
            self.main_window.feedback_widget.feedback_input.clear()
            self.main_window.feedback_widget.refresh_feedback()
        else:
            QMessageBox.critical(self.main_window, "Submission Error",
                                 "Failed to submit feedback. Please try again later.")

    def handle_export_system_metrics(self):
        """
        Handle exporting system metrics to a CSV file.
        """
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self.main_window,
            "Save System Metrics",
            "",
            "CSV Files (*.csv);;All Files (*)",
            options=options
        )
        if file_path:
            try:
                # Fetch all system metrics
                query = "SELECT * FROM system_metrics ORDER BY timestamp DESC"
                df = self.db.fetch_data(query)
                if not df.empty:
                    df.to_csv(file_path, index=False)
                    QMessageBox.information(self.main_window, "Export Successful",
                                            f"System metrics exported to {file_path}.")
                else:
                    QMessageBox.warning(self.main_window, "No Data", "No system metrics available to export.")
            except Exception as e:
                logging.error(f"Error exporting system metrics: {e}")
                QMessageBox.critical(self.main_window, "Export Failed", "Failed to export system metrics.")

    def handle_start_process(self, process_name, params=None):
        """
        Handle starting a backend process based on user input.

        :param process_name: Name of the process to start.
        :param params: Additional parameters for the process.
        """
        success = self.backend_api.start_process(process_name, params)
        if success:
            QMessageBox.information(self.main_window, "Process Started",
                                    f"Process '{process_name}' started successfully.")
        else:
            QMessageBox.critical(self.main_window, "Process Error", f"Failed to start process '{process_name}'.")

    def handle_key_press(self, event):
        """
        Handle keyboard input events.

        :param event: QKeyEvent object containing event details.
        """
        if event.matches(QKeySequence.Save):
            # Example: Handle Ctrl+S to save settings
            self.handle_save_settings()
        elif event.matches(QKeySequence.Open):
            # Example: Handle Ctrl+O to open a file dialog
            self.handle_export_system_metrics()
        else:
            # Pass the event to the base class
            event.ignore()

    def handle_save_settings(self):
        """
        Handle saving application settings.
        """
        # Assuming there's a Settings Dialog that has already been opened and settings saved
        QMessageBox.information(self.main_window, "Settings", "Settings have been saved successfully.")

    def handle_drag_and_drop(self, event, target_widget):
        """
        Handle drag-and-drop events.

        :param event: QDragEnterEvent or QDropEvent object.
        :param target_widget: The widget where the drop is targeted.
        """
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            file_paths = [url.toLocalFile() for url in urls]
            # Process the files as needed
            logging.info(f"Files dropped: {file_paths}")
            QMessageBox.information(self.main_window, "Files Dropped", f"Files received: {', '.join(file_paths)}")
            # Example: Start a process with the dropped files
            self.handle_start_process("process_files", {'files': file_paths})
            event.acceptProposedAction()
        else:
            event.ignore()

    def connect_events(self):
        """
        Connect GUI events to their respective handler methods.
        This method should be called after the GUI components are initialized.
        """
        # Connect feedback submission
        self.main_window.feedback_widget.submit_button.clicked.connect(self.handle_submit_feedback)

        # Connect export system metrics
        self.main_window.dashboard_widget.export_button.clicked.connect(self.handle_export_system_metrics)

        # Example: Connect a keyboard shortcut to export system metrics
        self.main_window.shortcut_export = QShortcut(QKeySequence("Ctrl+O"), self.main_window)
        self.main_window.shortcut_export.activated.connect(self.handle_export_system_metrics)

        # Example: Connect a keyboard shortcut to save settings
        self.main_window.shortcut_save = QShortcut(QKeySequence("Ctrl+S"), self.main_window)
        self.main_window.shortcut_save.activated.connect(self.handle_save_settings)

        # Connect drag-and-drop for a specific widget (e.g., a file upload area)
        self.main_window.upload_widget.setAcceptDrops(True)
        self.main_window.upload_widget.dragEnterEvent = lambda event: self.handle_drag_enter(event,
                                                                                             self.main_window.upload_widget)
        self.main_window.upload_widget.dropEvent = lambda event: self.handle_drag_and_drop(event,
                                                                                           self.main_window.upload_widget)

    def handle_drag_enter(self, event, target_widget):
        """
        Handle the drag enter event to accept or reject the drag.

        :param event: QDragEnterEvent object.
        :param target_widget: The widget where the drag is entering.
        """
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()
