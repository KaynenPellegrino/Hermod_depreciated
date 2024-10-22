#!/usr/bin/env python3
"""
gui_manager.py

Function: Central GUI Manager
Purpose: Manages the initialization and rendering of the entire graphical user interface (GUI).
         Coordinates between different windows, components, and backend interactions.
         Handles the core loop of the interface, ensuring smooth transitions between
         different sections like the dashboard, feedback, and project overview.
"""

import sys
import os
import yaml
import logging
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QMessageBox,
    QTextEdit,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QTabWidget,
    QFormLayout,
    QComboBox,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
import pandas as pd
import requests
from sqlalchemy import create_engine

# Load environment variables from .env if present
from dotenv import load_dotenv

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
    log_file = os.path.join(log_dir, f'gui_manager_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

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

    def get_system_metrics(self, params):
        """
        Fetch system metrics from the backend API.
        """
        try:
            response = self.session.get(f"{self.api_base_url}/metrics/system", params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logging.error(f"Error fetching system metrics: {e}")
            return []

    def get_application_metrics(self, params):
        """
        Fetch application metrics from the backend API.
        """
        try:
            response = self.session.get(f"{self.api_base_url}/metrics/application", params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logging.error(f"Error fetching application metrics: {e}")
            return []

    def get_actions(self, params):
        """
        Fetch actions from the backend API.
        """
        try:
            response = self.session.get(f"{self.api_base_url}/actions", params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logging.error(f"Error fetching actions: {e}")
            return []


# ----------------------------
# GUI Components
# ----------------------------

class DashboardWidget(QWidget):
    """
    Dashboard section displaying system and application performance metrics.
    """

    def __init__(self, db, backend_api, parent=None):
        super().__init__(parent)
        self.db = db
        self.backend_api = backend_api
        self.init_ui()
        self.refresh_data()
        # Set up a timer to refresh data periodically
        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh_data)
        self.timer.start(60000)  # Refresh every 60 seconds

    def init_ui(self):
        layout = QVBoxLayout()

        title = QLabel("Dashboard")
        title.setFont(QFont('Arial', 16))
        layout.addWidget(title)

        # System Metrics Table
        self.system_metrics_table = QTableWidget()
        self.system_metrics_table.setColumnCount(6)
        self.system_metrics_table.setHorizontalHeaderLabels([
            'ID', 'Timestamp', 'CPU %', 'Memory %', 'Disk %', 'Network (Sent/Recv Bytes)'
        ])
        self.system_metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(QLabel("System Metrics"))
        layout.addWidget(self.system_metrics_table)

        # Application Metrics Table
        self.application_metrics_table = QTableWidget()
        self.application_metrics_table.setColumnCount(6)
        self.application_metrics_table.setHorizontalHeaderLabels([
            'ID', 'Timestamp', 'Endpoint', 'Method', 'Response Time (s)', 'Status Code'
        ])
        self.application_metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(QLabel("Application Metrics"))
        layout.addWidget(self.application_metrics_table)

        self.setLayout(layout)

    def refresh_data(self):
        """
        Fetch and display the latest system and application metrics.
        """
        logging.info("Refreshing Dashboard data.")
        # Fetch system metrics
        system_metrics = self.backend_api.get_system_metrics({'limit': 10})
        self.populate_system_metrics(system_metrics)

        # Fetch application metrics
        application_metrics = self.backend_api.get_application_metrics({'limit': 10})
        self.populate_application_metrics(application_metrics)

    def populate_system_metrics(self, data):
        """
        Populate the system metrics table with fetched data.
        """
        self.system_metrics_table.setRowCount(0)
        for record in data:
            row_position = self.system_metrics_table.rowCount()
            self.system_metrics_table.insertRow(row_position)
            self.system_metrics_table.setItem(row_position, 0, QTableWidgetItem(str(record.get('id', ''))))
            self.system_metrics_table.setItem(row_position, 1, QTableWidgetItem(record.get('timestamp', '')))
            self.system_metrics_table.setItem(row_position, 2, QTableWidgetItem(str(record.get('cpu_percent', ''))))
            self.system_metrics_table.setItem(row_position, 3, QTableWidgetItem(str(record.get('memory_percent', ''))))
            self.system_metrics_table.setItem(row_position, 4, QTableWidgetItem(str(record.get('disk_percent', ''))))
            network = f"{record.get('bytes_sent', 0)} / {record.get('bytes_recv', 0)}"
            self.system_metrics_table.setItem(row_position, 5, QTableWidgetItem(network))

    def populate_application_metrics(self, data):
        """
        Populate the application metrics table with fetched data.
        """
        self.application_metrics_table.setRowCount(0)
        for record in data:
            row_position = self.application_metrics_table.rowCount()
            self.application_metrics_table.insertRow(row_position)
            self.application_metrics_table.setItem(row_position, 0, QTableWidgetItem(str(record.get('id', ''))))
            self.application_metrics_table.setItem(row_position, 1, QTableWidgetItem(record.get('timestamp', '')))
            self.application_metrics_table.setItem(row_position, 2, QTableWidgetItem(record.get('endpoint', '')))
            self.application_metrics_table.setItem(row_position, 3, QTableWidgetItem(record.get('method', '')))
            self.application_metrics_table.setItem(row_position, 4,
                                                   QTableWidgetItem(str(record.get('response_time_sec', ''))))
            self.application_metrics_table.setItem(row_position, 5,
                                                   QTableWidgetItem(str(record.get('status_code', ''))))


class FeedbackWidget(QWidget):
    """
    Feedback section allowing users to view and submit feedback.
    """

    def __init__(self, db, backend_api, parent=None):
        super().__init__(parent)
        self.db = db
        self.backend_api = backend_api
        self.init_ui()
        self.refresh_feedback()
        # Set up a timer to refresh feedback periodically
        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh_feedback)
        self.timer.start(60000)  # Refresh every 60 seconds

    def init_ui(self):
        layout = QVBoxLayout()

        title = QLabel("User Feedback")
        title.setFont(QFont('Arial', 16))
        layout.addWidget(title)

        # Feedback Submission Form
        form_layout = QFormLayout()
        self.user_id_input = QLineEdit()
        self.feedback_input = QTextEdit()
        submit_button = QPushButton("Submit Feedback")
        submit_button.clicked.connect(self.submit_feedback)

        form_layout.addRow("User ID:", self.user_id_input)
        form_layout.addRow("Feedback:", self.feedback_input)
        form_layout.addRow(submit_button)

        layout.addLayout(form_layout)

        # Feedback Display Table
        self.feedback_table = QTableWidget()
        self.feedback_table.setColumnCount(4)
        self.feedback_table.setHorizontalHeaderLabels([
            'ID', 'User ID', 'Feedback', 'Timestamp'
        ])
        self.feedback_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(QLabel("Recent Feedback"))
        layout.addWidget(self.feedback_table)

        self.setLayout(layout)

    def submit_feedback(self):
        """
        Submit user feedback to the backend API or database.
        """
        user_id = self.user_id_input.text().strip()
        feedback_text = self.feedback_input.toPlainText().strip()

        if not user_id or not feedback_text:
            QMessageBox.warning(self, "Input Error", "Please provide both User ID and Feedback.")
            return

        # Here, you can choose to send feedback via API or insert directly into the database
        # For demonstration, we'll assume there's an API endpoint to submit feedback
        try:
            payload = {
                'user_id': int(user_id),
                'feedback_text': feedback_text
            }
            response = self.backend_api.session.post(f"{self.backend_api.api_base_url}/submit_feedback", json=payload)
            response.raise_for_status()
            QMessageBox.information(self, "Success", "Feedback submitted successfully.")
            self.user_id_input.clear()
            self.feedback_input.clear()
            self.refresh_feedback()
        except Exception as e:
            logging.error(f"Error submitting feedback: {e}")
            QMessageBox.critical(self, "Submission Error", "Failed to submit feedback. Please try again later.")

    def refresh_feedback(self):
        """
        Fetch and display the latest user feedback.
        """
        logging.info("Refreshing Feedback data.")
        # Fetch processed feedback from the backend API or database
        feedback_data = self.backend_api.session.get(f"{self.backend_api.api_base_url}/processed_feedback",
                                                     params={'limit': 10}).json()
        self.populate_feedback_table(feedback_data)

    def populate_feedback_table(self, data):
        """
        Populate the feedback table with fetched data.
        """
        self.feedback_table.setRowCount(0)
        for record in data:
            row_position = self.feedback_table.rowCount()
            self.feedback_table.insertRow(row_position)
            self.feedback_table.setItem(row_position, 0, QTableWidgetItem(str(record.get('id', ''))))
            self.feedback_table.setItem(row_position, 1, QTableWidgetItem(str(record.get('user_id', ''))))
            self.feedback_table.setItem(row_position, 2, QTableWidgetItem(record.get('feedback_text', '')))
            self.feedback_table.setItem(row_position, 3, QTableWidgetItem(record.get('timestamp', '')))


class ProjectOverviewWidget(QWidget):
    """
    Project Overview section displaying summaries and key metrics.
    """

    def __init__(self, db, backend_api, parent=None):
        super().__init__(parent)
        self.db = db
        self.backend_api = backend_api
        self.init_ui()
        self.refresh_overview()
        # Set up a timer to refresh overview periodically
        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh_overview)
        self.timer.start(60000)  # Refresh every 60 seconds

    def init_ui(self):
        layout = QVBoxLayout()

        title = QLabel("Project Overview")
        title.setFont(QFont('Arial', 16))
        layout.addWidget(title)

        # Summary Labels
        self.total_feedback_label = QLabel("Total Feedback: Loading...")
        self.average_sentiment_label = QLabel("Average Sentiment Score: Loading...")
        self.top_topics_label = QLabel("Top Topics: Loading...")

        layout.addWidget(self.total_feedback_label)
        layout.addWidget(self.average_sentiment_label)
        layout.addWidget(self.top_topics_label)

        self.setLayout(layout)

    def refresh_overview(self):
        """
        Fetch and display project overview metrics.
        """
        logging.info("Refreshing Project Overview data.")
        try:
            # Fetch total feedback count
            total_feedback = self.db.fetch_data("SELECT COUNT(*) as total FROM processed_feedback").iloc[0]['total']
            self.total_feedback_label.setText(f"Total Feedback: {total_feedback}")

            # Fetch average sentiment score
            avg_sentiment = \
            self.db.fetch_data("SELECT AVG(sentiment_score) as average_sentiment FROM processed_feedback").iloc[0][
                'average_sentiment']
            self.average_sentiment_label.setText(f"Average Sentiment Score: {avg_sentiment:.2f}")

            # Fetch top topics
            top_topics = self.db.fetch_data("""
                SELECT topic, COUNT(*) as count 
                FROM processed_feedback 
                GROUP BY topic 
                ORDER BY count DESC 
                LIMIT 5
            """)
            topics = ', '.join(top_topics['topic'].tolist())
            self.top_topics_label.setText(f"Top Topics: {topics}")
        except Exception as e:
            logging.error(f"Error refreshing project overview: {e}")
            self.total_feedback_label.setText("Total Feedback: Error fetching data.")
            self.average_sentiment_label.setText("Average Sentiment Score: Error fetching data.")
            self.top_topics_label.setText("Top Topics: Error fetching data.")


class MainWindow(QMainWindow):
    """
    Main application window managing different sections.
    """

    def __init__(self, db, backend_api):
        super().__init__()
        self.db = db
        self.backend_api = backend_api
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Hermod AI Assistant Dashboard")
        self.setGeometry(100, 100, 1200, 800)

        # Central Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Layouts
        main_layout = QHBoxLayout()
        sidebar_layout = QVBoxLayout()
        content_layout = QVBoxLayout()

        # Sidebar Buttons
        dashboard_button = QPushButton("Dashboard")
        feedback_button = QPushButton("Feedback")
        overview_button = QPushButton("Project Overview")
        exit_button = QPushButton("Exit")

        # Connect buttons to methods
        dashboard_button.clicked.connect(self.show_dashboard)
        feedback_button.clicked.connect(self.show_feedback)
        overview_button.clicked.connect(self.show_overview)
        exit_button.clicked.connect(self.close_application)

        # Add buttons to sidebar
        sidebar_layout.addWidget(dashboard_button)
        sidebar_layout.addWidget(feedback_button)
        sidebar_layout.addWidget(overview_button)
        sidebar_layout.addStretch()
        sidebar_layout.addWidget(exit_button)

        # Stacked Widget to switch between different sections
        self.stacked_widget = QStackedWidget()
        self.dashboard_widget = DashboardWidget(self.db, self.backend_api)
        self.feedback_widget = FeedbackWidget(self.db, self.backend_api)
        self.overview_widget = ProjectOverviewWidget(self.db, self.backend_api)

        self.stacked_widget.addWidget(self.dashboard_widget)
        self.stacked_widget.addWidget(self.feedback_widget)
        self.stacked_widget.addWidget(self.overview_widget)

        content_layout.addWidget(self.stacked_widget)

        # Add sidebar and content to main layout
        main_layout.addLayout(sidebar_layout, 1)
        main_layout.addLayout(content_layout, 4)

        central_widget.setLayout(main_layout)

    def show_dashboard(self):
        """
        Display the Dashboard section.
        """
        logging.info("Navigating to Dashboard.")
        self.stacked_widget.setCurrentWidget(self.dashboard_widget)

    def show_feedback(self):
        """
        Display the Feedback section.
        """
        logging.info("Navigating to Feedback.")
        self.stacked_widget.setCurrentWidget(self.feedback_widget)

    def show_overview(self):
        """
        Display the Project Overview section.
        """
        logging.info("Navigating to Project Overview.")
        self.stacked_widget.setCurrentWidget(self.overview_widget)

    def close_application(self):
        """
        Gracefully close the application.
        """
        reply = QMessageBox.question(
            self, 'Exit Confirmation',
            "Are you sure you want to exit?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            logging.info("Application exited by user.")
            QApplication.instance().quit()


# ----------------------------
# Main Function
# ----------------------------

def main():
    """
    Entry point for the GUI Manager.
    """
    # Load configuration
    config = load_config()

    # Setup logging
    setup_logging(config.get('log_dir', 'logs'))

    logging.info("Initializing GUI Manager.")

    # Initialize Database connection
    db_config = config.get('database')
    if not db_config:
        logging.error("Database configuration not found in config.yaml.")
        sys.exit(1)

    db = Database(db_config)

    # Initialize Backend API interaction
    api_config = config.get('backend_api', {})
    backend_api = BackendAPI(api_config)

    # Initialize and run the application
    app = QApplication(sys.argv)
    main_window = MainWindow(db, backend_api)
    main_window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
