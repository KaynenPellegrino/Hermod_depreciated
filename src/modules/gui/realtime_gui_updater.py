#!/usr/bin/env python3
"""
realtime_gui_updater.py

Function: Real-Time GUI Updates
Purpose: Handles live updates of the GUI based on real-time system changes. Listens for system events
         such as performance changes or alerts and updates the user interface dynamically to reflect new data.
"""

import sys
import os
import yaml
import logging
from datetime import datetime

from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QApplication, QMessageBox, QVBoxLayout, QLabel, QWidget, QTableWidgetItem, QTableWidget, \
    QMainWindow
from PyQt5.QtCore import QObject, QThread, pyqtSignal, QTimer, Qt
import pandas as pd
import requests
from sqlalchemy import create_engine
from dotenv import load_dotenv
import psutil

from gui_interface import GUIInterface
from style_manager import StyleManager

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
    log_file = os.path.join(log_dir, f'realtime_gui_updater_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


# Initialize logging
setup_logging()


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

    def fetch_latest_metrics(self):
        """
        Fetch the latest system metrics from the database.
        """
        query = """
            SELECT *
            FROM system_metrics
            ORDER BY timestamp DESC
            LIMIT 1
        """
        try:
            df = pd.read_sql_query(query, self.engine)
            logging.info("Fetched latest system metrics successfully.")
            return df
        except Exception as e:
            logging.error(f"Error fetching system metrics: {e}")
            return pd.DataFrame()

    def fetch_latest_alerts(self):
        """
        Fetch the latest alerts from the database.
        """
        query = """
            SELECT *
            FROM alerts
            ORDER BY timestamp DESC
            LIMIT 10
        """
        try:
            df = pd.read_sql_query(query, self.engine)
            logging.info("Fetched latest alerts successfully.")
            return df
        except Exception as e:
            logging.error(f"Error fetching alerts: {e}")
            return pd.DataFrame()


# ----------------------------
# Backend API Interaction
# ----------------------------

class BackendAPI:
    """
    Handles interactions with backend APIs.
    """

    def __init__(self, api_config):
        self.api_base_url = api_config.get('base_url', 'http://localhost:8000')
        self.session = requests.Session()
        # Add authentication headers if needed
        self.api_key = api_config.get('api_key')
        if self.api_key:
            self.session.headers.update({'Authorization': f'Bearer {self.api_key}'})

    def fetch_alerts(self, params=None):
        """
        Fetch alerts from the backend API.
        """
        try:
            response = self.session.get(f"{self.api_base_url}/alerts", params=params)
            response.raise_for_status()
            alerts = response.json()
            logging.info("Fetched alerts successfully from backend API.")
            return alerts
        except Exception as e:
            logging.error(f"Error fetching alerts from backend API: {e}")
            return []

    def fetch_system_metrics_api(self, params=None):
        """
        Fetch system metrics from the backend API.
        """
        try:
            response = self.session.get(f"{self.api_base_url}/metrics/system", params=params)
            response.raise_for_status()
            metrics = response.json()
            logging.info("Fetched system metrics successfully from backend API.")
            return metrics
        except Exception as e:
            logging.error(f"Error fetching system metrics from backend API: {e}")
            return {}

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
# Signals for Inter-Interface Communication
# ----------------------------

class Communicate(QObject):
    """
    Custom signals for communication between RealTimeGUIUpdater and other modules.
    """
    update_system_metrics = pyqtSignal(pd.DataFrame)
    update_alerts = pyqtSignal(pd.DataFrame)
    show_notification = pyqtSignal(str, str, str)  # title, message, type


# ----------------------------
# RealTimeGUIUpdater Class
# ----------------------------

class RealTimeGUIUpdater(QObject):
    """
    Handles real-time updates of the GUI based on system changes and alerts.
    """

    def __init__(self, db, backend_api, communicate, main_window, refresh_interval=1000):
        """
        Initialize the RealTimeGUIUpdater with references to the database, backend API,
        communication signals, main window, and refresh interval.

        :param db: Database instance for data operations.
        :param backend_api: BackendAPI instance for API interactions.
        :param communicate: Communicate instance for signal handling.
        :param main_window: Reference to the main GUI window.
        :param refresh_interval: Interval in milliseconds for refreshing data.
        """
        super().__init__()
        self.db = db
        self.backend_api = backend_api
        self.communicate = communicate
        self.main_window = main_window
        self.refresh_interval = refresh_interval  # in milliseconds
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_gui)
        self.timer.start(self.refresh_interval)
        logging.info("RealTimeGUIUpdater initialized and timer started.")

    def update_gui(self):
        """
        Fetch latest data and emit signals to update the GUI.
        """
        logging.info("RealTimeGUIUpdater: Fetching latest system metrics and alerts.")
        system_metrics_df = self.db.fetch_latest_metrics()
        alerts_df = self.db.fetch_latest_alerts()

        if not system_metrics_df.empty:
            self.communicate.update_system_metrics.emit(system_metrics_df)

        if not alerts_df.empty:
            self.communicate.update_alerts.emit(alerts_df)

        # Optionally, fetch data from backend API
        # alerts_api = self.backend_api.fetch_alerts()
        # metrics_api = self.backend_api.fetch_system_metrics_api()
        # Process and emit as needed

    def stop(self):
        """
        Stop the timer to cease real-time updates.
        """
        self.timer.stop()
        logging.info("RealTimeGUIUpdater: Timer stopped.")


# ----------------------------
# GUI Components Update Slots
# ----------------------------

class MainWindow(QMainWindow):
    """
    Main application window managing different sections and additional windows.
    """

    def __init__(self, db, backend_api):
        super().__init__()
        self.db = db
        self.backend_api = backend_api
        self.communicate = Communicate()
        self.init_ui()
        self.window_manager = WindowManager(self, self.communicate)
        self.gui_interface = GUIInterface(self.db, self.backend_api, self.communicate, self)  # Initialize GUIInterface
        self.style_manager = StyleManager(QApplication.instance(), self.load_config(), self.communicate,
                                          self)  # Initialize StyleManager
        self.real_time_updater = RealTimeGUIUpdater(self.db, self.backend_api, self.communicate,
                                                    self)  # Initialize RealTimeGUIUpdater
        self.connect_signals()
        self.event_handler.connect_events()  # Connect events

    def load_config(self):
        """
        Load configuration settings.
        """
        config = load_config('config.yaml')
        return config

    def init_ui(self):
        """
        Initialize the GUI components.
        """
        self.setWindowTitle("Hermod AI Assistant")
        self.setGeometry(100, 100, 1200, 800)
        # Initialize widgets, layouts, etc.
        # For brevity, not fully implemented here
        pass

    def connect_signals(self):
        """
        Connect communication signals to GUI update slots.
        """
        self.communicate.update_system_metrics.connect(self.update_system_metrics_display)
        self.communicate.update_alerts.connect(self.update_alerts_display)
        self.communicate.show_notification.connect(self.show_notification)

    def update_system_metrics_display(self, metrics_df):
        """
        Update the system metrics display in the GUI.

        :param metrics_df: DataFrame containing the latest system metrics.
        """
        # Example: Update labels or charts with new metrics
        cpu_percent = metrics_df.iloc[0]['cpu_percent']
        memory_percent = metrics_df.iloc[0]['memory_percent']
        disk_percent = metrics_df.iloc[0]['disk_percent']

        self.dashboard_widget.cpu_label.setText(f"CPU Usage: {cpu_percent}%")
        self.dashboard_widget.memory_label.setText(f"Memory Usage: {memory_percent}%")
        self.dashboard_widget.disk_label.setText(f"Disk Usage: {disk_percent}%")

        # Update charts if any
        self.dashboard_widget.cpu_chart.update_data(cpu_percent)
        self.dashboard_widget.memory_chart.update_data(memory_percent)
        self.dashboard_widget.disk_chart.update_data(disk_percent)

    def update_alerts_display(self, alerts_df):
        """
        Update the alerts display in the GUI.

        :param alerts_df: DataFrame containing the latest alerts.
        """
        # Example: Populate a table widget with alerts
        self.alerts_widget.populate_table(alerts_df)
        # Optionally, display pop-up notifications for critical alerts
        critical_alerts = alerts_df[alerts_df['severity'] == 'Critical']
        for _, alert in critical_alerts.iterrows():
            self.show_notification("Critical Alert", alert['message'], "error")

    def show_notification(self, title, message, notification_type="info"):
        """
        Display a pop-up notification to the user.

        :param title: Title of the notification.
        :param message: Message content of the notification.
        :param notification_type: Type of notification ('info', 'warning', 'error').
        """
        if notification_type == "info":
            QMessageBox.information(self, title, message)
        elif notification_type == "warning":
            QMessageBox.warning(self, title, message)
        elif notification_type == "error":
            QMessageBox.critical(self, title, message)
        else:
            QMessageBox.information(self, title, message)


# ----------------------------
# WindowManager and Other Classes
# ----------------------------

# Placeholder for WindowManager, DashboardWidget, AlertsWidget, etc.
# These should be implemented with the necessary GUI components.

class WindowManager:
    """
    Manages different sections and additional windows within the application.
    """

    def __init__(self, main_window, communicate):
        self.main_window = main_window
        self.communicate = communicate
        # Initialize and manage different windows
        pass


class DashboardWidget(QWidget):
    """
    Dashboard section displaying system metrics.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        # Initialize labels, charts, etc.
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.cpu_label = QLabel("CPU Usage: N/A")
        self.memory_label = QLabel("Memory Usage: N/A")
        self.disk_label = QLabel("Disk Usage: N/A")

        # Initialize charts (placeholder)
        self.cpu_chart = ChartWidget("CPU Usage")
        self.memory_chart = ChartWidget("Memory Usage")
        self.disk_chart = ChartWidget("Disk Usage")

        layout.addWidget(self.cpu_label)
        layout.addWidget(self.cpu_chart)
        layout.addWidget(self.memory_label)
        layout.addWidget(self.memory_chart)
        layout.addWidget(self.disk_label)
        layout.addWidget(self.disk_chart)

        self.setLayout(layout)


class AlertsWidget(QWidget):
    """
    Alerts section displaying system alerts.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        # Initialize table widget to display alerts
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.alerts_table = QTableWidget()
        self.alerts_table.setColumnCount(3)
        self.alerts_table.setHorizontalHeaderLabels(['Timestamp', 'Severity', 'Message'])
        layout.addWidget(self.alerts_table)
        self.setLayout(layout)

    def populate_table(self, alerts_df):
        """
        Populate the alerts table with data from a DataFrame.

        :param alerts_df: DataFrame containing alert data.
        """
        self.alerts_table.setRowCount(len(alerts_df))
        for row_idx, row in alerts_df.iterrows():
            timestamp_item = QTableWidgetItem(str(row['timestamp']))
            severity_item = QTableWidgetItem(row['severity'])
            message_item = QTableWidgetItem(row['message'])

            # Optionally, set text color based on severity
            if row['severity'] == 'Critical':
                severity_item.setForeground(QColor('red'))
            elif row['severity'] == 'Warning':
                severity_item.setForeground(QColor('orange'))
            else:
                severity_item.setForeground(QColor('black'))

            self.alerts_table.setItem(row_idx, 0, timestamp_item)
            self.alerts_table.setItem(row_idx, 1, severity_item)
            self.alerts_table.setItem(row_idx, 2, message_item)


class ChartWidget(QWidget):
    """
    Placeholder widget for displaying charts.
    """

    def __init__(self, title, parent=None):
        super().__init__(parent)
        # Implement chart using PyQt5's QChart or other charting libraries
        self.title = title
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.label = QLabel(self.title)
        self.label.setAlignment(Qt.AlignCenter)
        # Placeholder for chart
        self.chart_view = QLabel("Chart Placeholder")
        self.chart_view.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)
        layout.addWidget(self.chart_view)
        self.setLayout(layout)

    def update_data(self, value):
        """
        Update the chart with new data.

        :param value: New value to display.
        """
        # Implement actual chart updating logic
        self.chart_view.setText(f"Value: {value}")


# ----------------------------
# Main Application Execution
# ----------------------------

def main():
    # Load configuration
    config = load_config('config.yaml')

    # Initialize database
    db_config = config.get('database', {})
    db = Database(db_config)

    # Initialize backend API
    api_config = config.get('backend_api', {})
    backend_api = BackendAPI(api_config)

    # Initialize application
    app = QApplication(sys.argv)

    # Initialize main window
    main_window = MainWindow(db, backend_api)
    main_window.show()

    # Execute application
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
