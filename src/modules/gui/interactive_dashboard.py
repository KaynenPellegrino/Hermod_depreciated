#!/usr/bin/env python3
"""
interactive_dashboard.py

Function: User Dashboard Interface
Purpose: Provides a user-friendly, interactive dashboard where users can view project statistics,
         real-time performance, and system metrics. Integrates with dashboard_builder.py from the
         ui_ux/ module to create a fully visual experience for monitoring and interacting with Hermod.
"""

import sys
import os
import yaml
import logging
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QMessageBox, QTabWidget, QPushButton, \
    QFileDialog
from PyQt5.QtCore import QObject, pyqtSignal, QTimer
from PyQt5.QtGui import QFont
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import psutil

# Import DashboardBuilder from ui_ux module
from modules.ui_ux.dashboard_builder import DashboardBuilder

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
    log_file = os.path.join(log_dir, f'interactive_dashboard_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

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

    def fetch_project_statistics(self):
        """
        Fetch project statistics from the database.
        """
        query = """
            SELECT project_name, status, completion_percentage, start_date, end_date
            FROM projects
        """
        try:
            df = pd.read_sql_query(query, self.engine)
            logging.info("Fetched project statistics successfully.")
            return df
        except Exception as e:
            logging.error(f"Error fetching project statistics: {e}")
            return pd.DataFrame()

    def fetch_system_metrics(self):
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

    def fetch_alerts(self):
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
# Signals for Inter-Interface Communication
# ----------------------------

class Communicate(QObject):
    """
    Custom signals for communication between InteractiveDashboard and other modules.
    """
    update_project_stats = pyqtSignal(pd.DataFrame)
    update_system_metrics = pyqtSignal(pd.DataFrame)
    update_alerts = pyqtSignal(pd.DataFrame)
    show_notification = pyqtSignal(str, str, str)  # title, message, type


# ----------------------------
# InteractiveDashboard Class
# ----------------------------

class InteractiveDashboard(QMainWindow):
    """
    Provides a user-friendly, interactive dashboard where users can view project statistics,
    real-time performance, and system metrics. Integrates with dashboard_builder.py from the
    ui_ux/ module to create a fully visual experience for monitoring and interacting with Hermod.
    """

    def __init__(self, db, communicate, config):
        """
        Initialize the InteractiveDashboard with database reference, communication signals,
        and configuration.

        :param db: Database instance for data operations.
        :param communicate: Communicate instance for signal handling.
        :param config: Configuration dictionary loaded from config.yaml.
        """
        super().__init__()
        self.db = db
        self.communicate = communicate
        self.config = config
        self.init_ui()
        self.connect_signals()
        self.setup_timers()
        logging.info("InteractiveDashboard initialized successfully.")

    def init_ui(self):
        """
        Initialize the GUI components using DashboardBuilder.
        """
        self.setWindowTitle("Hermod AI Assistant - Dashboard")
        self.setGeometry(100, 100, 1200, 800)

        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # Initialize DashboardBuilder
        self.dashboard_builder = DashboardBuilder()

        # Create tabs for different dashboard sections
        self.tabs = QTabWidget()

        # Project Statistics Tab
        self.project_stats_tab = self.dashboard_builder.build_project_statistics_tab()
        self.tabs.addTab(self.project_stats_tab, "Project Statistics")

        # System Performance Tab
        self.system_performance_tab = self.dashboard_builder.build_system_performance_tab()
        self.tabs.addTab(self.system_performance_tab, "System Performance")

        # Alerts Tab
        self.alerts_tab = self.dashboard_builder.build_alerts_tab()
        self.tabs.addTab(self.alerts_tab, "Alerts")

        # Add Export Buttons (Optional Enhancement)
        self.add_export_buttons()

        # Add tabs to the main layout
        layout.addWidget(self.tabs)

    def add_export_buttons(self):
        """
        Add export buttons to each tab for data export functionality.
        """
        # Project Statistics Export Button
        export_project_btn = QPushButton("Export Project Statistics")
        export_project_btn.clicked.connect(lambda: self.export_data('projects', 'project_statistics.csv'))
        self.project_stats_tab.layout().addWidget(export_project_btn)

        # System Performance Export Button
        export_system_btn = QPushButton("Export System Metrics")
        export_system_btn.clicked.connect(lambda: self.export_data('system_metrics', 'system_metrics.csv'))
        self.system_performance_tab.layout().addWidget(export_system_btn)

        # Alerts Export Button
        export_alerts_btn = QPushButton("Export Alerts")
        export_alerts_btn.clicked.connect(lambda: self.export_data('alerts', 'alerts.csv'))
        self.alerts_tab.layout().addWidget(export_alerts_btn)

    def connect_signals(self):
        """
        Connect communication signals to update methods.
        """
        self.communicate.update_project_stats.connect(self.update_project_statistics_display)
        self.communicate.update_system_metrics.connect(self.update_system_performance_display)
        self.communicate.update_alerts.connect(self.update_alerts_display)
        self.communicate.show_notification.connect(self.show_notification)

    def setup_timers(self):
        """
        Setup timers for periodic data fetching.
        """
        self.refresh_interval = self.config.get('interactive_dashboard', {}).get('refresh_interval',
                                                                                 5000)  # Default 5 seconds
        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh_dashboard_data)
        self.timer.start(self.refresh_interval)
        logging.info(f"Dashboard refresh timer started with interval: {self.refresh_interval} ms.")

    def refresh_dashboard_data(self):
        """
        Fetch latest data from the database and emit signals to update the dashboard.
        """
        logging.info("Refreshing dashboard data...")
        project_stats_df = self.db.fetch_project_statistics()
        system_metrics_df = self.db.fetch_system_metrics()
        alerts_df = self.db.fetch_alerts()

        if not project_stats_df.empty:
            self.communicate.update_project_stats.emit(project_stats_df)

        if not system_metrics_df.empty:
            self.communicate.update_system_metrics.emit(system_metrics_df)

        if not alerts_df.empty:
            self.communicate.update_alerts.emit(alerts_df)

    def update_project_statistics_display(self, project_stats_df):
        """
        Update the Project Statistics section in the dashboard.

        :param project_stats_df: DataFrame containing project statistics.
        """
        logging.info("Updating Project Statistics display.")
        self.dashboard_builder.update_project_statistics(project_stats_df)

    def update_system_performance_display(self, system_metrics_df):
        """
        Update the System Performance section in the dashboard.

        :param system_metrics_df: DataFrame containing system metrics.
        """
        logging.info("Updating System Performance display.")
        self.dashboard_builder.update_system_performance(system_metrics_df)

    def update_alerts_display(self, alerts_df):
        """
        Update the Alerts section in the dashboard.

        :param alerts_df: DataFrame containing alerts.
        """
        logging.info("Updating Alerts display.")
        self.dashboard_builder.update_alerts(alerts_df)

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

    def export_data(self, data_type, default_filename):
        """
        Export specified data to a CSV file.

        :param data_type: Type of data to export ('projects', 'system_metrics', 'alerts').
        :param default_filename: Default filename for the exported CSV.
        """
        logging.info(f"Exporting {data_type} data...")
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, f"Save {data_type.capitalize()} Data", default_filename,
                                                   "CSV Files (*.csv)", options=options)
        if file_path:
            try:
                if data_type == 'projects':
                    df = self.db.fetch_project_statistics()
                elif data_type == 'system_metrics':
                    df = self.db.fetch_system_metrics()
                elif data_type == 'alerts':
                    df = self.db.fetch_alerts()
                else:
                    logging.error(f"Unknown data type for export: {data_type}")
                    self.communicate.show_notification.emit("Export Error", f"Unknown data type: {data_type}", "error")
                    return

                if not df.empty:
                    df.to_csv(file_path, index=False)
                    logging.info(f"{data_type.capitalize()} data exported successfully to {file_path}.")
                    self.communicate.show_notification.emit("Export Successful",
                                                            f"{data_type.capitalize()} data exported to {file_path}.",
                                                            "info")
                else:
                    logging.warning(f"No {data_type} data available to export.")
                    self.communicate.show_notification.emit("Export Warning",
                                                            f"No {data_type.capitalize()} data available to export.",
                                                            "warning")
            except Exception as e:
                logging.error(f"Failed to export {data_type} data: {e}")
                self.communicate.show_notification.emit("Export Failed",
                                                        f"Failed to export {data_type.capitalize()} data.", "error")


# ----------------------------
# Main Application Execution
# ----------------------------

def main():
    # Load configuration
    config = load_config('config.yaml')

    # Initialize database
    db_config = config.get('database', {})
    db = Database(db_config)

    # Initialize communication signals
    communicate = Communicate()

    # Initialize application
    app = QApplication(sys.argv)

    # Initialize Interactive Dashboard
    dashboard = InteractiveDashboard(db, communicate, config)
    dashboard.show()

    # Execute application
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
