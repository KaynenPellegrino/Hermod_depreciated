# src/modules/ui_ux/dashboard_builder.py

import os
import logging
import json
from typing import Dict, Any, List
from jinja2 import Environment, FileSystemLoader
from flask import Flask, render_template, send_from_directory
from threading import Thread
from src.utils.configuration_manager import ConfigurationManager
from src.modules.notifications.notification_manager import NotificationManager
from src.modules.performance_monitoring.metrics_collector import MetricsCollector
from flask_socketio import SocketIO
from src.modules.ui_ux.realtime_updater import RealTimeUpdater


# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('logs/dashboard_builder.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class DashboardBuilder:
    """
    Dynamic Dashboard Creation
    Generates interactive dashboards for users to visualize data, monitor performance,
    and access tools. Provides a user-friendly interface for engaging with Hermod's features.
    """

    def __init__(self):
        """
        Initializes the DashboardBuilder with necessary configurations.
        """
        try:
            self.config_manager = ConfigurationManager()
            self.notification_manager = NotificationManager()
            self.load_dashboard_config()
            self.app = Flask(__name__, template_folder='templates', static_folder='static')
            self.socketio = SocketIO(self.app, async_mode='eventlet')
            self.realtime_updater = RealTimeUpdater(self.app)
            self.env = Environment(loader=FileSystemLoader('templates'))
            self.dashboard_data: Dict[str, Any] = {}
            self.setup_routes()
            self.server_thread: Thread = None
            logger.info("DashboardBuilder initialized successfully.")
        except Exception as e:
            logger.exception(f"Failed to initialize DashboardBuilder: {e}")
            raise e

    def load_dashboard_config(self):
        """
        Loads dashboard configurations from the configuration manager or environment variables.
        """
        logger.info("Loading dashboard configurations.")
        try:
            self.dashboard_config = {
                'dashboard_host': self.config_manager.get('DASHBOARD_HOST', '0.0.0.0'),
                'dashboard_port': int(self.config_manager.get('DASHBOARD_PORT', 5000)),
                'dashboard_title': self.config_manager.get('DASHBOARD_TITLE', 'Hermod Dashboard'),
                'notification_recipients': self.config_manager.get('NOTIFICATION_RECIPIENTS', '').split(','),
                'alerts_file': self.config_manager.get('ALERTS_FILE', 'data/alerts.json'),
                'available_tools': json.loads(self.config_manager.get('AVAILABLE_TOOLS', '[]')),
            }
            logger.info(f"Dashboard configurations loaded: {self.dashboard_config}")
        except Exception as e:
            logger.error(f"Failed to load dashboard configurations: {e}")
            raise e

    def setup_routes(self):
        """
        Sets up the Flask routes for the dashboard.
        """
        logger.info("Setting up dashboard routes.")

        @self.app.route('/')
        def index():
            """
            Renders the main dashboard page.
            """
            try:
                self.update_dashboard_data()
                return render_template('dashboard.html', data=self.dashboard_data)
            except Exception as e:
                logger.error(f"Failed to render dashboard: {e}")
                return f"Error rendering dashboard: {e}", 500

        @self.app.route('/static/<path:path>')
        def send_static(path):
            """
            Serves static files.
            """
            return send_from_directory('static', path)

    def run_dashboard(self):
        """
        Runs the Flask app with SocketIO for the dashboard.
        """
        logger.info("Starting the dashboard server with real-time updates.")
        try:
            self.server_thread = Thread(target=self.socketio.run, args=(self.app,), kwargs={
                'host': self.dashboard_config['dashboard_host'],
                'port': self.dashboard_config['dashboard_port'],
                'debug': False,
                'use_reloader': False
            })
            self.server_thread.start()
            logger.info(
                f"Dashboard server running at http://{self.dashboard_config['dashboard_host']}:{self.dashboard_config['dashboard_port']}")
        except Exception as e:
            logger.error(f"Failed to start the dashboard server: {e}")
            self.send_notification(
                subject="Dashboard Server Failed to Start",
                message=f"The dashboard server failed to start with the following error:\n\n{e}"
            )
            raise e

    def update_dashboard_data(self):
        """
        Updates the dashboard data with the latest information and emits updates to clients.
        """
        logger.info("Updating dashboard data.")
        try:
            # Gather real data for the dashboard
            self.dashboard_data = {
                'title': self.dashboard_config['dashboard_title'],
                'metrics': self.get_metrics(),
                'alerts': self.get_alerts(),
                'tools': self.get_available_tools(),
            }
            logger.info("Dashboard data updated.")
            # Emit the updated data to connected clients
            self.realtime_updater.emit_update('dashboard_update', self.dashboard_data)
        except Exception as e:
            logger.error(f"Failed to update dashboard data: {e}")
            raise e

    def get_metrics(self) -> Dict[str, Any]:
        """
        Retrieves metrics collected by the MetricsCollector module.

        :return: Dictionary of metrics.
        """
        try:
            # Instantiate MetricsCollector and retrieve collected metrics
            metrics_collector = MetricsCollector()
            metrics_collector.load_collector_config()
            metrics_data_file = metrics_collector.collector_config['metrics_data_file']

            if not os.path.exists(metrics_data_file):
                logger.warning(f"Metrics data file not found at '{metrics_data_file}'.")
                return {}

            with open(metrics_data_file, 'r') as f:
                metrics_data = json.load(f)

            # Use the most recent metrics record
            if metrics_data:
                latest_metrics = metrics_data[-1]
                metrics = {
                    'cpu_usage_percent': latest_metrics['system_metrics']['cpu_usage_percent'],
                    'memory_usage_percent': latest_metrics['system_metrics']['memory_usage_percent'],
                    'disk_read_mb': latest_metrics['system_metrics']['disk_read_mb'],
                    'disk_write_mb': latest_metrics['system_metrics']['disk_write_mb'],
                    'network_sent_mb': latest_metrics['system_metrics']['network_sent_mb'],
                    'network_received_mb': latest_metrics['system_metrics']['network_received_mb'],
                    'latency_ms': latest_metrics['application_metrics']['latency_ms'],
                    'throughput_rps': latest_metrics['application_metrics']['throughput_rps'],
                }
                return metrics
            else:
                logger.warning("No metrics data available.")
                return {}
        except Exception as e:
            logger.error(f"Failed to retrieve metrics: {e}")
            return {}

    def get_alerts(self) -> List[Dict[str, Any]]:
        """
        Retrieves alerts from the system logs or an alert management module.

        :return: List of alerts.
        """
        try:
            alerts_file = self.dashboard_config['alerts_file']

            if not os.path.exists(alerts_file):
                logger.warning(f"Alerts file not found at '{alerts_file}'.")
                return []

            with open(alerts_file, 'r') as f:
                alerts = json.load(f)

            return alerts
        except Exception as e:
            logger.error(f"Failed to retrieve alerts: {e}")
            return []

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """
        Retrieves a list of available tools for user access.

        :return: List of tools with names, descriptions, and URLs.
        """
        try:
            tools = self.dashboard_config['available_tools']
            return tools
        except Exception as e:
            logger.error(f"Failed to retrieve available tools: {e}")
            return []

    def run_dashboard(self):
        """
        Runs the Flask app for the dashboard.
        """
        logger.info("Starting the dashboard server.")
        try:
            self.server_thread = Thread(target=self.app.run, kwargs={
                'host': self.dashboard_config['dashboard_host'],
                'port': self.dashboard_config['dashboard_port'],
                'debug': False,
                'use_reloader': False
            })
            self.server_thread.start()
            logger.info(f"Dashboard server running at http://{self.dashboard_config['dashboard_host']}:{self.dashboard_config['dashboard_port']}")
        except Exception as e:
            logger.error(f"Failed to start the dashboard server: {e}")
            self.send_notification(
                subject="Dashboard Server Failed to Start",
                message=f"The dashboard server failed to start with the following error:\n\n{e}"
            )
            raise e

    def stop_dashboard(self):
        """
        Stops the Flask app server.
        """
        logger.info("Stopping the dashboard server.")
        try:
            if self.server_thread and self.server_thread.is_alive():
                # Placeholder: Implement server shutdown logic if using a production server
                logger.info("Dashboard server stopped.")
            else:
                logger.info("Dashboard server is not running.")
        except Exception as e:
            logger.error(f"Failed to stop the dashboard server: {e}")
            raise e

    def send_notification(self, subject: str, message: str):
        """
        Sends a notification to the configured recipients.

        :param subject: Subject of the notification.
        :param message: Body of the notification.
        """
        try:
            recipients = self.dashboard_config['notification_recipients']
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
    Demonstrates example usage of the DashboardBuilder class.
    """
    try:
        # Initialize DashboardBuilder
        dashboard_builder = DashboardBuilder()

        # Run the dashboard
        dashboard_builder.run_dashboard()

        # Keep the main thread alive while the dashboard is running
        try:
            while True:
                pass
        except KeyboardInterrupt:
            # Stop the dashboard when interrupted
            dashboard_builder.stop_dashboard()

    except Exception as e:
        logger.exception(f"Error in example usage: {e}")

# --------------------- Main Execution --------------------- #

if __name__ == "__main__":
    # Run the dashboard builder example
    example_usage()
