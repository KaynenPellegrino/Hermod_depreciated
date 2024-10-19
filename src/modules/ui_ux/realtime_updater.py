# src/modules/ui_ux/realtime_updater.py

import logging
from flask_socketio import SocketIO, emit
from threading import Thread
from src.utils.configuration_manager import ConfigurationManager
from src.modules.notifications.notification_manager import NotificationManager

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('logs/realtime_updater.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class RealTimeUpdater:
    """
    Real-Time UI Updates
    Implements mechanisms to update the UI in real-time based on data changes or events,
    enhancing interactivity and responsiveness.
    """

    def __init__(self, app):
        """
        Initializes the RealTimeUpdater with necessary configurations and Flask app.

        :param app: The Flask application instance.
        """
        try:
            self.config_manager = ConfigurationManager()
            self.notification_manager = NotificationManager()
            self.load_realtime_config()
            self.socketio = SocketIO(app, async_mode='eventlet')
            self.setup_socketio_events()
            self.server_thread: Thread = None
            logger.info("RealTimeUpdater initialized successfully.")
        except Exception as e:
            logger.exception(f"Failed to initialize RealTimeUpdater: {e}")
            raise e

    def load_realtime_config(self):
        """
        Loads real-time updater configurations from the configuration manager or environment variables.
        """
        logger.info("Loading real-time updater configurations.")
        try:
            self.realtime_config = {
                'notification_recipients': self.config_manager.get('NOTIFICATION_RECIPIENTS', '').split(','),
            }
            logger.info(f"Real-time updater configurations loaded: {self.realtime_config}")
        except Exception as e:
            logger.error(f"Failed to load real-time updater configurations: {e}")
            raise e

    def setup_socketio_events(self):
        """
        Sets up the SocketIO events.
        """
        logger.info("Setting up SocketIO events.")

        @self.socketio.on('connect')
        def handle_connect():
            logger.info(f"Client connected: {self.get_client_sid()}")
            emit('connected', {'message': 'Connected to real-time updates.'})

        @self.socketio.on('disconnect')
        def handle_disconnect():
            logger.info(f"Client disconnected: {self.get_client_sid()}")

    def get_client_sid(self):
        """
        Retrieves the session ID of the connected client.

        :return: Client session ID.
        """
        from flask import request
        return request.sid

    def run_realtime_server(self, host='0.0.0.0', port=5000):
        """
        Runs the SocketIO server.

        :param host: Host IP address.
        :param port: Port number.
        """
        logger.info("Starting the real-time server.")
        try:
            self.server_thread = Thread(target=self.socketio.run, args=(self.socketio.server,), kwargs={
                'app': self.socketio.server.app,
                'host': host,
                'port': port,
                'debug': False,
                'use_reloader': False
            })
            self.server_thread.start()
            logger.info(f"Real-time server running at http://{host}:{port}")
        except Exception as e:
            logger.error(f"Failed to start the real-time server: {e}")
            self.send_notification(
                subject="Real-Time Server Failed to Start",
                message=f"The real-time server failed to start with the following error:\n\n{e}"
            )
            raise e

    def emit_update(self, event_name: str, data):
        """
        Emits an event to all connected clients.

        :param event_name: Name of the event.
        :param data: Data to send with the event.
        """
        try:
            logger.info(f"Emitting event '{event_name}' with data: {data}")
            self.socketio.emit(event_name, data)
        except Exception as e:
            logger.error(f"Failed to emit event '{event_name}': {e}")

    def send_notification(self, subject: str, message: str):
        """
        Sends a notification to the configured recipients.

        :param subject: Subject of the notification.
        :param message: Body of the notification.
        """
        try:
            recipients = self.realtime_config['notification_recipients']
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
