# src/modules/collaboration/video_voice_tools.py

import os
import logging
from typing import Optional

from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit, leave_room, join_room
from utils.logger import get_logger
from utils.configuration_manager import ConfigurationManager


class VideoVoiceTools:
    """
    Adds live video and voice conferencing features to Hermod’s real-time collaboration system.
    Allows users to initiate voice or video calls directly within the workspace, making collaborative
    efforts smoother and more interactive. Integrates with existing collaboration tools like real-time
    editing and project sharing.
    """

    def __init__(self, project_id: str):
        """
        Initializes the VideoVoiceTools with necessary configurations and tools.

        Args:
            project_id (str): Unique identifier for the project.
        """
        self.logger = get_logger(__name__)
        self.project_id = project_id
        self.config_manager = ConfigurationManager()
        self.config = self.config_manager.get_configuration(project_id)

        # Initialize Flask app
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")

        # Define routes and SocketIO events
        self._setup_routes()
        self._setup_socketio_events()

        self.logger.info(f"VideoVoiceTools initialized for project '{project_id}'.")

    # ----------------------------
    # Flask Routes
    # ----------------------------

    def _setup_routes(self):
        """
        Sets up Flask routes for video and voice conferencing.
        """

        @self.app.route('/video_conference')
        def video_conference():
            """
            Renders the video conference page.
            """
            room = request.args.get('room')
            username = request.args.get('username')
            if not room or not username:
                self.logger.warning("Video conference access denied: 'room' or 'username' not provided.")
                return "Access Denied. Missing 'room' or 'username'.", 400
            return render_template('video_conference.html', room=room, username=username)

    # ----------------------------
    # SocketIO Events
    # ----------------------------

    def _setup_socketio_events(self):
        """
        Sets up SocketIO events for signaling in video and voice conferencing.
        """

        @self.socketio.on('join')
        def handle_join(data):
            """
            Handles a user joining a conference room.
            """
            room = data.get('room')
            username = data.get('username')
            if not room or not username:
                self.logger.warning("Join conference failed: 'room' or 'username' not provided.")
                emit('error', {'message': "Missing 'room' or 'username'."})
                return
            join_room(room)
            self.logger.info(f"User '{username}' joined room '{room}'.")
            emit('user_joined', {'username': username}, room=room)

        @self.socketio.on('leave')
        def handle_leave(data):
            """
            Handles a user leaving a conference room.
            """
            room = data.get('room')
            username = data.get('username')
            if not room or not username:
                self.logger.warning("Leave conference failed: 'room' or 'username' not provided.")
                emit('error', {'message': "Missing 'room' or 'username'."})
                return
            leave_room(room)
            self.logger.info(f"User '{username}' left room '{room}'.")
            emit('user_left', {'username': username}, room=room)

        @self.socketio.on('signal')
        def handle_signal(data):
            """
            Handles signaling data for WebRTC connections.
            """
            room = data.get('room')
            signal_data = data.get('signal')
            sender = data.get('sender')
            receiver = data.get('receiver')
            if not room or not signal_data or not sender or not receiver:
                self.logger.warning("Signal event missing required data.")
                emit('error', {'message': "Missing required signaling data."})
                return
            emit('signal', {'signal': signal_data, 'sender': sender}, room=room, include_self=False)
            self.logger.debug(f"Signal from '{sender}' to '{receiver}' in room '{room}' forwarded.")

    # ----------------------------
    # Server Run Method
    # ----------------------------

    def start_conference_server(self, ssl_cert: Optional[str] = None, ssl_key: Optional[str] = None):
        """
        Starts the Flask-SocketIO server for video and voice conferencing.

        Args:
            ssl_cert (Optional[str]): Path to the SSL certificate file.
            ssl_key (Optional[str]): Path to the SSL key file.
        """
        self.logger.info("Starting Video/Voice Collaboration server.")
        try:
            if ssl_cert and ssl_key:
                self.socketio.run(
                    self.app,
                    host='0.0.0.0',
                    port=5004,
                    ssl_context=(ssl_cert, ssl_key)
                )
            else:
                self.socketio.run(
                    self.app,
                    host='0.0.0.0',
                    port=5004
                )
        except Exception as e:
            self.logger.error(f"Failed to start Video/Voice Collaboration server: {e}")

    # ----------------------------
    # Example Usage and Test Cases
    # ----------------------------

    def run_sample_operations(self):
        """
        Demonstrates sample video and voice conferencing operations.
        """
        self.logger.info("Running sample video and voice conferencing operations.")

        # Example 1: Access the video conference page
        # This would typically be accessed via a web browser with appropriate query parameters
        # Example URL: https://localhost:5004/video_conference?room=room1&username=user1

        # Example 2: Simulate SocketIO events
        # For testing purposes, use a SocketIO test client or perform manual tests via the frontend
        pass


# Example Usage and Test Cases
if __name__ == "__main__":
    import os

    # Initialize VideoVoiceTools
    project_id = os.getenv('PROJECT_ID', 'default_project')  # Ensure PROJECT_ID is set
    video_voice_tools = VideoVoiceTools(project_id=project_id)

    # Run sample operations
    video_voice_tools.run_sample_operations()

    # To start the video/voice collaboration server, uncomment the following lines:
    # ssl_certificate = 'path/to/cert.pem'
    # ssl_key = 'path/to/key.pem'
    # video_voice_tools.start_conference_server(ssl_certificate, ssl_key)
