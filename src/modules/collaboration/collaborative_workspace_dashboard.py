# src/modules/collaboration/collaborative_workspace_dashboard.py

import os
from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_socketio import SocketIO, emit
import logging

from utils.logger import get_logger
from utils.configuration_manager import ConfigurationManager
from src.modules.collaboration.collaboration_tools import CollaborationTools


class CollaborativeWorkspaceDashboard:
    """
    Provides a user interface for managing real-time collaborative sessions within Hermod.
    Integrates with existing collaboration tools, allowing users to join live sessions,
    view active collaborators, and manage shared project details directly from a visual dashboard.
    """

    def __init__(self, project_id: str):
        """
        Initializes the CollaborativeWorkspaceDashboard with necessary configurations and tools.

        Args:
            project_id (str): Unique identifier for the project.
        """
        self.logger = get_logger(__name__)
        self.project_id = project_id
        self.config_manager = ConfigurationManager()
        self.config = self.config_manager.get_configuration(project_id)

        # Initialize collaboration tools
        self.collaboration_tools = CollaborationTools(project_id=project_id)

        # Initialize Flask app
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")

        # Initialize session data
        self.active_sessions = {}  # session_id: {'users': [username1, username2, ...]}

        # Define routes and SocketIO events
        self._setup_routes()
        self._setup_socketio_events()

    # ----------------------------
    # Flask Routes
    # ----------------------------

    def _setup_routes(self):
        """
        Sets up Flask routes for the collaborative workspace dashboard.
        """

        @self.app.route('/')
        def index():
            """
            Renders the main dashboard page.
            """
            return render_template('dashboard.html')

        @self.app.route('/create_session', methods=['POST'])
        def create_session():
            """
            Endpoint to create a new collaborative session.
            Expects form data with 'session_name'.
            """
            session_name = request.form.get('session_name')
            if not session_name:
                self.logger.warning("Session creation failed: 'session_name' not provided.")
                return redirect(url_for('index'))

            # Generate a unique session ID
            session_id = f"session_{len(self.active_sessions) + 1}"
            self.active_sessions[session_id] = {'session_name': session_name, 'users': []}
            self.logger.info(f"Created new session '{session_name}' with ID '{session_id}'.")
            return redirect(url_for('session', session_id=session_id))

        @self.app.route('/session/<session_id>')
        def session(session_id):
            """
            Renders the collaborative session page.
            """
            session_info = self.active_sessions.get(session_id)
            if not session_info:
                self.logger.warning(f"Session '{session_id}' not found.")
                return redirect(url_for('index'))

            return render_template('session.html', session_id=session_id, session_name=session_info['session_name'])

    # ----------------------------
    # SocketIO Events
    # ----------------------------

    def _setup_socketio_events(self):
        """
        Sets up SocketIO events for real-time communication within sessions.
        """

        @self.socketio.on('join')
        def handle_join(data):
            """
            Handles a user joining a session.
            """
            username = data.get('username')
            session_id = data.get('session_id')
            if not username or not session_id:
                self.logger.warning("Join event missing 'username' or 'session_id'.")
                emit('error', {'message': "Missing 'username' or 'session_id'."})
                return

            if session_id not in self.active_sessions:
                self.logger.warning(f"Join event for non-existent session '{session_id}'.")
                emit('error', {'message': f"Session '{session_id}' does not exist."})
                return

            # Add user to session
            if username not in self.active_sessions[session_id]['users']:
                self.active_sessions[session_id]['users'].append(username)
                self.logger.info(f"User '{username}' joined session '{session_id}'.")
                emit('user_joined', {'username': username, 'session_id': session_id}, broadcast=True)

        @self.socketio.on('send_message')
        def handle_send_message(data):
            """
            Handles incoming messages and broadcasts them to all users in the session.
            """
            username = data.get('username')
            message = data.get('message')
            session_id = data.get('session_id')

            if not username or not message or not session_id:
                self.logger.warning("Send_message event missing data.")
                emit('error', {'message': "Missing 'username', 'message', or 'session_id'."})
                return

            if session_id not in self.active_sessions:
                self.logger.warning(f"Message sent to non-existent session '{session_id}'.")
                emit('error', {'message': f"Session '{session_id}' does not exist."})
                return

            self.logger.info(f"Message from '{username}' in session '{session_id}': {message}")
            emit('receive_message', {'username': username, 'message': message, 'session_id': session_id}, broadcast=True)

        @self.socketio.on('leave')
        def handle_leave(data):
            """
            Handles a user leaving a session.
            """
            username = data.get('username')
            session_id = data.get('session_id')
            if not username or not session_id:
                self.logger.warning("Leave event missing 'username' or 'session_id'.")
                emit('error', {'message': "Missing 'username' or 'session_id'."})
                return

            if session_id in self.active_sessions and username in self.active_sessions[session_id]['users']:
                self.active_sessions[session_id]['users'].remove(username)
                self.logger.info(f"User '{username}' left session '{session_id}'.")
                emit('user_left', {'username': username, 'session_id': session_id}, broadcast=True)

    # ----------------------------
    # Dashboard Server
    # ----------------------------

    def start_dashboard_server(self):
        """
        Starts the Flask server for the collaborative workspace dashboard.
        """
        self.logger.info("Starting Collaborative Workspace Dashboard server.")
        try:
            self.socketio.run(self.app, host='0.0.0.0', port=5002)
        except Exception as e:
            self.logger.error(f"Failed to start dashboard server: {e}")

    # ----------------------------
    # Example Usage and Test Cases
    # ----------------------------

    def run_sample_operations(self):
        """
        Demonstrates sample dashboard operations.
        """
        self.logger.info("Running sample dashboard operations.")

        # Example 1: Create a new session programmatically
        session_id = "session_1"
        self.active_sessions[session_id] = {'session_name': 'Development Sprint', 'users': []}
        self.logger.info(f"Programmatically created session '{session_id}'.")

        # Example 2: Share session with users via Slack
        user_emails = ['user1@example.com', 'user2@example.com']
        self.collaboration_tools.share_project(session_id, user_emails)

        # Example 3: Create a shared document
        self.collaboration_tools.create_shared_document('Sprint_Plan')

        # Example 4: Start the dashboard server
        # Uncomment the line below to start the server (Note: This will block the execution)
        # self.start_dashboard_server()


# Example Usage and Test Cases
if __name__ == "__main__":
    import os

    # Initialize CollaborativeWorkspaceDashboard
    project_id = os.getenv('PROJECT_ID', 'default_project')  # Ensure PROJECT_ID is set
    dashboard = CollaborativeWorkspaceDashboard(project_id=project_id)

    # Run sample operations
    dashboard.run_sample_operations()

    # To start the dashboard server, uncomment the following line:
    # dashboard.start_dashboard_server()
