# src/modules/collaboration/collaboration_tools.py

import os
import logging
from typing import List, Dict, Any
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from flask import Flask, request
from flask_socketio import SocketIO, emit

from utils.logger import get_logger
from utils.configuration_manager import ConfigurationManager
from src.modules.self_optimization.persistent_memory import PersistentMemory


class CollaborationTools:
    """
    Provides functionalities that enable collaboration among developers or users within the Hermod platform.
    This includes features like shared editing, project sharing, real-time communication tools,
    and integration with collaboration platforms like Slack or Microsoft Teams.
    """

    def __init__(self, project_id: str):
        """
        Initializes the CollaborationTools with necessary configurations.

        Args:
            project_id (str): Unique identifier for the project.
        """
        self.logger = get_logger(__name__)
        self.project_id = project_id
        self.config_manager = ConfigurationManager()
        self.config = self.config_manager.get_configuration(project_id)
        self.persistent_memory = PersistentMemory(project_id=project_id)

        # Initialize Slack client
        self.slack_token = self.config.get('slack_token')
        if not self.slack_token:
            self.logger.warning("Slack token not found in configuration. Slack integration will be disabled.")
            self.slack_client = None
        else:
            self.slack_client = WebClient(token=self.slack_token)
            self.logger.info("Slack client initialized.")

        # Initialize Flask app for real-time communication
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")

        # Define routes and SocketIO events
        self._setup_routes()
        self._setup_socketio_events()

    # ----------------------------
    # Slack Integration
    # ----------------------------

    def send_slack_message(self, channel: str, message: str) -> bool:
        """
        Sends a message to a specified Slack channel.

        Args:
            channel (str): Slack channel ID or name.
            message (str): Message content.

        Returns:
            bool: True if message was sent successfully, False otherwise.
        """
        if not self.slack_client:
            self.logger.error("Slack client is not initialized.")
            return False

        try:
            response = self.slack_client.chat_postMessage(channel=channel, text=message)
            self.logger.info(f"Message sent to Slack channel '{channel}': {message}")
            return True
        except SlackApiError as e:
            self.logger.error(f"Failed to send message to Slack: {e.response['error']}")
            return False

    def create_slack_channel(self, channel_name: str) -> bool:
        """
        Creates a new Slack channel.

        Args:
            channel_name (str): Name of the channel to create.

        Returns:
            bool: True if channel was created successfully, False otherwise.
        """
        if not self.slack_client:
            self.logger.error("Slack client is not initialized.")
            return False

        try:
            response = self.slack_client.conversations_create(name=channel_name)
            self.logger.info(f"Slack channel '{channel_name}' created successfully.")
            return True
        except SlackApiError as e:
            self.logger.error(f"Failed to create Slack channel: {e.response['error']}")
            return False

    # ----------------------------
    # Real-Time Communication
    # ----------------------------

    def _setup_routes(self):
        """
        Sets up Flask routes for the collaboration dashboard.
        """
        @self.app.route('/api/join_session', methods=['POST'])
        def join_session():
            """
            Endpoint for users to join a collaborative session.
            Expects JSON data with 'username' and 'session_id'.
            """
            data = request.get_json()
            username = data.get('username')
            session_id = data.get('session_id')
            if not username or not session_id:
                self.logger.warning("Missing 'username' or 'session_id' in join_session request.")
                return {"status": "fail", "message": "Missing 'username' or 'session_id'."}, 400

            self.logger.info(f"User '{username}' is joining session '{session_id}'.")
            emit('user_joined', {'username': username, 'session_id': session_id}, broadcast=True)
            return {"status": "success", "message": f"User '{username}' joined session '{session_id}'."}, 200

    def _setup_socketio_events(self):
        """
        Sets up SocketIO events for real-time communication.
        """

        @self.socketio.on('send_message')
        def handle_send_message(data):
            """
            Handles incoming messages and broadcasts them to all connected clients.
            """
            username = data.get('username')
            message = data.get('message')
            session_id = data.get('session_id')

            if not username or not message or not session_id:
                self.logger.warning("Missing data in send_message event.")
                emit('error', {'message': "Missing 'username', 'message', or 'session_id'."})
                return

            self.logger.info(f"Message from '{username}' in session '{session_id}': {message}")
            emit('receive_message', {'username': username, 'message': message, 'session_id': session_id}, broadcast=True)

    def start_real_time_server(self):
        """
        Starts the Flask-SocketIO server for real-time communication.
        """
        self.logger.info("Starting real-time communication server.")
        try:
            self.socketio.run(self.app, host='0.0.0.0', port=5001)
        except Exception as e:
            self.logger.error(f"Failed to start real-time server: {e}")

    # ----------------------------
    # Project Sharing
    # ----------------------------

    def share_project(self, project_id: str, user_emails: List[str]) -> bool:
        """
        Shares a project with specified users via Slack invitations.

        Args:
            project_id (str): ID of the project to share.
            user_emails (List[str]): List of user email addresses to share the project with.

        Returns:
            bool: True if project was shared successfully, False otherwise.
        """
        if not self.slack_client:
            self.logger.error("Slack client is not initialized.")
            return False

        try:
            # Create a dedicated Slack channel for the project if it doesn't exist
            channel_name = f"project-{project_id}"
            if not self.create_slack_channel(channel_name):
                self.logger.warning(f"Slack channel '{channel_name}' might already exist.")

            # Invite users to the Slack channel
            for email in user_emails:
                try:
                    # Find user ID by email
                    user_response = self.slack_client.users_lookupByEmail(email=email)
                    user_id = user_response['user']['id']
                    self.slack_client.conversations_invite(channel=channel_name, users=user_id)
                    self.logger.info(f"Invited '{email}' to Slack channel '{channel_name}'.")
                except SlackApiError as e:
                    self.logger.error(f"Failed to invite '{email}' to Slack channel: {e.response['error']}")

            return True
        except Exception as e:
            self.logger.error(f"Failed to share project '{project_id}': {e}", exc_info=True)
            return False

    # ----------------------------
    # Shared Editing (Basic Implementation)
    # ----------------------------

    def create_shared_document(self, document_name: str) -> bool:
        """
        Creates a shared document for collaborative editing.
        This is a placeholder implementation. Integration with services like Google Docs or Etherpad can be added.

        Args:
            document_name (str): Name of the document to create.

        Returns:
            bool: True if document was created successfully, False otherwise.
        """
        self.logger.info(f"Creating shared document '{document_name}'.")
        try:
            # Placeholder: Create a simple markdown file as a shared document
            doc_path = os.path.join(self.codebase_path, 'shared_documents', f"{document_name}.md")
            os.makedirs(os.path.dirname(doc_path), exist_ok=True)
            with open(doc_path, 'w') as doc_file:
                doc_file.write(f"# {document_name}\n\nCollaborative editing started.\n")
            self.logger.info(f"Shared document '{doc_path}' created successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to create shared document '{document_name}': {e}", exc_info=True)
            return False

    # ----------------------------
    # Example Usage and Test Cases
    # ----------------------------

    def run_sample_operations(self):
        """
        Demonstrates sample collaboration operations.
        """
        self.logger.info("Running sample collaboration operations.")

        # Example 1: Send Slack Message
        if self.slack_client:
            self.send_slack_message(channel='#general', message="Hermod Collaboration Tools Initialized.")

        # Example 2: Share Project with Users
        project_id = self.project_id
        user_emails = ['user1@example.com', 'user2@example.com']
        self.share_project(project_id, user_emails)

        # Example 3: Create Shared Document
        self.create_shared_document('Project_Plan')

        # Example 4: Start Real-Time Communication Server
        # Uncomment the line below to start the server (Note: This will block the execution)
        # self.start_real_time_server()


# Example Usage and Test Cases
if __name__ == "__main__":
    import os

    # Initialize CollaborationTools
    project_id = os.getenv('PROJECT_ID', 'default_project')  # Ensure PROJECT_ID is set
    collaboration_tools = CollaborationTools(project_id=project_id)

    # Run sample operations
    collaboration_tools.run_sample_operations()

    # To start the real-time communication server, uncomment the following line:
    # collaboration_tools.start_real_time_server()
