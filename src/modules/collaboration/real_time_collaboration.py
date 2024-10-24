# src/modules/collaboration/real_time_collaboration.py

import os
import logging
from typing import Dict, Any, List
from flask import Flask, request
from flask_socketio import SocketIO, emit, join_room, leave_room
from utils.logger import get_logger
from utils.configuration_manager import ConfigurationManager
from src.modules.collaboration.collaboration_tools import CollaborationTools
from src.modules.collaboration.project_sharing_manager import ProjectSharingManager
from src.modules.collaboration.secure_collaboration_protocol import SecureCollaborationProtocol
from src.modules.collaboration.version_control import VersionControl


class RealTimeCollaboration:
    """
    Manages live editing sessions, real-time communication, and version control for shared codebases.
    Allows multiple users to work on the same project simultaneously, with changes reflected live.
    """

    def __init__(self, project_id: str):
        """
        Initializes the RealTimeCollaboration with necessary configurations and tools.

        Args:
            project_id (str): Unique identifier for the project.
        """
        self.logger = get_logger(__name__)
        self.project_id = project_id
        self.config_manager = ConfigurationManager()
        self.config = self.config_manager.get_configuration(project_id)

        # Initialize collaboration tools
        self.collaboration_tools = CollaborationTools(project_id=project_id)
        self.project_sharing_manager = ProjectSharingManager(project_id=project_id)

        # Initialize secure collaboration protocols
        self.secure_protocol = SecureCollaborationProtocol(project_id=project_id)

        # Initialize version control
        repo_path = self.config.get('repository_path', '/path/to/repo')  # Ensure correct path
        self.version_control = VersionControl(repo_path=repo_path)

        # Initialize Flask app and SocketIO
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")

        # Shared documents data structure
        # Format: {document_id: {'document_name': str, 'content': str, 'version': int, 'last_modified': str}}
        self.shared_documents: Dict[str, Dict[str, Any]] = {}

        # Define routes and SocketIO events
        self._setup_routes()
        self._setup_socketio_events()

        self.logger.info(f"RealTimeCollaboration initialized for project '{project_id}'.")

    # ----------------------------
    # Flask Routes
    # ----------------------------

    def _setup_routes(self):
        """
        Sets up Flask routes for real-time collaboration features.
        """

        @self.app.route('/api/create_document', methods=['POST'])
        def create_document():
            """
            Endpoint to create a new shared document.
            Expects JSON data with 'document_name'.
            """
            data = request.get_json()
            document_name = data.get('document_name')
            if not document_name:
                self.logger.warning("Create document failed: 'document_name' not provided.")
                return {"status": "fail", "message": "Missing 'document_name'."}, 400

            document_id = f"doc_{len(self.shared_documents) + 1}"
            self.shared_documents[document_id] = {
                'document_name': document_name,
                'content': '',
                'version': 1,
                'last_modified': None
            }
            self.logger.info(f"Created new shared document '{document_name}' with ID '{document_id}'.")
            return {"status": "success", "document_id": document_id}, 200

        @self.app.route('/api/get_document/<document_id>', methods=['GET'])
        def get_document(document_id):
            """
            Endpoint to retrieve a shared document's content.

            Args:
                document_id (str): ID of the document to retrieve.

            Returns:
                JSON response with document content.
            """
            document = self.shared_documents.get(document_id)
            if not document:
                self.logger.warning(f"Get document failed: Document ID '{document_id}' not found.")
                return {"status": "fail", "message": f"Document ID '{document_id}' not found."}, 404

            self.logger.info(f"Retrieved content for document ID '{document_id}'.")
            return {
                "status": "success",
                "document_name": document['document_name'],
                "content": document['content'],
                "version": document['version']
            }, 200

    # ----------------------------
    # SocketIO Events
    # ----------------------------

    def _setup_socketio_events(self):
        """
        Sets up SocketIO events for real-time collaboration.
        """

        @self.socketio.on('join_document')
        def handle_join_document(data):
            """
            Handles a user joining a document editing session.

            Args:
                data (dict): Data containing 'username' and 'document_id'.
            """
            username = data.get('username')
            document_id = data.get('document_id')

            if not username or not document_id:
                self.logger.warning("Join document event missing 'username' or 'document_id'.")
                emit('error', {'message': "Missing 'username' or 'document_id'."})
                return

            if document_id not in self.shared_documents:
                self.logger.warning(f"Join document failed: Document ID '{document_id}' does not exist.")
                emit('error', {'message': f"Document ID '{document_id}' does not exist."})
                return

            join_room(document_id)
            self.shared_documents[document_id]['last_modified'] = self._current_timestamp()
            self.logger.info(f"User '{username}' joined document '{document_id}'.")
            emit('user_joined', {'username': username, 'document_id': document_id}, room=document_id)

        @self.socketio.on('leave_document')
        def handle_leave_document(data):
            """
            Handles a user leaving a document editing session.

            Args:
                data (dict): Data containing 'username' and 'document_id'.
            """
            username = data.get('username')
            document_id = data.get('document_id')

            if not username or not document_id:
                self.logger.warning("Leave document event missing 'username' or 'document_id'.")
                emit('error', {'message': "Missing 'username' or 'document_id'."})
                return

            leave_room(document_id)
            self.logger.info(f"User '{username}' left document '{document_id}'.")
            emit('user_left', {'username': username, 'document_id': document_id}, room=document_id)

        @self.socketio.on('edit_document')
        def handle_edit_document(data):
            """
            Handles a document edit event. Broadcasts the change to all users in the room.

            Args:
                data (dict): Data containing 'document_id', 'username', 'changes', 'version'.
            """
            document_id = data.get('document_id')
            username = data.get('username')
            changes = data.get('changes')  # e.g., new content
            version = data.get('version')

            if not document_id or not username or changes is None or version is None:
                self.logger.warning("Edit document event missing required data.")
                emit('error', {'message': "Missing required data for editing document."})
                return

            document = self.shared_documents.get(document_id)
            if not document:
                self.logger.warning(f"Edit document failed: Document ID '{document_id}' does not exist.")
                emit('error', {'message': f"Document ID '{document_id}' does not exist."})
                return

            # Simple version control: ensure the edit is based on the latest version
            if version != document['version']:
                self.logger.warning(f"Version mismatch for document '{document_id}'. Expected: {document['version']}, Received: {version}.")
                emit('version_conflict', {
                    'message': "Version conflict detected.",
                    'current_version': document['version'],
                    'content': document['content']
                })
                return

            # Update document content and version
            document['content'] = changes
            document['version'] += 1
            document['last_modified'] = self._current_timestamp()

            self.logger.info(f"Document '{document_id}' edited by '{username}'. New version: {document['version']}.")
            emit('document_updated', {
                'document_id': document_id,
                'content': document['content'],
                'version': document['version'],
                'last_modified': document['last_modified']
            }, room=document_id)

            # Optionally, save changes to version control
            self._commit_changes(document_id, username, changes)

    # ----------------------------
    # Helper Methods
    # ----------------------------

    def _current_timestamp(self) -> str:
        """
        Returns the current timestamp as a string.

        Returns:
            str: Current timestamp.
        """
        from datetime import datetime
        return datetime.now().isoformat()

    def _commit_changes(self, document_id: str, username: str, changes: str):
        """
        Commits the document changes to version control.

        Args:
            document_id (str): ID of the document being edited.
            username (str): Name of the user who made the changes.
            changes (str): The new content of the document.
        """
        try:
            commit_message = f"Document '{document_id}' updated by '{username}' at {self._current_timestamp()}."
            # Save the document content to file
            document_name = self.shared_documents[document_id]['document_name']
            document_path = os.path.join(self.config.get('shared_documents_path', 'shared_documents'), f"{document_name}.md")
            os.makedirs(os.path.dirname(document_path), exist_ok=True)
            with open(document_path, 'w') as doc_file:
                doc_file.write(changes)

            # Commit changes to version control
            self.version_control.commit(commit_message)
            self.version_control.push()
            self.logger.info(f"Committed changes for document '{document_id}' by '{username}'.")
        except Exception as e:
            self.logger.error(f"Failed to commit changes for document '{document_id}': {e}", exc_info=True)

    # ----------------------------
    # Performance Optimization
    # ----------------------------

    def optimize_live_sessions(self):
        """
        Optimizes live editing sessions based on performance metrics or feedback.
        """
        self.logger.info("Optimizing live editing sessions.")
        # Placeholder for optimization logic
        # Could involve adjusting server resources, load balancing, etc.
        pass

    # ----------------------------
    # Example Usage and Test Cases
    # ----------------------------

    def run_sample_operations(self):
        """
        Demonstrates sample real-time collaboration operations.
        """
        self.logger.info("Running sample real-time collaboration operations.")

        # Example 1: Create a new document
        with self.app.test_client() as client:
            response = client.post('/api/create_document', json={'document_name': 'Collaborative_Document'})
            if response.status_code == 200:
                document_id = response.get_json().get('document_id')
                self.logger.info(f"Sample document created with ID '{document_id}'.")
            else:
                self.logger.error("Failed to create sample document.")
                return

        # Example 2: Simulate a user joining the document
        # This would normally be handled by the front-end via SocketIO
        # For testing purposes, we can simulate the event using SocketIO test client
        # Note: Flask-SocketIO test client can be used for more thorough testing
        # Here, we'll outline the steps without actual implementation

        # Example 3: Simulate editing the document
        # Similar to Example 2, use SocketIO test client or integrate with a front-end

    # ----------------------------
    # Server Run Method
    # ----------------------------

    def start_server(self):
        """
        Starts the Flask-SocketIO server for real-time collaboration.
        """
        self.logger.info("Starting Real-Time Collaboration server.")
        try:
            self.socketio.run(self.app, host='0.0.0.0', port=5003)
        except Exception as e:
            self.logger.error(f"Failed to start Real-Time Collaboration server: {e}")


# Example Usage and Test Cases
if __name__ == "__main__":
    import os

    # Initialize RealTimeCollaboration
    project_id = os.getenv('PROJECT_ID', 'default_project')  # Ensure PROJECT_ID is set
    real_time_collaboration = RealTimeCollaboration(project_id=project_id)

    # Run sample operations
    real_time_collaboration.run_sample_operations()

    # To start the real-time collaboration server, uncomment the following line:
    # real_time_collaboration.start_server()
