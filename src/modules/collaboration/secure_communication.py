# src/modules/collaboration/secure_communication.py

import os
import logging
from cryptography.fernet import Fernet, InvalidToken
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from src.utils.logger import get_logger
from src.utils.configuration_manager import ConfigurationManager


class SecureCommunication:
    """
    Implements encryption and security measures for communication between users and the Hermod system.
    Ensures that data exchanged during collaboration is protected against interception and unauthorized access.
    This includes SSL/TLS encryption, secure messaging protocols, and key management.
    """

    def __init__(self, project_id: str):
        """
        Initializes the SecureCommunication with necessary configurations and tools.

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

        # Initialize encryption key
        self.encryption_key = self._load_or_generate_key()
        self.cipher_suite = Fernet(self.encryption_key)

        # Define routes and SocketIO events
        self._setup_routes()
        self._setup_socketio_events()

        self.logger.info(f"SecureCommunication initialized for project '{project_id}'.")

    def _load_or_generate_key(self) -> bytes:
        """
        Loads the encryption key from configuration or generates a new one if not present.

        Returns:
            bytes: Encryption key.
        """
        key_path = self.config.get('encryption_key_path', 'encryption/key.key')
        if os.path.exists(key_path):
            try:
                with open(key_path, 'rb') as key_file:
                    key = key_file.read()
                self.logger.info("Encryption key loaded successfully.")
                return key
            except Exception as e:
                self.logger.error(f"Failed to load encryption key from '{key_path}': {e}", exc_info=True)
                raise e
        else:
            try:
                key = Fernet.generate_key()
                os.makedirs(os.path.dirname(key_path), exist_ok=True)
                with open(key_path, 'wb') as key_file:
                    key_file.write(key)
                self.logger.info(f"Generated new encryption key and saved to '{key_path}'.")
                return key
            except Exception as e:
                self.logger.error(f"Failed to generate encryption key: {e}", exc_info=True)
                raise e

    # ----------------------------
    # Flask Routes
    # ----------------------------

    def _setup_routes(self):
        """
        Sets up Flask routes for secure communication.
        """

        @self.app.route('/api/encrypt', methods=['POST'])
        def encrypt_message():
            """
            Endpoint to encrypt a message.
            Expects JSON data with 'message'.
            Returns encrypted message.
            """
            data = request.get_json()
            message = data.get('message')
            if not message:
                self.logger.warning("Encrypt message failed: 'message' not provided.")
                return jsonify({"status": "fail", "message": "Missing 'message'."}), 400

            try:
                encrypted_message = self.cipher_suite.encrypt(message.encode()).decode()
                self.logger.debug(f"Encrypted message: {encrypted_message}")
                return jsonify({"status": "success", "encrypted_message": encrypted_message}), 200
            except Exception as e:
                self.logger.error(f"Failed to encrypt message: {e}", exc_info=True)
                return jsonify({"status": "fail", "message": "Encryption failed."}), 500

        @self.app.route('/api/decrypt', methods=['POST'])
        def decrypt_message():
            """
            Endpoint to decrypt a message.
            Expects JSON data with 'encrypted_message'.
            Returns decrypted message.
            """
            data = request.get_json()
            encrypted_message = data.get('encrypted_message')
            if not encrypted_message:
                self.logger.warning("Decrypt message failed: 'encrypted_message' not provided.")
                return jsonify({"status": "fail", "message": "Missing 'encrypted_message'."}), 400

            try:
                decrypted_message = self.cipher_suite.decrypt(encrypted_message.encode()).decode()
                self.logger.debug(f"Decrypted message: {decrypted_message}")
                return jsonify({"status": "success", "decrypted_message": decrypted_message}), 200
            except InvalidToken:
                self.logger.error("Invalid encryption token. Decryption failed.")
                return jsonify({"status": "fail", "message": "Invalid encryption token."}), 400
            except Exception as e:
                self.logger.error(f"Failed to decrypt message: {e}", exc_info=True)
                return jsonify({"status": "fail", "message": "Decryption failed."}), 500

    # ----------------------------
    # SocketIO Events
    # ----------------------------

    def _setup_socketio_events(self):
        """
        Sets up SocketIO events for secure real-time communication.
        """

        @self.socketio.on('secure_send_message')
        def handle_secure_send_message(data):
            """
            Handles secure sending of messages.
            Expects 'message' in data.
            Encrypts the message and broadcasts it to the room.
            """
            room = data.get('room')
            message = data.get('message')
            if not room or not message:
                self.logger.warning("Secure send message failed: 'room' or 'message' not provided.")
                emit('error', {'message': "Missing 'room' or 'message'."})
                return

            try:
                encrypted_message = self.cipher_suite.encrypt(message.encode()).decode()
                self.logger.debug(f"Encrypted message for room '{room}': {encrypted_message}")
                emit('secure_receive_message', {'message': encrypted_message}, room=room)
            except Exception as e:
                self.logger.error(f"Failed to encrypt and send message: {e}", exc_info=True)
                emit('error', {'message': "Failed to encrypt and send message."})

        @self.socketio.on('secure_receive_message')
        def handle_secure_receive_message(data):
            """
            Handles receiving of secure messages.
            Expects 'encrypted_message' in data.
            Decrypts the message and emits it to the client.
            """
            encrypted_message = data.get('message')
            if not encrypted_message:
                self.logger.warning("Secure receive message failed: 'message' not provided.")
                emit('error', {'message': "Missing 'message'."})
                return

            try:
                decrypted_message = self.cipher_suite.decrypt(encrypted_message.encode()).decode()
                self.logger.debug(f"Decrypted message: {decrypted_message}")
                emit('receive_message', {'message': decrypted_message})
            except InvalidToken:
                self.logger.error("Invalid encryption token. Decryption failed.")
                emit('error', {'message': "Invalid encryption token."})
            except Exception as e:
                self.logger.error(f"Failed to decrypt received message: {e}", exc_info=True)
                emit('error', {'message': "Failed to decrypt message."})

    # ----------------------------
    # Server Run Method
    # ----------------------------

    def start_secure_server(self, ssl_cert: str, ssl_key: str):
        """
        Starts the Flask-SocketIO server with SSL/TLS encryption.

        Args:
            ssl_cert (str): Path to the SSL certificate file.
            ssl_key (str): Path to the SSL key file.
        """
        self.logger.info("Starting Secure Communication server with SSL/TLS.")
        try:
            self.socketio.run(
                self.app,
                host='0.0.0.0',
                port=5000,
                ssl_context=(ssl_cert, ssl_key)
            )
        except Exception as e:
            self.logger.error(f"Failed to start Secure Communication server: {e}")

    # ----------------------------
    # Example Usage and Test Cases
    # ----------------------------

    def run_sample_operations(self):
        """
        Demonstrates sample secure communication operations.
        """
        self.logger.info("Running sample secure communication operations.")

        import requests

        # Sample message
        sample_message = "Hello, this is a secure message."

        # Encrypt the message
        encrypt_response = requests.post(
            'https://localhost:5000/api/encrypt',
            json={'message': sample_message},
            verify=False  # For testing purposes only; remove in production
        )
        if encrypt_response.status_code == 200:
            encrypted_message = encrypt_response.json().get('encrypted_message')
            self.logger.info(f"Encrypted Message: {encrypted_message}")

            # Decrypt the message
            decrypt_response = requests.post(
                'https://localhost:5000/api/decrypt',
                json={'encrypted_message': encrypted_message},
                verify=False  # For testing purposes only; remove in production
            )
            if decrypt_response.status_code == 200:
                decrypted_message = decrypt_response.json().get('decrypted_message')
                self.logger.info(f"Decrypted Message: {decrypted_message}")
            else:
                self.logger.error("Failed to decrypt the message.")
        else:
            self.logger.error("Failed to encrypt the message.")


# Example Usage and Test Cases
if __name__ == "__main__":
    import os

    # Initialize SecureCommunication
    project_id = os.getenv('PROJECT_ID', 'default_project')  # Ensure PROJECT_ID is set
    secure_comm = SecureCommunication(project_id=project_id)

    # Run sample operations
    # Note: To run the server, you need SSL certificates. For testing, you can generate self-signed certificates.
    # Example command to generate self-signed cert:
    # openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

    # Uncomment the lines below to start the server with SSL (replace 'cert.pem' and 'key.pem' with your paths)
    # ssl_certificate = 'path/to/cert.pem'
    # ssl_key = 'path/to/key.pem'
    # secure_comm.start_secure_server(ssl_certificate, ssl_key)

    # Run sample encryption and decryption
    secure_comm.run_sample_operations()
