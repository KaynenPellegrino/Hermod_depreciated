# src/modules/collaboration/secure_collaboration_protocol.py

import os
import logging
from typing import Dict, Any
from cryptography.fernet import Fernet, InvalidToken
from src.utils.logger import get_logger
from src.utils.configuration_manager import ConfigurationManager
from staging import CollaborationTools


class SecureCollaborationProtocol:
    """
    Extends the secure_communication.py file by adding encrypted channels for real-time collaboration.
    Ensures that all communication between users during collaboration is secured, including shared documents,
    codebases, and real-time messaging.
    """

    def __init__(self, project_id: str):
        """
        Initializes the SecureCollaborationProtocol with necessary configurations and tools.

        Args:
            project_id (str): Unique identifier for the project.
        """
        self.logger = get_logger(__name__)
        self.project_id = project_id
        self.config_manager = ConfigurationManager()
        self.config = self.config_manager.get_configuration(project_id)

        # Initialize collaboration tools
        self.collaboration_tools = CollaborationTools(project_id=project_id)

        # Initialize encryption key
        self.encryption_key = self._load_or_generate_key()
        self.cipher_suite = Fernet(self.encryption_key)

        self.logger.info(f"SecureCollaborationProtocol initialized for project '{project_id}'.")

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
    # Encryption Methods
    # ----------------------------

    def encrypt_message(self, message: str) -> bytes:
        """
        Encrypts a message.

        Args:
            message (str): The message to encrypt.

        Returns:
            bytes: Encrypted message.
        """
        try:
            encrypted_message = self.cipher_suite.encrypt(message.encode())
            self.logger.debug(f"Encrypted message: {encrypted_message}")
            return encrypted_message
        except Exception as e:
            self.logger.error(f"Failed to encrypt message: {e}", exc_info=True)
            raise e

    def decrypt_message(self, encrypted_message: bytes) -> str:
        """
        Decrypts an encrypted message.

        Args:
            encrypted_message (bytes): The encrypted message to decrypt.

        Returns:
            str: Decrypted message.
        """
        try:
            decrypted_message = self.cipher_suite.decrypt(encrypted_message).decode()
            self.logger.debug(f"Decrypted message: {decrypted_message}")
            return decrypted_message
        except InvalidToken:
            self.logger.error("Invalid encryption token. Failed to decrypt message.")
            raise
        except Exception as e:
            self.logger.error(f"Failed to decrypt message: {e}", exc_info=True)
            raise e

    # ----------------------------
    # Secure Communication Integration
    # ----------------------------

    def secure_send_message(self, socketio_instance, room: str, message: str):
        """
        Encrypts and sends a message to a specific room.

        Args:
            socketio_instance (SocketIO): The SocketIO instance to use for sending the message.
            room (str): The room to send the message to.
            message (str): The plaintext message to send.
        """
        try:
            encrypted_message = self.encrypt_message(message)
            socketio_instance.emit('secure_receive_message', {'message': encrypted_message}, room=room)
            self.logger.info(f"Encrypted message sent to room '{room}'.")
        except Exception as e:
            self.logger.error(f"Failed to send secure message: {e}", exc_info=True)

    def secure_receive_message(self, encrypted_message: bytes) -> str:
        """
        Decrypts a received message.

        Args:
            encrypted_message (bytes): The encrypted message received.

        Returns:
            str: The decrypted plaintext message.
        """
        try:
            decrypted_message = self.decrypt_message(encrypted_message)
            self.logger.info("Decrypted received message successfully.")
            return decrypted_message
        except Exception as e:
            self.logger.error(f"Failed to decrypt received message: {e}", exc_info=True)
            return ""

    # ----------------------------
    # Example Usage and Test Cases
    # ----------------------------

    def run_sample_operations(self):
        """
        Demonstrates sample secure communication operations.
        """
        self.logger.info("Running sample secure communication operations.")

        # Example 1: Encrypt and decrypt a message
        sample_message = "Hello, this is a secure message."
        encrypted = self.encrypt_message(sample_message)
        decrypted = self.decrypt_message(encrypted)
        self.logger.info(f"Original Message: {sample_message}")
        self.logger.info(f"Encrypted Message: {encrypted}")
        self.logger.info(f"Decrypted Message: {decrypted}")

        # Example 2: Integrate with SocketIO to send encrypted messages
        # Note: Actual SocketIO integration would require a running server and client
        # This is a placeholder for demonstration purposes
        # Assuming socketio_instance is an active SocketIO instance
        # self.secure_send_message(socketio_instance, 'document_room', 'This is a secure update to the document.')

        # Example 3: Receive and decrypt a message
        # decrypted_msg = self.secure_receive_message(encrypted)
        # self.logger.info(f"Decrypted Received Message: {decrypted_msg}")


# Example Usage and Test Cases
if __name__ == "__main__":
    import os

    # Initialize SecureCollaborationProtocol
    project_id = os.getenv('PROJECT_ID', 'default_project')  # Ensure PROJECT_ID is set
    secure_collab_protocol = SecureCollaborationProtocol(project_id=project_id)

    # Run sample operations
    secure_collab_protocol.run_sample_operations()
