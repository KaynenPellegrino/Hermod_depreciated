# src/modules/advanced_security/quantum_resistant_algorithms.py

import oqs
import logging
import os
from typing import Tuple, Optional

# Configure logging with RotatingFileHandler to prevent log files from becoming too large
from logging.handlers import RotatingFileHandler

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = RotatingFileHandler('logs/quantum_resistant_algorithms.log', maxBytes=5*1024*1024, backupCount=5)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)


class QuantumResistantCryptography:
    """
    Implements quantum-safe cryptographic algorithms using the Open Quantum Safe (OQS) library.
    Provides functionalities for key encapsulation (encryption/decryption) and digital signatures.
    """

    def __init__(self, kem_algorithm: str = 'Kyber512', sig_algorithm: str = 'Dilithium2'):
        """
        Initializes the QuantumResistantCryptography with specified algorithms.

        :param kem_algorithm: Key Encapsulation Mechanism algorithm name.
        :param sig_algorithm: Digital Signature algorithm name.
        """
        self.kem_algorithm = kem_algorithm
        self.sig_algorithm = sig_algorithm

        # Initialize KEM
        try:
            self.kem = oqs.KeyEncapsulation(self.kem_algorithm)
            if not self.kem:
                raise ValueError(f"KEM algorithm '{self.kem_algorithm}' is not supported.")
            logger.info(f"KEM initialized with algorithm: {self.kem_algorithm}")
        except Exception as e:
            logger.exception(f"Failed to initialize KEM: {e}")
            raise e

        # Initialize Signature
        try:
            self.signature = oqs.Signature(self.sig_algorithm)
            if not self.signature:
                raise ValueError(f"Signature algorithm '{self.sig_algorithm}' is not supported.")
            logger.info(f"Signature initialized with algorithm: {self.sig_algorithm}")
        except Exception as e:
            logger.exception(f"Failed to initialize Signature: {e}")
            raise e

    # --------------------- Key Encapsulation Mechanism (KEM) Methods --------------------- #

    def generate_kem_keypair(self) -> Tuple[bytes, bytes]:
        """
        Generates a key pair for the KEM.

        :return: Tuple containing the public key and secret key.
        """
        try:
            public_key, secret_key = self.kem.generate_keypair()
            logger.info("KEM key pair generated successfully.")
            return public_key, secret_key
        except Exception as e:
            logger.error(f"Error generating KEM key pair: {e}")
            raise e

    def encapsulate_key(self, public_key: bytes) -> Tuple[bytes, bytes]:
        """
        Encapsulates a symmetric key using the recipient's public key.

        :param public_key: Recipient's public key.
        :return: Tuple containing the ciphertext and the shared secret key.
        """
        try:
            ciphertext, shared_secret = self.kem.encapsulate(public_key)
            logger.info("Key encapsulation successful.")
            return ciphertext, shared_secret
        except Exception as e:
            logger.error(f"Error during key encapsulation: {e}")
            raise e

    def decapsulate_key(self, ciphertext: bytes, secret_key: bytes) -> bytes:
        """
        Decapsulates the symmetric key from the ciphertext using the recipient's secret key.

        :param ciphertext: Ciphertext received.
        :param secret_key: Recipient's secret key.
        :return: The shared secret key.
        """
        try:
            shared_secret = self.kem.decapsulate(ciphertext, secret_key)
            logger.info("Key decapsulation successful.")
            return shared_secret
        except Exception as e:
            logger.error(f"Error during key decapsulation: {e}")
            raise e

    # --------------------- Digital Signature Methods --------------------- #

    def generate_signature_keypair(self) -> Tuple[bytes, bytes]:
        """
        Generates a key pair for digital signatures.

        :return: Tuple containing the public key and secret key.
        """
        try:
            public_key, secret_key = self.signature.generate_keypair()
            logger.info("Signature key pair generated successfully.")
            return public_key, secret_key
        except Exception as e:
            logger.error(f"Error generating signature key pair: {e}")
            raise e

    def sign_message(self, message: bytes, secret_key: bytes) -> bytes:
        """
        Signs a message using the secret key.

        :param message: The message to sign.
        :param secret_key: Signer's secret key.
        :return: The generated signature.
        """
        try:
            signature = self.signature.sign(message, secret_key)
            logger.info("Message signed successfully.")
            return signature
        except Exception as e:
            logger.error(f"Error signing message: {e}")
            raise e

    def verify_signature(self, message: bytes, signature: bytes, public_key: bytes) -> bool:
        """
        Verifies a signature against a message and public key.

        :param message: The original message.
        :param signature: The signature to verify.
        :param public_key: Signer's public key.
        :return: True if verification is successful, False otherwise.
        """
        try:
            valid = self.signature.verify(message, signature, public_key)
            if valid:
                logger.info("Signature verification successful.")
            else:
                logger.warning("Signature verification failed.")
            return valid
        except Exception as e:
            logger.error(f"Error verifying signature: {e}")
            return False

    # --------------------- Example Usage --------------------- #

    def example_usage(self):
        """
        Demonstrates example usage of the QuantumResistantCryptography class.
        """
        try:
            # KEM Example
            logger.info("----- KEM Example -----")
            sender_public_key, sender_secret_key = self.generate_kem_keypair()
            receiver_public_key, receiver_secret_key = self.generate_kem_keypair()

            ciphertext, shared_secret_sender = self.encapsulate_key(receiver_public_key)
            shared_secret_receiver = self.decapsulate_key(ciphertext, receiver_secret_key)

            assert shared_secret_sender == shared_secret_receiver, "Shared secrets do not match!"
            logger.info(f"Shared Secret: {shared_secret_sender.hex()}")

            # Signature Example
            logger.info("----- Signature Example -----")
            signer_public_key, signer_secret_key = self.generate_signature_keypair()
            message = b"Secure message for quantum-resistant signature."
            signature = self.sign_message(message, signer_secret_key)

            is_valid = self.verify_signature(message, signature, signer_public_key)
            logger.info(f"Signature valid: {is_valid}")

        except Exception as e:
            logger.exception(f"Error in example usage: {e}")


# --------------------- Main Execution --------------------- #

if __name__ == "__main__":
    # Initialize Quantum Resistant Cryptography
    try:
        qr_crypto = QuantumResistantCryptography(kem_algorithm='Kyber512', sig_algorithm='Dilithium2')
    except Exception as e:
        logger.critical(f"Failed to initialize QuantumResistantCryptography: {e}")
        exit(1)

    # Run example usage
    qr_crypto.example_usage()
