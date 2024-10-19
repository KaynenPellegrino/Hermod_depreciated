# src/modules/risk_management/blockchain_auditor.py

import os
import logging
import hashlib
import json
import datetime
from typing import List, Dict, Any
from src.utils.configuration_manager import ConfigurationManager
from src.modules.notifications.notification_manager import NotificationManager

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('logs/blockchain_auditor.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class Block:
    """
    Represents a single block in the blockchain.
    """

    def __init__(self, index: int, timestamp: str, data: Dict[str, Any], previous_hash: str):
        self.index = index
        self.timestamp = timestamp
        self.data = data  # Data to store in the block (e.g., audit logs)
        self.previous_hash = previous_hash  # Hash of the previous block
        self.hash = self.calculate_hash()  # Hash of the current block

    def calculate_hash(self) -> str:
        """
        Calculates the SHA-256 hash of the block's contents.
        """
        block_string = json.dumps({
            'index': self.index,
            'timestamp': self.timestamp,
            'data': self.data,
            'previous_hash': self.previous_hash
        }, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

class Blockchain:
    """
    Represents the blockchain containing a list of blocks.
    """

    def __init__(self):
        self.chain: List[Block] = []
        self.create_genesis_block()

    def create_genesis_block(self):
        """
        Creates the genesis block and appends it to the chain.
        """
        genesis_block = Block(
            index=0,
            timestamp=str(datetime.datetime.utcnow()),
            data={'message': 'Genesis Block'},
            previous_hash='0'
        )
        self.chain.append(genesis_block)
        logger.info("Genesis block created.")

    def get_latest_block(self) -> Block:
        """
        Retrieves the latest block in the chain.
        """
        return self.chain[-1]

    def add_block(self, data: Dict[str, Any]):
        """
        Adds a new block to the chain with the given data.
        """
        latest_block = self.get_latest_block()
        new_block = Block(
            index=latest_block.index + 1,
            timestamp=str(datetime.datetime.utcnow()),
            data=data,
            previous_hash=latest_block.hash
        )
        self.chain.append(new_block)
        logger.info(f"Block added to blockchain: Index {new_block.index}, Hash {new_block.hash}")

    def is_chain_valid(self) -> bool:
        """
        Validates the integrity of the blockchain.
        """
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]

            # Check if current block's hash is valid
            if current_block.hash != current_block.calculate_hash():
                logger.error(f"Invalid hash at block {current_block.index}")
                return False

            # Check if current block's previous hash matches previous block's hash
            if current_block.previous_hash != previous_block.hash:
                logger.error(f"Invalid previous hash at block {current_block.index}")
                return False

        logger.info("Blockchain is valid.")
        return True

    def to_json(self) -> str:
        """
        Serializes the blockchain to a JSON-formatted string.
        """
        chain_data = []
        for block in self.chain:
            chain_data.append({
                'index': block.index,
                'timestamp': block.timestamp,
                'data': block.data,
                'previous_hash': block.previous_hash,
                'hash': block.hash
            })
        return json.dumps(chain_data, indent=4)

class BlockchainAuditor:
    """
    Blockchain Integration
    Implements blockchain technology for immutable logging and auditing purposes,
    ensuring that Hermod’s processes and data are tracked transparently and securely.
    """

    def __init__(self):
        """
        Initializes the BlockchainAuditor with necessary configurations.
        """
        try:
            self.config_manager = ConfigurationManager()
            self.notification_manager = NotificationManager()
            self.load_auditor_config()
            self.blockchain = Blockchain()
            logger.info("BlockchainAuditor initialized successfully.")
        except Exception as e:
            logger.exception(f"Failed to initialize BlockchainAuditor: {e}")
            raise e

    def load_auditor_config(self):
        """
        Loads auditor configurations from the configuration manager or environment variables.
        """
        logger.info("Loading auditor configurations.")
        try:
            self.auditor_config = {
                'blockchain_file': self.config_manager.get('BLOCKCHAIN_FILE', 'data/blockchain.json'),
                'notification_recipients': self.config_manager.get('NOTIFICATION_RECIPIENTS', '').split(','),
            }
            logger.info(f"Auditor configurations loaded: {self.auditor_config}")
        except Exception as e:
            logger.error(f"Failed to load auditor configurations: {e}")
            raise e

    def log_event(self, data: Dict[str, Any]):
        """
        Logs an event by adding a new block to the blockchain.

        :param data: The data to log in the block.
        """
        try:
            self.blockchain.add_block(data)
            self.save_blockchain()
            logger.info("Event logged to blockchain.")
        except Exception as e:
            logger.error(f"Failed to log event: {e}")
            self.send_notification(
                subject="Blockchain Logging Failed",
                message=f"An error occurred while logging an event to the blockchain:\n\n{e}"
            )
            raise e

    def save_blockchain(self):
        """
        Saves the blockchain to a file.
        """
        try:
            blockchain_file = self.auditor_config['blockchain_file']
            os.makedirs(os.path.dirname(blockchain_file), exist_ok=True)
            with open(blockchain_file, 'w') as f:
                f.write(self.blockchain.to_json())
            logger.info(f"Blockchain saved to '{blockchain_file}'.")
        except Exception as e:
            logger.error(f"Failed to save blockchain: {e}")
            raise e

    def load_blockchain(self):
        """
        Loads the blockchain from a file.
        """
        try:
            blockchain_file = self.auditor_config['blockchain_file']
            if os.path.exists(blockchain_file):
                with open(blockchain_file, 'r') as f:
                    chain_data = json.load(f)
                self.blockchain.chain = []
                for block_data in chain_data:
                    block = Block(
                        index=block_data['index'],
                        timestamp=block_data['timestamp'],
                        data=block_data['data'],
                        previous_hash=block_data['previous_hash']
                    )
                    block.hash = block_data['hash']
                    self.blockchain.chain.append(block)
                logger.info(f"Blockchain loaded from '{blockchain_file}'.")
            else:
                logger.warning(f"Blockchain file '{blockchain_file}' does not exist. Creating a new blockchain.")
                self.blockchain = Blockchain()
                self.save_blockchain()
        except Exception as e:
            logger.error(f"Failed to load blockchain: {e}")
            raise e

    def validate_blockchain(self) -> bool:
        """
        Validates the integrity of the blockchain.
        """
        try:
            is_valid = self.blockchain.is_chain_valid()
            if not is_valid:
                self.send_notification(
                    subject="Blockchain Validation Failed",
                    message="The blockchain integrity check failed. Data may have been tampered with."
                )
            return is_valid
        except Exception as e:
            logger.error(f"Failed to validate blockchain: {e}")
            raise e

    def send_notification(self, subject: str, message: str):
        """
        Sends a notification to the configured recipients.

        :param subject: Subject of the notification.
        :param message: Body of the notification.
        """
        try:
            recipients = self.auditor_config['notification_recipients']
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
    Demonstrates example usage of the BlockchainAuditor class.
    """
    try:
        # Initialize BlockchainAuditor
        auditor = BlockchainAuditor()

        # Load existing blockchain or create a new one
        auditor.load_blockchain()

        # Log an event
        event_data = {
            'event_type': 'UserLogin',
            'user_id': 'user123',
            'timestamp': str(datetime.datetime.utcnow()),
            'details': 'User logged in successfully.'
        }
        auditor.log_event(event_data)

        # Validate the blockchain
        is_valid = auditor.validate_blockchain()
        logger.info(f"Blockchain valid: {is_valid}")

        # Access the blockchain data
        print("Blockchain Data:")
        print(auditor.blockchain.to_json())

    except Exception as e:
        logger.exception(f"Error in example usage: {e}")

# --------------------- Main Execution --------------------- #

if __name__ == "__main__":
    # Run the blockchain auditor example
    example_usage()
