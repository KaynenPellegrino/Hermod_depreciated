# src/modules/disaster_recovery/recovery_system.py

import os
import logging
import shutil
from datetime import datetime
from typing import List, Dict, Any, Optional
import subprocess
import threading
import time

from src.utils.configuration_manager import ConfigurationManager
from src.modules.disaster_recovery.backup_manager import BackupManager
from src.modules.notifications.notification_manager import NotificationManager

# Configure logging with RotatingFileHandler to prevent log files from becoming too large
from logging.handlers import RotatingFileHandler

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = RotatingFileHandler(
    os.path.join('logs', 'recovery_system.log'),
    maxBytes=5 * 1024 * 1024,  # 5 MB
    backupCount=5
)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)


class RecoverySystem:
    """
    Provides tools and processes to restore systems and data after a disaster or failure.
    Includes methods for data restoration, system reconfiguration, and validation of recovery procedures.
    """

    def __init__(self):
        """
        Initializes the RecoverySystem with necessary configurations.
        """
        try:
            self.config_manager = ConfigurationManager()
            self.backup_manager = BackupManager()
            self.notification_manager = NotificationManager()
            self.load_recovery_config()
            logger.info("RecoverySystem initialized successfully.")
        except Exception as e:
            logger.exception(f"Failed to initialize RecoverySystem: {e}")
            raise e

    def load_recovery_config(self):
        """
        Loads recovery configurations from the configuration manager or environment variables.
        """
        logger.info("Loading recovery configurations.")
        try:
            # Load configurations from a configuration file or environment variables
            self.recovery_config = {
                'backup_storage_location': self.config_manager.get('BACKUP_STORAGE_LOCATION', 'backups/'),
                'critical_services': self.config_manager.get('CRITICAL_SERVICES', ['webserver', 'database']),
                'recovery_steps': self.config_manager.get('RECOVERY_STEPS', ['restore_data', 'reconfigure_system', 'validate_recovery']),
                'alert_recipients': os.getenv('ALERT_RECIPIENTS', '').split(','),
            }
            logger.info(f"Recovery configurations loaded: {self.recovery_config}")
        except Exception as e:
            logger.error(f"Failed to load recovery configurations: {e}")
            raise e

    def initiate_recovery(self, backup_timestamp: Optional[str] = None):
        """
        Initiates the disaster recovery process.

        :param backup_timestamp: Timestamp of the backup to restore (format: 'YYYYMMDD_HHMMSS').
                                 If None, uses the latest backup.
        """
        logger.info("Initiating disaster recovery process.")
        try:
            if backup_timestamp is None:
                backup_timestamp = self.get_latest_backup_timestamp()
                if backup_timestamp is None:
                    logger.error("No backups available to restore.")
                    raise FileNotFoundError("No backups available to restore.")

            logger.info(f"Using backup timestamp: {backup_timestamp}")

            # Perform recovery steps
            for step in self.recovery_config['recovery_steps']:
                if hasattr(self, step):
                    getattr(self, step)(backup_timestamp)
                else:
                    logger.warning(f"Recovery step '{step}' is not defined.")

            # Send success notification
            self.notification_manager.send_notification(
                recipients=self.recovery_config['alert_recipients'],
                subject="Disaster Recovery Completed Successfully",
                message=f"Disaster recovery completed successfully using backup from {backup_timestamp}.",
            )

            logger.info("Disaster recovery process completed successfully.")
        except Exception as e:
            logger.error(f"Disaster recovery process failed: {e}")
            # Send failure notification
            self.notification_manager.send_notification(
                recipients=self.recovery_config['alert_recipients'],
                subject="Disaster Recovery Failed",
                message=f"Disaster recovery failed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.\nError: {e}",
            )
            raise e

    def get_latest_backup_timestamp(self) -> Optional[str]:
        """
        Retrieves the latest backup timestamp from the backup storage location.

        :return: Latest backup timestamp as a string or None if no backups are found.
        """
        logger.info("Retrieving the latest backup timestamp.")
        try:
            backup_root = self.recovery_config['backup_storage_location']
            backups = [d for d in os.listdir(backup_root) if os.path.isdir(os.path.join(backup_root, d))]
            backups.sort(reverse=True)
            if backups:
                latest_backup = backups[0]
                logger.info(f"Latest backup found: {latest_backup}")
                return latest_backup
            else:
                logger.warning("No backups found in backup storage location.")
                return None
        except Exception as e:
            logger.error(f"Failed to retrieve the latest backup timestamp: {e}")
            return None

    def restore_data(self, backup_timestamp: str):
        """
        Restores data from the specified backup timestamp.

        :param backup_timestamp: Timestamp of the backup to restore.
        """
        logger.info(f"Restoring data from backup timestamp: {backup_timestamp}")
        try:
            self.backup_manager.restore_backup(backup_timestamp)
            logger.info("Data restoration completed successfully.")
        except Exception as e:
            logger.error(f"Data restoration failed: {e}")
            raise e

    def reconfigure_system(self, backup_timestamp: str):
        """
        Reconfigures the system after data restoration.

        :param backup_timestamp: Timestamp of the backup used for restoration.
        """
        logger.info("Reconfiguring system after data restoration.")
        try:
            # Example: Reapply configurations, restart services, etc.
            self.apply_system_configurations()
            self.restart_services()
            logger.info("System reconfiguration completed successfully.")
        except Exception as e:
            logger.error(f"System reconfiguration failed: {e}")
            raise e

    def validate_recovery(self, backup_timestamp: str):
        """
        Validates the recovery procedures to ensure systems are operational.

        :param backup_timestamp: Timestamp of the backup used for restoration.
        """
        logger.info("Validating recovery procedures.")
        try:
            all_services_running = self.check_services_status()
            if all_services_running:
                logger.info("All critical services are running. Recovery validation successful.")
            else:
                logger.error("One or more critical services are not running. Recovery validation failed.")
                raise Exception("Recovery validation failed due to services not running.")
        except Exception as e:
            logger.error(f"Recovery validation failed: {e}")
            raise e

    def apply_system_configurations(self):
        """
        Applies system configurations as part of system reconfiguration.
        """
        logger.info("Applying system configurations.")
        try:
            # Load configuration files from a predefined location or backup
            config_files = self.config_manager.get('SYSTEM_CONFIG_FILES', [])
            for config_file in config_files:
                source_path = os.path.join('config_backup/', config_file)
                destination_path = os.path.join('config/', config_file)
                if os.path.exists(source_path):
                    shutil.copy2(source_path, destination_path)
                    logger.info(f"Configuration file '{config_file}' restored successfully.")
                else:
                    logger.warning(f"Configuration file '{config_file}' not found in backup.")

            logger.info("System configurations applied successfully.")
        except Exception as e:
            logger.error(f"Failed to apply system configurations: {e}")
            raise e

    def restart_services(self):
        """
        Restarts critical services after restoration.
        """
        logger.info("Restarting critical services.")
        try:
            services = self.recovery_config['critical_services']
            for service in services:
                self.restart_service(service)
            logger.info("All critical services restarted successfully.")
        except Exception as e:
            logger.error(f"Failed to restart critical services: {e}")
            raise e

    def restart_service(self, service_name: str):
        """
        Restarts a single service.

        :param service_name: Name of the service to restart.
        """
        logger.info(f"Restarting service: {service_name}")
        try:
            # Platform-dependent service restart command
            if os.name == 'nt':  # Windows
                cmd = ['net', 'stop', service_name]
                subprocess.run(cmd, check=True)
                cmd = ['net', 'start', service_name]
                subprocess.run(cmd, check=True)
            else:  # Unix/Linux
                cmd = ['systemctl', 'restart', service_name]
                subprocess.run(cmd, check=True)
            logger.info(f"Service '{service_name}' restarted successfully.")
        except Exception as e:
            logger.error(f"Failed to restart service '{service_name}': {e}")
            raise e

    def check_services_status(self) -> bool:
        """
        Checks the status of critical services.

        :return: True if all services are running, False otherwise.
        """
        logger.info("Checking status of critical services.")
        try:
            services = self.recovery_config['critical_services']
            all_services_running = True
            for service in services:
                if not self.is_service_running(service):
                    logger.error(f"Service '{service}' is not running.")
                    all_services_running = False
            return all_services_running
        except Exception as e:
            logger.error(f"Failed to check services status: {e}")
            raise e

    def is_service_running(self, service_name: str) -> bool:
        """
        Checks if a specific service is running.

        :param service_name: Name of the service.
        :return: True if service is running, False otherwise.
        """
        logger.info(f"Checking if service '{service_name}' is running.")
        try:
            # Platform-dependent service status command
            if os.name == 'nt':  # Windows
                cmd = ['sc', 'query', service_name]
                result = subprocess.run(cmd, capture_output=True, text=True)
                running = 'RUNNING' in result.stdout
            else:  # Unix/Linux
                cmd = ['systemctl', 'is-active', service_name]
                result = subprocess.run(cmd, capture_output=True, text=True)
                running = 'active' in result.stdout
            logger.info(f"Service '{service_name}' running: {running}")
            return running
        except Exception as e:
            logger.error(f"Failed to check status of service '{service_name}': {e}")
            return False

    # --------------------- Example Usage --------------------- #

    def example_usage(self):
        """
        Demonstrates example usage of the RecoverySystem class.
        """
        try:
            # Initialize RecoverySystem
            recovery_system = RecoverySystem()

            # Initiate disaster recovery process
            # Optionally specify a backup timestamp, or leave as None to use the latest backup
            recovery_system.initiate_recovery(backup_timestamp=None)

        except Exception as e:
            logger.exception(f"Error in example usage: {e}")

    # --------------------- Main Execution --------------------- #

    if __name__ == "__main__":
        # Run the recovery system example
        example_usage()
