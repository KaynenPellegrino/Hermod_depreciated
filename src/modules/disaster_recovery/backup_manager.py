# src/modules/disaster_recovery/backup_manager.py

import os
import shutil
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import threading
import schedule
import time

from src.utils.configuration_manager import ConfigurationManager
from src.modules.notifications.notification_manager import NotificationManager

# Database backup utilities
import subprocess
import pymongo
from dotenv import load_dotenv

# Configure logging with RotatingFileHandler to prevent log files from becoming too large
from logging.handlers import RotatingFileHandler

# Load environment variables
load_dotenv()

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = RotatingFileHandler(
    os.path.join('logs', 'backup_manager.log'),
    maxBytes=5 * 1024 * 1024,  # 5 MB
    backupCount=5
)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)

class BackupManager:
    """
    Handles regular backups of data, configurations, and critical components to prevent data loss.
    Manages backup schedules, storage locations, and retention policies.
    """

    def __init__(self):
        """
        Initializes the BackupManager with necessary configurations.
        """
        try:
            self.config_manager = ConfigurationManager()
            self.notification_manager = NotificationManager()
            self.load_backup_config()
            logger.info("BackupManager initialized successfully.")
        except Exception as e:
            logger.exception(f"Failed to initialize BackupManager: {e}")
            raise e

    def load_backup_config(self):
        """
        Loads backup configurations from the configuration manager or environment variables.
        """
        logger.info("Loading backup configurations.")
        try:
            # Load configurations from a configuration file or environment variables
            self.backup_config = {
                'backup_directories': self.config_manager.get('BACKUP_DIRECTORIES', [
                    'src/',
                    'config/',
                    'data/models/',
                    'src/modules/',
                ]),
                'backup_databases': self.config_manager.get('BACKUP_DATABASES', {
                    'postgresql': True,
                    'mongodb': True,
                }),
                'backup_schedule': self.config_manager.get('BACKUP_SCHEDULE', 'daily'),  # Options: 'hourly', 'daily', 'weekly'
                'backup_storage_location': self.config_manager.get('BACKUP_STORAGE_LOCATION', 'backups/'),
                'retention_policy_days': int(self.config_manager.get('RETENTION_POLICY_DAYS', 30)),
                'alert_recipients': os.getenv('ALERT_RECIPIENTS', '').split(','),
            }
            logger.info(f"Backup configurations loaded: {self.backup_config}")
        except Exception as e:
            logger.error(f"Failed to load backup configurations: {e}")
            raise e

    def schedule_backups(self):
        """
        Schedules backups based on the configured schedule.
        """
        logger.info("Scheduling backups.")
        schedule_interval = self.backup_config['backup_schedule'].lower()
        if schedule_interval == 'hourly':
            schedule.every().hour.do(self.run_backup)
        elif schedule_interval == 'daily':
            schedule.every().day.at("02:00").do(self.run_backup)  # Default time at 2 AM
        elif schedule_interval == 'weekly':
            schedule.every().monday.at("03:00").do(self.run_backup)  # Default time at 3 AM on Mondays
        else:
            logger.error(f"Invalid backup schedule interval: {schedule_interval}")
            raise ValueError(f"Invalid backup schedule interval: {schedule_interval}")

        # Start the scheduler in a separate thread
        threading.Thread(target=self.run_scheduler, daemon=True).start()
        logger.info(f"Backups scheduled to run {schedule_interval}.")

    def run_scheduler(self):
        """
        Runs the scheduler to execute scheduled tasks.
        """
        while True:
            schedule.run_pending()
            time.sleep(1)

    def run_backup(self):
        """
        Executes the backup process for directories and databases.
        """
        logger.info("Starting backup process.")
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_folder = os.path.join(self.backup_config['backup_storage_location'], timestamp)
            os.makedirs(backup_folder, exist_ok=True)

            # Backup directories
            for directory in self.backup_config['backup_directories']:
                self.backup_directory(directory, backup_folder)

            # Backup databases
            if self.backup_config['backup_databases'].get('postgresql', False):
                self.backup_postgresql(backup_folder)
            if self.backup_config['backup_databases'].get('mongodb', False):
                self.backup_mongodb(backup_folder)

            # Apply retention policy
            self.apply_retention_policy()

            # Send success notification
            self.notification_manager.send_notification(
                recipients=self.backup_config['alert_recipients'],
                subject="Backup Completed Successfully",
                message=f"Backup completed successfully at {timestamp}.",
            )

            logger.info("Backup process completed successfully.")
        except Exception as e:
            logger.error(f"Backup process failed: {e}")
            # Send failure notification
            self.notification_manager.send_notification(
                recipients=self.backup_config['alert_recipients'],
                subject="Backup Failed",
                message=f"Backup failed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.\nError: {e}",
            )

    def backup_directory(self, directory: str, backup_folder: str):
        """
        Backs up a directory by copying its contents to the backup folder.

        :param directory: Path to the directory to back up.
        :param backup_folder: Path to the backup destination folder.
        """
        logger.info(f"Backing up directory: {directory}")
        try:
            if not os.path.exists(directory):
                logger.warning(f"Directory '{directory}' does not exist and will be skipped.")
                return
            dest_path = os.path.join(backup_folder, os.path.basename(directory.rstrip('/\\')))
            shutil.copytree(directory, dest_path)
            logger.info(f"Directory '{directory}' backed up successfully to '{dest_path}'.")
        except Exception as e:
            logger.error(f"Failed to back up directory '{directory}': {e}")
            raise e

    def backup_postgresql(self, backup_folder: str):
        """
        Backs up the PostgreSQL database using pg_dump.

        :param backup_folder: Path to the backup destination folder.
        """
        logger.info("Backing up PostgreSQL database.")
        try:
            postgres_host = os.getenv('POSTGRES_HOST', 'localhost')
            postgres_port = os.getenv('POSTGRES_PORT', '5432')
            postgres_db = os.getenv('POSTGRES_DB', 'hermod_db')
            postgres_user = os.getenv('POSTGRES_USER', 'user')
            postgres_password = os.getenv('POSTGRES_PASSWORD', 'password')

            backup_file = os.path.join(backup_folder, f"{postgres_db}_backup.sql")
            os.environ['PGPASSWORD'] = postgres_password

            cmd = [
                'pg_dump',
                '-h', postgres_host,
                '-p', postgres_port,
                '-U', postgres_user,
                '-F', 'c',  # Custom format
                '-b',       # Include blobs
                '-v',       # Verbose mode
                '-f', backup_file,
                postgres_db
            ]

            subprocess.run(cmd, check=True)
            logger.info(f"PostgreSQL database '{postgres_db}' backed up successfully to '{backup_file}'.")
        except Exception as e:
            logger.error(f"Failed to back up PostgreSQL database: {e}")
            raise e

    def backup_mongodb(self, backup_folder: str):
        """
        Backs up the MongoDB database using mongodump.

        :param backup_folder: Path to the backup destination folder.
        """
        logger.info("Backing up MongoDB database.")
        try:
            mongodb_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
            mongodb_db_name = os.getenv('MONGODB_DB_NAME', 'hermod_db')

            backup_path = os.path.join(backup_folder, 'mongodb_backup')
            os.makedirs(backup_path, exist_ok=True)

            cmd = [
                'mongodump',
                '--uri', mongodb_uri,
                '--db', mongodb_db_name,
                '--out', backup_path
            ]

            subprocess.run(cmd, check=True)
            logger.info(f"MongoDB database '{mongodb_db_name}' backed up successfully to '{backup_path}'.")
        except Exception as e:
            logger.error(f"Failed to back up MongoDB database: {e}")
            raise e

    def apply_retention_policy(self):
        """
        Applies the retention policy by deleting backups older than the specified number of days.
        """
        logger.info("Applying retention policy.")
        try:
            retention_days = self.backup_config['retention_policy_days']
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            backup_root = self.backup_config['backup_storage_location']

            for folder_name in os.listdir(backup_root):
                folder_path = os.path.join(backup_root, folder_name)
                if os.path.isdir(folder_path):
                    folder_date = datetime.strptime(folder_name, '%Y%m%d_%H%M%S')
                    if folder_date < cutoff_date:
                        shutil.rmtree(folder_path)
                        logger.info(f"Deleted old backup '{folder_path}' as per retention policy.")
        except Exception as e:
            logger.error(f"Failed to apply retention policy: {e}")
            raise e

    def restore_backup(self, backup_timestamp: str):
        """
        Restores a backup from the specified timestamp.

        :param backup_timestamp: Timestamp of the backup to restore (format: 'YYYYMMDD_HHMMSS').
        """
        logger.info(f"Restoring backup from timestamp: {backup_timestamp}")
        try:
            backup_folder = os.path.join(self.backup_config['backup_storage_location'], backup_timestamp)
            if not os.path.exists(backup_folder):
                logger.error(f"Backup folder '{backup_folder}' does not exist.")
                raise FileNotFoundError(f"Backup folder '{backup_folder}' does not exist.")

            # Restore directories
            for directory in self.backup_config['backup_directories']:
                self.restore_directory(directory, backup_folder)

            # Restore databases
            if self.backup_config['backup_databases'].get('postgresql', False):
                self.restore_postgresql(backup_folder)
            if self.backup_config['backup_databases'].get('mongodb', False):
                self.restore_mongodb(backup_folder)

            logger.info(f"Backup from '{backup_timestamp}' restored successfully.")
        except Exception as e:
            logger.error(f"Failed to restore backup: {e}")
            raise e

    def restore_directory(self, directory: str, backup_folder: str):
        """
        Restores a directory from the backup folder.

        :param directory: Path to the directory to restore.
        :param backup_folder: Path to the backup folder.
        """
        logger.info(f"Restoring directory: {directory}")
        try:
            backup_dir_name = os.path.basename(directory.rstrip('/\\'))
            backup_dir_path = os.path.join(backup_folder, backup_dir_name)
            if not os.path.exists(backup_dir_path):
                logger.warning(f"Backup for directory '{directory}' does not exist in backup folder.")
                return
            # Remove current directory if it exists
            if os.path.exists(directory):
                shutil.rmtree(directory)
            shutil.copytree(backup_dir_path, directory)
            logger.info(f"Directory '{directory}' restored successfully from backup.")
        except Exception as e:
            logger.error(f"Failed to restore directory '{directory}': {e}")
            raise e

    def restore_postgresql(self, backup_folder: str):
        """
        Restores the PostgreSQL database from the backup.

        :param backup_folder: Path to the backup folder.
        """
        logger.info("Restoring PostgreSQL database from backup.")
        try:
            postgres_host = os.getenv('POSTGRES_HOST', 'localhost')
            postgres_port = os.getenv('POSTGRES_PORT', '5432')
            postgres_db = os.getenv('POSTGRES_DB', 'hermod_db')
            postgres_user = os.getenv('POSTGRES_USER', 'user')
            postgres_password = os.getenv('POSTGRES_PASSWORD', 'password')

            backup_file = os.path.join(backup_folder, f"{postgres_db}_backup.sql")
            if not os.path.exists(backup_file):
                logger.error(f"Backup file '{backup_file}' does not exist.")
                raise FileNotFoundError(f"Backup file '{backup_file}' does not exist.")

            os.environ['PGPASSWORD'] = postgres_password

            # Drop and recreate the database
            drop_cmd = [
                'psql',
                '-h', postgres_host,
                '-p', postgres_port,
                '-U', postgres_user,
                '-c', f"DROP DATABASE IF EXISTS {postgres_db}; CREATE DATABASE {postgres_db};"
            ]
            subprocess.run(drop_cmd, check=True)

            # Restore the database
            restore_cmd = [
                'pg_restore',
                '-h', postgres_host,
                '-p', postgres_port,
                '-U', postgres_user,
                '-d', postgres_db,
                '-v',
                backup_file
            ]
            subprocess.run(restore_cmd, check=True)
            logger.info(f"PostgreSQL database '{postgres_db}' restored successfully from backup.")
        except Exception as e:
            logger.error(f"Failed to restore PostgreSQL database: {e}")
            raise e

    def restore_mongodb(self, backup_folder: str):
        """
        Restores the MongoDB database from the backup.

        :param backup_folder: Path to the backup folder.
        """
        logger.info("Restoring MongoDB database from backup.")
        try:
            mongodb_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
            mongodb_db_name = os.getenv('MONGODB_DB_NAME', 'hermod_db')

            backup_path = os.path.join(backup_folder, 'mongodb_backup', mongodb_db_name)
            if not os.path.exists(backup_path):
                logger.error(f"Backup for MongoDB database '{mongodb_db_name}' does not exist in backup folder.")
                raise FileNotFoundError(f"Backup for MongoDB database '{mongodb_db_name}' does not exist.")

            # Drop the existing database
            client = pymongo.MongoClient(mongodb_uri)
            client.drop_database(mongodb_db_name)

            # Restore the database
            cmd = [
                'mongorestore',
                '--uri', mongodb_uri,
                '--db', mongodb_db_name,
                '--dir', backup_path
            ]
            subprocess.run(cmd, check=True)
            logger.info(f"MongoDB database '{mongodb_db_name}' restored successfully from backup.")
        except Exception as e:
            logger.error(f"Failed to restore MongoDB database: {e}")
            raise e

    # --------------------- Example Usage --------------------- #

    def example_usage(self):
        """
        Demonstrates example usage of the BackupManager class.
        """
        try:
            # Initialize BackupManager
            backup_manager = BackupManager()

            # Schedule backups
            backup_manager.schedule_backups()

            # Keep the script running to allow scheduled backups to occur
            logger.info("BackupManager is running. Press Ctrl+C to exit.")
            while True:
                time.sleep(1)

        except KeyboardInterrupt:
            logger.info("BackupManager stopped by user.")
        except Exception as e:
            logger.exception(f"Error in example usage: {e}")

    # --------------------- Main Execution --------------------- #

    if __name__ == "__main__":
        # Run the backup manager example
        example_usage()
