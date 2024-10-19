# src/modules/deployment/packager.py

import os
import logging
import subprocess
from typing import Dict, Any
from src.utils.configuration_manager import ConfigurationManager
from src.modules.notifications.notification_manager import NotificationManager

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('logs/packager.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class Packager:
    """
    Application Packaging Tool
    Packages applications and dependencies into deployable units, such as Docker images, Helm charts, or archives.
    Ensures that applications are packaged consistently for deployment.
    """

    def __init__(self):
        """
        Initializes the Packager with necessary configurations.
        """
        try:
            self.config_manager = ConfigurationManager()
            self.notification_manager = NotificationManager()
            self.load_packager_config()
            logger.info("Packager initialized successfully.")
        except Exception as e:
            logger.exception(f"Failed to initialize Packager: {e}")
            raise e

    def load_packager_config(self):
        """
        Loads packager configurations from the configuration manager or environment variables.
        """
        logger.info("Loading packager configurations.")
        try:
            self.packager_config = {
                'package_type': self.config_manager.get('PACKAGE_TYPE', 'docker'),
                'docker_image_name': self.config_manager.get('DOCKER_IMAGE_NAME', 'hermod_app'),
                'docker_image_tag': self.config_manager.get('DOCKER_IMAGE_TAG', 'latest'),
                'helm_chart_name': self.config_manager.get('HELM_CHART_NAME', 'hermod-app-chart'),
                'archive_name': self.config_manager.get('ARCHIVE_NAME', 'hermod_app.tar.gz'),
                'source_directory': self.config_manager.get('SOURCE_DIRECTORY', '.'),
                'build_commands': self.config_manager.get('BUILD_COMMANDS', []),
                'environment_variables': self.config_manager.get('ENVIRONMENT_VARIABLES', {}),
                'notification_recipients': self.config_manager.get('NOTIFICATION_RECIPIENTS', '').split(','),
            }
            logger.info(f"Packager configurations loaded: {self.packager_config}")
        except Exception as e:
            logger.error(f"Failed to load packager configurations: {e}")
            raise e

    def package_application(self):
        """
        Packages the application based on the specified package type.
        """
        logger.info("Packaging application.")
        try:
            package_type = self.packager_config['package_type'].lower()
            if package_type == 'docker':
                self.build_docker_image()
            elif package_type == 'helm':
                self.package_helm_chart()
            elif package_type == 'archive':
                self.create_archive()
            else:
                raise ValueError(f"Unsupported package type: {package_type}")
            logger.info("Application packaged successfully.")
        except Exception as e:
            logger.error(f"Application packaging failed: {e}")
            self.send_notification(
                subject="Application Packaging Failed",
                message=f"Application packaging failed with the following error:\n\n{e}"
            )
            raise e

    def build_docker_image(self):
        """
        Builds a Docker image for the application.
        """
        logger.info("Building Docker image.")
        try:
            image_name = self.packager_config['docker_image_name']
            image_tag = self.packager_config['docker_image_tag']
            source_directory = self.packager_config['source_directory']
            build_commands = self.packager_config['build_commands']
            env = os.environ.copy()
            env.update(self.packager_config['environment_variables'])

            # Run build commands if any
            if build_commands:
                for command in build_commands:
                    logger.info(f"Executing build command: {command}")
                    subprocess.run(command, shell=True, check=True, cwd=source_directory, env=env)

            # Build Docker image
            command = [
                'docker', 'build',
                '-t', f"{image_name}:{image_tag}",
                source_directory
            ]
            logger.info(f"Executing command: {' '.join(command)}")
            subprocess.run(command, check=True, env=env)
            logger.info("Docker image built successfully.")
        except Exception as e:
            logger.error(f"Docker image build failed: {e}")
            raise e

    def package_helm_chart(self):
        """
        Packages the application into a Helm chart.
        """
        logger.info("Packaging Helm chart.")
        try:
            helm_chart_name = self.packager_config['helm_chart_name']
            source_directory = self.packager_config['source_directory']
            env = os.environ.copy()
            env.update(self.packager_config['environment_variables'])

            # Ensure Helm is installed
            if subprocess.run(['helm', 'version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode != 0:
                raise EnvironmentError("Helm is not installed or not in PATH.")

            # Package Helm chart
            command = [
                'helm', 'package',
                helm_chart_name,
                '--destination', source_directory
            ]
            logger.info(f"Executing command: {' '.join(command)}")
            subprocess.run(command, check=True, cwd=source_directory, env=env)
            logger.info("Helm chart packaged successfully.")
        except Exception as e:
            logger.error(f"Helm chart packaging failed: {e}")
            raise e

    def create_archive(self):
        """
        Creates an archive of the application source code.
        """
        logger.info("Creating application archive.")
        try:
            archive_name = self.packager_config['archive_name']
            source_directory = self.packager_config['source_directory']
            env = os.environ.copy()
            env.update(self.packager_config['environment_variables'])

            # Create tar.gz archive
            command = [
                'tar', '-czvf',
                archive_name,
                '-C', source_directory,
                '.'
            ]
            logger.info(f"Executing command: {' '.join(command)}")
            subprocess.run(command, check=True, env=env)
            logger.info("Application archive created successfully.")
        except Exception as e:
            logger.error(f"Archive creation failed: {e}")
            raise e

    def send_notification(self, subject: str, message: str):
        """
        Sends a notification to the configured recipients.

        :param subject: Subject of the notification.
        :param message: Body of the notification.
        """
        try:
            recipients = self.packager_config['notification_recipients']
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
    Demonstrates example usage of the Packager class.
    """
    try:
        # Initialize Packager
        packager = Packager()

        # Package the application
        packager.package_application()

    except Exception as e:
        logger.exception(f"Error in example usage: {e}")

# --------------------- Main Execution --------------------- #

if __name__ == "__main__":
    # Run the packager example
    example_usage()
