# src/modules/deployment/docker_manager.py

import os
import logging
import subprocess
from typing import Optional, Dict, Any, List
from src.utils.configuration_manager import ConfigurationManager
from src.modules.notifications.notification_manager import NotificationManager

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('logs/docker_manager.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class DockerManager:
    """
    Docker Container Management
    Handles the creation, management, and orchestration of Docker containers for deploying applications.
    Includes building Docker images, managing container registries, and running containers.
    """

    def __init__(self):
        """
        Initializes the DockerManager with necessary configurations.
        """
        try:
            self.config_manager = ConfigurationManager()
            self.notification_manager = NotificationManager()
            self.load_docker_config()
            logger.info("DockerManager initialized successfully.")
        except Exception as e:
            logger.exception(f"Failed to initialize DockerManager: {e}")
            raise e

    def load_docker_config(self):
        """
        Loads Docker configurations from the configuration manager or environment variables.
        """
        logger.info("Loading Docker configurations.")
        try:
            self.docker_config = {
                'dockerfile_path': self.config_manager.get('DOCKERFILE_PATH', 'Dockerfile'),
                'image_name': self.config_manager.get('IMAGE_NAME', 'hermod_app'),
                'image_tag': self.config_manager.get('IMAGE_TAG', 'latest'),
                'registry_url': self.config_manager.get('REGISTRY_URL', ''),
                'registry_username': self.config_manager.get('REGISTRY_USERNAME', ''),
                'registry_password': self.config_manager.get('REGISTRY_PASSWORD', ''),
                'container_name': self.config_manager.get('CONTAINER_NAME', 'hermod_container'),
                'docker_compose_file': self.config_manager.get('DOCKER_COMPOSE_FILE', 'docker-compose.yml'),
                'environment_variables': self.config_manager.get('ENVIRONMENT_VARIABLES', {}),
                'notification_recipients': self.config_manager.get('NOTIFICATION_RECIPIENTS', '').split(','),
            }
            logger.info(f"Docker configurations loaded: {self.docker_config}")
        except Exception as e:
            logger.error(f"Failed to load Docker configurations: {e}")
            raise e

    def build_image(self):
        """
        Builds the Docker image using the specified Dockerfile.
        """
        logger.info("Building Docker image.")
        try:
            dockerfile_path = self.docker_config['dockerfile_path']
            image_name = self.docker_config['image_name']
            image_tag = self.docker_config['image_tag']

            if not os.path.exists(dockerfile_path):
                raise FileNotFoundError(f"Dockerfile '{dockerfile_path}' not found.")

            command = [
                'docker', 'build',
                '-t', f"{image_name}:{image_tag}",
                '-f', dockerfile_path,
                '.'
            ]
            logger.info(f"Executing command: {' '.join(command)}")
            subprocess.run(command, check=True)
            logger.info("Docker image built successfully.")
        except Exception as e:
            logger.error(f"Docker image build failed: {e}")
            self.send_notification(
                subject="Docker Image Build Failed",
                message=f"Docker image build failed with the following error:\n\n{e}"
            )
            raise e

    def push_image(self):
        """
        Pushes the Docker image to the specified registry.
        """
        logger.info("Pushing Docker image to registry.")
        try:
            registry_url = self.docker_config['registry_url']
            registry_username = self.docker_config['registry_username']
            registry_password = self.docker_config['registry_password']
            image_name = self.docker_config['image_name']
            image_tag = self.docker_config['image_tag']

            if not registry_url:
                raise ValueError("Registry URL is not configured.")

            full_image_name = f"{registry_url}/{image_name}:{image_tag}"

            # Tag the image
            subprocess.run(
                ['docker', 'tag', f"{image_name}:{image_tag}", full_image_name],
                check=True
            )

            # Login to the registry
            if registry_username and registry_password:
                subprocess.run(
                    ['docker', 'login', registry_url, '-u', registry_username, '-p', registry_password],
                    check=True
                )

            # Push the image
            subprocess.run(['docker', 'push', full_image_name], check=True)
            logger.info("Docker image pushed to registry successfully.")
        except Exception as e:
            logger.error(f"Docker image push failed: {e}")
            self.send_notification(
                subject="Docker Image Push Failed",
                message=f"Docker image push failed with the following error:\n\n{e}"
            )
            raise e

    def run_container(self):
        """
        Runs a Docker container from the image.
        """
        logger.info("Running Docker container.")
        try:
            image_name = self.docker_config['image_name']
            image_tag = self.docker_config['image_tag']
            container_name = self.docker_config['container_name']
            env_vars = self.docker_config['environment_variables']

            # Prepare environment variables
            env_options = []
            for key, value in env_vars.items():
                env_options.extend(['-e', f"{key}={value}"])

            # Run the container
            command = [
                'docker', 'run', '-d',
                '--name', container_name,
            ] + env_options + [
                f"{image_name}:{image_tag}"
            ]
            logger.info(f"Executing command: {' '.join(command)}")
            subprocess.run(command, check=True)
            logger.info("Docker container is running.")
        except Exception as e:
            logger.error(f"Running Docker container failed: {e}")
            self.send_notification(
                subject="Docker Container Run Failed",
                message=f"Running Docker container failed with the following error:\n\n{e}"
            )
            raise e

    def stop_container(self):
        """
        Stops and removes the Docker container.
        """
        logger.info("Stopping Docker container.")
        try:
            container_name = self.docker_config['container_name']

            # Stop the container
            subprocess.run(['docker', 'stop', container_name], check=True)
            # Remove the container
            subprocess.run(['docker', 'rm', container_name], check=True)
            logger.info("Docker container stopped and removed successfully.")
        except Exception as e:
            logger.error(f"Stopping Docker container failed: {e}")
            raise e

    def deploy_with_compose(self):
        """
        Deploys services using Docker Compose.
        """
        logger.info("Deploying services with Docker Compose.")
        try:
            compose_file = self.docker_config['docker_compose_file']
            env = os.environ.copy()
            env.update(self.docker_config['environment_variables'])

            if not os.path.exists(compose_file):
                raise FileNotFoundError(f"Docker Compose file '{compose_file}' not found.")

            command = ['docker-compose', '-f', compose_file, 'up', '-d']
            logger.info(f"Executing command: {' '.join(command)}")
            subprocess.run(command, check=True, env=env)
            logger.info("Services deployed with Docker Compose successfully.")
        except Exception as e:
            logger.error(f"Docker Compose deployment failed: {e}")
            self.send_notification(
                subject="Docker Compose Deployment Failed",
                message=f"Docker Compose deployment failed with the following error:\n\n{e}"
            )
            raise e

    def remove_with_compose(self):
        """
        Stops and removes services deployed with Docker Compose.
        """
        logger.info("Removing services with Docker Compose.")
        try:
            compose_file = self.docker_config['docker_compose_file']
            env = os.environ.copy()
            env.update(self.docker_config['environment_variables'])

            if not os.path.exists(compose_file):
                raise FileNotFoundError(f"Docker Compose file '{compose_file}' not found.")

            command = ['docker-compose', '-f', compose_file, 'down']
            logger.info(f"Executing command: {' '.join(command)}")
            subprocess.run(command, check=True, env=env)
            logger.info("Services removed with Docker Compose successfully.")
        except Exception as e:
            logger.error(f"Removing services with Docker Compose failed: {e}")
            raise e

    def send_notification(self, subject: str, message: str):
        """
        Sends a notification to the configured recipients.

        :param subject: Subject of the notification.
        :param message: Body of the notification.
        """
        try:
            recipients = self.docker_config['notification_recipients']
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

    def example_usage(self):
        """
        Demonstrates example usage of the DockerManager class.
        """
        try:
            # Initialize DockerManager
            docker_manager = DockerManager()

            # Build the Docker image
            docker_manager.build_image()

            # Push the Docker image to registry
            docker_manager.push_image()

            # Run the Docker container
            docker_manager.run_container()

            # Deploy services with Docker Compose
            docker_manager.deploy_with_compose()

            # Stop and remove Docker container
            # docker_manager.stop_container()

            # Remove services with Docker Compose
            # docker_manager.remove_with_compose()

        except Exception as e:
            logger.exception(f"Error in example usage: {e}")

    # --------------------- Main Execution --------------------- #

    if __name__ == "__main__":
        # Run the Docker manager example
        example_usage()
