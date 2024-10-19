# src/modules/deployment/deployment_engine.py

import os
import logging
import subprocess
from typing import Optional, Dict, Any, List
from src.utils.configuration_manager import ConfigurationManager
from src.modules.notifications.notification_manager import NotificationManager

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('logs/deployment_engine.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class DeploymentEngine:
    """
    Deployment Orchestration Engine
    Manages the deployment process across different environments, handling tasks like
    resource provisioning, configuration management, and deployment strategies
    (e.g., blue-green, canary deployments).
    """

    def __init__(self):
        """
        Initializes the DeploymentEngine with necessary configurations.
        """
        try:
            self.config_manager = ConfigurationManager()
            self.notification_manager = NotificationManager()
            self.load_deployment_config()
            logger.info("DeploymentEngine initialized successfully.")
        except Exception as e:
            logger.exception(f"Failed to initialize DeploymentEngine: {e}")
            raise e

    def load_deployment_config(self):
        """
        Loads deployment configurations from the configuration manager or environment variables.
        """
        logger.info("Loading deployment configurations.")
        try:
            self.deployment_config = {
                'environments': self.config_manager.get('ENVIRONMENTS', ['staging', 'production']),
                'current_environment': self.config_manager.get('CURRENT_ENVIRONMENT', 'staging'),
                'deployment_strategy': self.config_manager.get('DEPLOYMENT_STRATEGY', 'blue_green'),
                'resource_provisioning_script': self.config_manager.get(
                    'RESOURCE_PROVISIONING_SCRIPT', 'scripts/provision_resources.sh'
                ),
                'configuration_management_tool': self.config_manager.get(
                    'CONFIGURATION_MANAGEMENT_TOOL', 'Ansible'
                ),
                'deployment_scripts': self.config_manager.get('DEPLOYMENT_SCRIPTS', {
                    'blue_green': 'scripts/deploy_blue_green.sh',
                    'canary': 'scripts/deploy_canary.sh'
                }),
                'notification_recipients': self.config_manager.get('NOTIFICATION_RECIPIENTS', '').split(','),
                'environment_variables': self.config_manager.get('ENVIRONMENT_VARIABLES', {}),
            }
            logger.info(f"Deployment configurations loaded: {self.deployment_config}")
        except Exception as e:
            logger.error(f"Failed to load deployment configurations: {e}")
            raise e

    def provision_resources(self):
        """
        Provisions resources required for deployment.
        """
        logger.info("Provisioning resources.")
        try:
            script = self.deployment_config['resource_provisioning_script']
            env = os.environ.copy()
            env.update(self.deployment_config['environment_variables'])

            if not os.path.exists(script):
                raise FileNotFoundError(f"Resource provisioning script '{script}' not found.")

            subprocess.run(['bash', script], check=True, env=env)
            logger.info("Resource provisioning completed successfully.")
        except Exception as e:
            logger.error(f"Resource provisioning failed: {e}")
            self.send_notification(
                subject="Resource Provisioning Failed",
                message=f"Resource provisioning failed with the following error:\n\n{e}"
            )
            raise e

    def configure_environment(self):
        """
        Configures the environment using the specified configuration management tool.
        """
        logger.info("Configuring environment.")
        try:
            tool = self.deployment_config['configuration_management_tool'].lower()
            env = os.environ.copy()
            env.update(self.deployment_config['environment_variables'])

            if tool == 'ansible':
                playbook = self.config_manager.get('ANSIBLE_PLAYBOOK', 'ansible/site.yml')
                inventory = self.config_manager.get('ANSIBLE_INVENTORY', 'ansible/hosts')
                subprocess.run(
                    ['ansible-playbook', '-i', inventory, playbook],
                    check=True, env=env
                )
            elif tool == 'chef':
                # Implement Chef configuration commands here
                pass
            elif tool == 'puppet':
                # Implement Puppet configuration commands here
                pass
            else:
                raise ValueError(f"Unsupported configuration management tool: {tool}")

            logger.info("Environment configuration completed successfully.")
        except Exception as e:
            logger.error(f"Environment configuration failed: {e}")
            self.send_notification(
                subject="Environment Configuration Failed",
                message=f"Environment configuration failed with the following error:\n\n{e}"
            )
            raise e

    def deploy_application(self):
        """
        Deploys the application using the specified deployment strategy.
        """
        logger.info("Deploying application.")
        try:
            strategy = self.deployment_config['deployment_strategy']
            script = self.deployment_config['deployment_scripts'].get(strategy)

            if not script:
                raise ValueError(f"No deployment script configured for strategy '{strategy}'.")

            if not os.path.exists(script):
                raise FileNotFoundError(f"Deployment script '{script}' not found.")

            env = os.environ.copy()
            env.update(self.deployment_config['environment_variables'])

            subprocess.run(['bash', script], check=True, env=env)
            logger.info(f"Application deployed successfully using '{strategy}' strategy.")
            self.send_notification(
                subject="Application Deployed Successfully",
                message=f"The application was deployed successfully using the '{strategy}' strategy."
            )
        except Exception as e:
            logger.error(f"Application deployment failed: {e}")
            self.send_notification(
                subject="Application Deployment Failed",
                message=f"Application deployment failed with the following error:\n\n{e}"
            )
            raise e

    def execute_deployment(self):
        """
        Executes the deployment process: provision resources, configure environment, deploy application.
        """
        logger.info("Starting deployment process.")
        try:
            self.provision_resources()
            self.configure_environment()
            self.deploy_application()
            logger.info("Deployment process completed successfully.")
        except Exception as e:
            logger.error(f"Deployment process failed: {e}")
            raise e

    def send_notification(self, subject: str, message: str):
        """
        Sends a notification to the configured recipients.

        :param subject: Subject of the notification.
        :param message: Body of the notification.
        """
        try:
            recipients = self.deployment_config['notification_recipients']
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
        Demonstrates example usage of the DeploymentEngine class.
        """
        try:
            # Initialize DeploymentEngine
            deployment_engine = DeploymentEngine()

            # Execute the deployment process
            deployment_engine.execute_deployment()

        except Exception as e:
            logger.exception(f"Error in example usage: {e}")

    # --------------------- Main Execution --------------------- #

    if __name__ == "__main__":
        # Run the deployment engine example
        example_usage()
