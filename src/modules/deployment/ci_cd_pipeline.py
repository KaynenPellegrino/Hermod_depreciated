# src/modules/deployment/ci_cd_pipeline.py

import os
import logging
import subprocess
from typing import Optional, Dict, Any, List
from src.utils.configuration_manager import ConfigurationManager
from src.modules.notifications.notification_manager import NotificationManager

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('logs/ci_cd_pipeline.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class CICDPipeline:
    """
    Continuous Integration and Deployment Pipeline
    Implements the logic for automating the building, testing, and deployment of applications.
    Integrates with tools like Jenkins, GitHub Actions, or GitLab CI/CD to streamline the deployment process.
    """

    def __init__(self):
        """
        Initializes the CICDPipeline with necessary configurations.
        """
        try:
            self.config_manager = ConfigurationManager()
            self.notification_manager = NotificationManager()
            self.load_pipeline_config()
            logger.info("CICDPipeline initialized successfully.")
        except Exception as e:
            logger.exception(f"Failed to initialize CICDPipeline: {e}")
            raise e

    def load_pipeline_config(self):
        """
        Loads pipeline configurations from the configuration manager or environment variables.
        """
        logger.info("Loading pipeline configurations.")
        try:
            self.pipeline_config = {
                'repository_url': self.config_manager.get('REPOSITORY_URL', ''),
                'branch': self.config_manager.get('BRANCH', 'main'),
                'build_commands': self.config_manager.get('BUILD_COMMANDS', ['pip install -r requirements.txt']),
                'test_commands': self.config_manager.get('TEST_COMMANDS', ['pytest']),
                'deploy_commands': self.config_manager.get('DEPLOY_COMMANDS', ['bash deploy.sh']),
                'notification_recipients': self.config_manager.get('NOTIFICATION_RECIPIENTS', '').split(','),
                'working_directory': self.config_manager.get('WORKING_DIRECTORY', '/tmp/ci_cd_pipeline'),
                'environment_variables': self.config_manager.get('ENVIRONMENT_VARIABLES', {}),
            }
            logger.info(f"Pipeline configurations loaded: {self.pipeline_config}")
        except Exception as e:
            logger.error(f"Failed to load pipeline configurations: {e}")
            raise e

    def clone_repository(self):
        """
        Clones the repository to the working directory.
        """
        logger.info("Cloning repository.")
        try:
            repo_url = self.pipeline_config['repository_url']
            branch = self.pipeline_config['branch']
            working_dir = self.pipeline_config['working_directory']

            if not repo_url:
                raise ValueError("Repository URL is not configured.")

            if os.path.exists(working_dir):
                logger.info(f"Working directory '{working_dir}' already exists. Deleting it.")
                subprocess.run(['rm', '-rf', working_dir], check=True)

            subprocess.run(['git', 'clone', '-b', branch, repo_url, working_dir], check=True)
            logger.info(f"Repository cloned successfully to '{working_dir}'.")
        except Exception as e:
            logger.error(f"Failed to clone repository: {e}")
            raise e

    def run_commands(self, commands: List[str], stage: str):
        """
        Runs a list of shell commands in the working directory.

        :param commands: List of shell commands to run.
        :param stage: The stage of the pipeline (e.g., 'build', 'test', 'deploy').
        """
        logger.info(f"Running {stage} commands.")
        try:
            env = os.environ.copy()
            env.update(self.pipeline_config['environment_variables'])
            working_dir = self.pipeline_config['working_directory']

            for command in commands:
                logger.info(f"Executing command: {command}")
                subprocess.run(command, shell=True, check=True, cwd=working_dir, env=env)
                logger.info(f"Command executed successfully: {command}")

            logger.info(f"{stage.capitalize()} stage completed successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"{stage.capitalize()} command failed: {e}")
            self.send_notification(
                subject=f"CI/CD Pipeline {stage.capitalize()} Failed",
                message=f"The {stage} stage failed with the following error:\n\n{e}"
            )
            raise e

    def build(self):
        """
        Runs the build commands.
        """
        self.run_commands(self.pipeline_config['build_commands'], 'build')

    def test(self):
        """
        Runs the test commands.
        """
        self.run_commands(self.pipeline_config['test_commands'], 'test')

    def deploy(self):
        """
        Runs the deploy commands.
        """
        self.run_commands(self.pipeline_config['deploy_commands'], 'deploy')

    def execute_pipeline(self):
        """
        Executes the CI/CD pipeline stages: clone, build, test, deploy.
        """
        logger.info("Executing CI/CD pipeline.")
        try:
            self.clone_repository()
            self.build()
            self.test()
            self.deploy()
            logger.info("CI/CD pipeline executed successfully.")
            self.send_notification(
                subject="CI/CD Pipeline Execution Successful",
                message="The CI/CD pipeline completed successfully."
            )
        except Exception as e:
            logger.error(f"CI/CD pipeline execution failed: {e}")
            raise e

    def send_notification(self, subject: str, message: str):
        """
        Sends a notification to the configured recipients.

        :param subject: Subject of the notification.
        :param message: Body of the notification.
        """
        try:
            recipients = self.pipeline_config['notification_recipients']
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
        Demonstrates example usage of the CICDPipeline class.
        """
        try:
            # Initialize CICDPipeline
            ci_cd_pipeline = CICDPipeline()

            # Execute the pipeline
            ci_cd_pipeline.execute_pipeline()

        except Exception as e:
            logger.exception(f"Error in example usage: {e}")

    # --------------------- Main Execution --------------------- #

    if __name__ == "__main__":
        # Run the CI/CD pipeline example
        example_usage()
