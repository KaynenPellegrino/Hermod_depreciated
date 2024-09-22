# hermod/core/deploy.py

"""
Module: deploy.py

Handles deployment of the generated projects.
"""

import os
import subprocess
from hermod.utils.logger import setup_logger

# Initialize logger
logger = setup_logger()

def deploy_project(project_path, deployment_type='local'):
    """
    Deploys the project based on the specified deployment type.

    Args:
        project_path (str): The path to the project directory.
        deployment_type (str): The type of deployment ('local', 'docker', 'cloud').
    """
    logger.info("Deploying project: %s with deployment type: %s", project_path, deployment_type)

    if deployment_type == 'local':
        # Simple local deployment: run the main application
        main_file = os.path.join(project_path, 'main.py')
        if os.path.exists(main_file):
            try:
                subprocess.Popen(['python', main_file], cwd=project_path)
                logger.info("Project deployed locally by running main.py.")
            except Exception as e:
                logger.error("Failed to deploy project locally: %s", e)
        else:
            logger.error("main.py not found in the project directory: %s", project_path)

    elif deployment_type == 'docker':
        # Deploy using Docker
        try:
            create_dockerfile(project_path)
            image_name = os.path.basename(project_path).lower()
            subprocess.run(['docker', 'build', '-t', image_name, '.'], cwd=project_path, check=True)
            subprocess.run(['docker', 'run', '-d', '--name', image_name, image_name], cwd=project_path, check=True)
            logger.info("Project deployed using Docker with image name: %s", image_name)
        except subprocess.CalledProcessError as e:
            logger.error("Docker deployment failed: %s", e)
        except Exception as e:
            logger.error("Unexpected error during Docker deployment: %s", e)

    elif deployment_type == 'cloud':
        # Placeholder for cloud deployment (e.g., AWS, GCP)
        logger.error("Cloud deployment is not yet implemented.")
        # Implement cloud deployment logic here
    else:
        logger.error("Unsupported deployment type: %s", deployment_type)

def create_dockerfile(project_path):
    """
    Creates a Dockerfile in the project directory.

    Args:
        project_path (str): The path to the project directory.
    """
    dockerfile_content = f"""
    FROM python:3.9-slim

    WORKDIR /app

    COPY . /app

    RUN pip install --upgrade pip
    RUN pip install -r requirements.txt

    CMD ["python", "main.py"]
    """

    dockerfile_path = os.path.join(project_path, 'Dockerfile')
    try:
        with open(dockerfile_path, 'w', encoding='utf-8') as f:
            f.write(dockerfile_content)
        logger.info("Dockerfile created at %s", dockerfile_path)
    except OSError as e:
        logger.error("Failed to create Dockerfile at %s: %s", dockerfile_path, e)
        raise
