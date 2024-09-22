# tests/test_deploy.py

"""
Unit tests for the deploy module.
"""

import unittest
from unittest.mock import patch, mock_open, MagicMock
import os
import subprocess

from hermod.core.deploy import deploy_project, create_dockerfile
from hermod.utils.logger import logger

class TestDeploy(unittest.TestCase):

    @patch('hermod.core.deploy.create_dockerfile')
    @patch('subprocess.run')
    @patch('hermod.core.deploy.logger.info')
    @patch('hermod.core.deploy.logger.error')
    def test_deploy_local_success(self, mock_logger_error, mock_logger_info, mock_subprocess_run, mock_create_dockerfile):
        project_path = 'TestProject'
        deployment_type = 'local'
        main_file = os.path.join(project_path, 'main.py')

        with patch('os.path.exists', return_value=True):
            with patch('subprocess.Popen') as mock_popen:
                deploy_project(project_path, deployment_type)
                mock_popen.assert_called_with(['python', main_file], cwd=project_path)
                mock_logger_info.assert_called_with("Project deployed locally by running main.py.")

    @patch('hermod.core.deploy.create_dockerfile')
    @patch('subprocess.run')
    @patch('hermod.core.deploy.logger.info')
    @patch('hermod.core.deploy.logger.error')
    def test_deploy_docker_success(self, mock_logger_error, mock_logger_info, mock_subprocess_run, mock_create_dockerfile):
        project_path = 'TestProject'
        deployment_type = 'docker'
        image_name = project_path.lower()

        with patch('os.path.exists', return_value=True):
            deploy_project(project_path, deployment_type)
            mock_create_dockerfile.assert_called_with(project_path)
            mock_subprocess_run.assert_any_call(['docker', 'build', '-t', image_name, '.'], cwd=project_path, check=True)
            mock_subprocess_run.assert_any_call(['docker', 'run', '-d', '--name', image_name, image_name], cwd=project_path, check=True)
            mock_logger_info.assert_called_with("Project deployed using Docker with image name: %s", image_name)

    @patch('hermod.core.deploy.logger.error')
    @patch('os.path.exists', return_value=False)
    def test_deploy_local_missing_main(self, mock_exists, mock_logger_error):
        project_path = 'TestProject'
        deployment_type = 'local'

        deploy_project(project_path, deployment_type)
        mock_logger_error.assert_called_with("main.py not found in the project directory: %s", project_path)

    @patch('hermod.core.deploy.create_dockerfile', side_effect=subprocess.CalledProcessError(1, 'cmd'))
    @patch('subprocess.run', side_effect=subprocess.CalledProcessError(1, 'cmd'))
    @patch('hermod.core.deploy.logger.error')
    def test_deploy_docker_failure(self, mock_logger_error, mock_subprocess_run, mock_create_dockerfile):
        project_path = 'TestProject'
        deployment_type = 'docker'

        with patch('os.path.exists', return_value=True):
            deploy_project(project_path, deployment_type)
            mock_logger_error.assert_called_with("Docker deployment failed: %s", unittest.mock.ANY)

    def test_create_dockerfile_success(self):
        project_path = 'TestProject'
        dockerfile_path = os.path.join(project_path, 'Dockerfile')
        dockerfile_content = """
    FROM python:3.9-slim

    WORKDIR /app

    COPY . /app

    RUN pip install --upgrade pip
    RUN pip install -r requirements.txt

    CMD ["python", "main.py"]
    """

        with patch('builtins.open', mock_open()) as mock_file:
            create_dockerfile(project_path)
            mock_file.assert_called_with(dockerfile_path, 'w', encoding='utf-8')
            mock_file().write.assert_called_with(dockerfile_content)

    def test_create_dockerfile_failure(self):
        project_path = 'TestProject'
        with patch('builtins.open', mock_open()) as mock_file:
            mock_file.side_effect = OSError("Permission denied")
            with self.assertRaises(OSError):
                create_dockerfile(project_path)

if __name__ == '__main__':
    unittest.main()
