# tests/test_documentation.py

"""
Unit tests for the documentation module.
"""

import unittest
from unittest.mock import patch, mock_open
import os
import subprocess

from hermod.core.documentation import generate_documentation
from hermod.utils.logger import logger


class TestDocumentation(unittest.TestCase):

    @patch('os.makedirs')
    @patch('os.listdir', return_value=[])
    def test_create_docs_directory(self, mock_listdir, mock_makedirs):
        """
        Test that the docs directory is created if it doesn't exist.
        """
        project_path = "TestProject"

        # Run the function to generate documentation
        generate_documentation(project_path)

        # Ensure os.listdir was called to check for the docs directory
        mock_listdir.assert_called_once_with(os.path.join(project_path, 'docs'))

        # Ensure os.makedirs was called to create the docs directory
        mock_makedirs.assert_called_once_with(os.path.join(project_path, 'docs'), exist_ok=True)

    @patch('hermod.core.documentation.subprocess.run')
    @patch('hermod.core.documentation.os.path.exists', return_value=True)
    def test_generate_documentation_success(self, mock_exists, mock_run):
        """
        Test that documentation is generated successfully.
        """
        project_path = 'TestProject'
        docs_path = os.path.join(project_path, 'docs')

        generate_documentation(project_path)

        # Check if sphinx commands were called
        mock_run.assert_any_call(['sphinx-quickstart', '--quiet', '--project', 'HermodProject', '--author', 'Your Name', '--sep',
                                  '--makefile', '--no-batchfile'], cwd=docs_path, check=True)
        mock_run.assert_any_call(['sphinx-apidoc', '-o', 'docs/source', '../core'], cwd=project_path, check=True)
        mock_run.assert_any_call(['make', 'html'], cwd=docs_path, check=True)

    @patch('hermod.core.documentation.subprocess.run', side_effect=subprocess.CalledProcessError(1, 'cmd'))
    @patch('hermod.core.documentation.os.path.exists', return_value=True)
    def test_generate_documentation_initialization_failure(self, mock_exists, mock_run):
        """
        Test failure during Sphinx initialization.
        """
        project_path = 'TestProject'
        generate_documentation(project_path)

        mock_run.assert_called_with(['sphinx-quickstart', '--quiet', '--project', 'HermodProject', '--author', 'Your Name', '--sep',
                                     '--makefile', '--no-batchfile'], cwd=os.path.join(project_path, 'docs'), check=True)
        logger.error.assert_called_with("Failed to initialize Sphinx: %s", unittest.mock.ANY)

    @patch('hermod.core.documentation.subprocess.run', side_effect=subprocess.CalledProcessError(1, 'cmd'))
    @patch('hermod.core.documentation.os.path.exists', return_value=True)
    def test_generate_documentation_failure(self, mock_exists, mock_run):
        """
        Test failure during documentation generation.
        """
        project_path = 'TestProject'
        generate_documentation(project_path)

        mock_run.assert_called_with(['make', 'html'], cwd=os.path.join(project_path, 'docs'), check=True)
        logger.error.assert_called_with("Failed to generate documentation: %s", unittest.mock.ANY)

    @patch('hermod.core.documentation.os.path.exists', return_value=False)
    @patch('hermod.core.documentation.os.makedirs')
    def test_create_docs_directory(self, mock_makedirs, mock_exists):
        """
        Test that the docs directory is created if it doesn't exist.
        """
        project_path = 'TestProject'
        generate_documentation(project_path)

        docs_path = os.path.join(project_path, 'docs')
        mock_makedirs.assert_called_once_with(docs_path)


if __name__ == '__main__':
    unittest.main()
