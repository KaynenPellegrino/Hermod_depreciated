# tests/test_project_generator.py

"""
Unit tests for the project_generator module.
"""

import unittest
from unittest.mock import patch, mock_open
import os
import json

from hermod.core.project_generator import generate_project, parse_specification
from hermod.utils.logger import logger

class TestProjectGenerator(unittest.TestCase):

    @patch('hermod.core.project_generator.generate_code')
    def test_parse_specification_success(self, mock_generate_code):
        # Mock the generate_code response
        mock_generate_code.return_value = json.dumps([
            {"description": "A module for data processing.", "filename": "data_processing"},
            {"description": "A module for data visualization.", "filename": "visualization"},
            {"description": "A README file explaining the project.", "filename": "README"}
        ])

        specification = "Create a project for data analysis with modules for data processing and visualization."
        language = "Python"
        components = parse_specification(specification, language)
        self.assertEqual(len(components), 3)
        self.assertEqual(components[0]['filename'], 'data_processing')

    @patch('hermod.core.project_generator.generate_code', return_value=None)
    def test_parse_specification_failure(self, mock_generate_code):
        specification = "Some faulty specification."
        language = "Python"
        components = parse_specification(specification, language)
        self.assertEqual(components, [])

    @patch('hermod.core.project_generator.parse_specification')
    @patch('hermod.core.project_generator.generate_code')
    @patch('hermod.core.project_generator.save_code')
    @patch('os.makedirs')
    def test_generate_project_success(self, mock_makedirs, mock_save_code, mock_generate_code, mock_parse_spec):
        # Mock parse_specification
        mock_parse_spec.return_value = [
            {"description": "A module for data processing.", "filename": "data_processing"},
            {"description": "A module for data visualization.", "filename": "visualization"},
            {"description": "A README file explaining the project.", "filename": "README"}
        ]

        # Mock generate_code
        mock_generate_code.return_value = "print('Hello from data_processing')"

        specification = "Create a project for data analysis with modules for data processing and visualization."
        project_name = "DataAnalysisProject"
        language = "Python"

        generate_project(specification, project_name=project_name, language=language)

        mock_makedirs.assert_called_once_with(project_name, exist_ok=True)
        self.assertEqual(mock_save_code.call_count, 3)
        mock_save_code.assert_any_call("print('Hello from data_processing')", "data_processing.py", directory=project_name)
        mock_save_code.assert_any_call("print('Hello from data_processing')", "visualization.py", directory=project_name)
        mock_save_code.assert_any_call("print('Hello from data_processing')", "README.md", directory=project_name)

    @patch('hermod.core.project_generator.parse_specification', return_value=[])
    def test_generate_project_no_components(self, mock_parse_spec):
        specification = "Faulty specification that results in no components."
        project_name = "FaultyProject"
        language = "Python"

        generate_project(specification, project_name=project_name, language=language)
        # No exception should be raised, but a log should indicate failure

if __name__ == '__main__':
    unittest.main()
