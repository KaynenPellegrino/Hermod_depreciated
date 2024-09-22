# tests/test_self_refactor.py

"""
Unit tests for the self_refactor module.
"""

import unittest
from unittest.mock import patch, mock_open
import os

from hermod.core.self_refactor import refactor_module
from hermod.utils.logger import logger

class TestSelfRefactor(unittest.TestCase):

    @patch('builtins.open', new_callable=mock_open, read_data='def old_function():\n    pass')
    @patch('hermod.core.self_refactor.generate_code', return_value='def new_function():\n    pass')
    @patch('hermod.core.self_refactor.logger.info')
    @patch('hermod.core.self_refactor.logger.error')
    def test_refactor_module_success(self, mock_logger_error, mock_logger_info, mock_generate_code, mock_file):
        module_path = 'hermod/core/code_generation.py'
        refactor_module(module_path)

        mock_file.assert_any_call(module_path, 'r', encoding='utf-8')
        mock_file.assert_any_call('hermod/core/code_generation_refactored.py', 'w', encoding='utf-8')
        mock_logger_info.assert_called_with(f"Refactored code saved to hermod/core/code_generation_refactored.py for review.")

    @patch('builtins.open', side_effect=FileNotFoundError)
    @patch('hermod.core.self_refactor.logger.error')
    def test_refactor_module_file_not_found(self, mock_logger_error, mock_file):
        module_path = 'hermod/core/nonexistent.py'
        refactor_module(module_path)

        mock_file.assert_called_once_with(module_path, 'r', encoding='utf-8')
        mock_logger_error.assert_called_once_with(f"Module {module_path} not found.")

    @patch('builtins.open', new_callable=mock_open, read_data='def old_function():\n    pass')
    @patch('hermod.core.self_refactor.generate_code', return_value=None)
    @patch('hermod.core.self_refactor.logger.error')
    def test_refactor_module_generate_code_failure(self, mock_logger_error, mock_generate_code, mock_file):
        module_path = 'hermod/core/code_generation.py'
        refactor_module(module_path)

        mock_file.assert_any_call(module_path, 'r', encoding='utf-8')
        mock_logger_error.assert_called_once_with(f"Failed to refactor module {module_path}")

if __name__ == '__main__':
    unittest.main()
