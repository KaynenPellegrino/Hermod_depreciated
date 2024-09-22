# tests/test_auto_debugger.py

"""
Unit tests for the auto_debugger module.
"""

import unittest
from unittest.mock import patch, mock_open
import os

from hermod.core.auto_debugger import analyze_logs_and_suggest_fixes
from hermod.utils.logger import logger

class TestAutoDebugger(unittest.TestCase):

    @patch('builtins.open', new_callable=mock_open, read_data="Error: Something went wrong")
    @patch('hermod.core.auto_debugger.generate_code', return_value='def fix(): pass')
    @patch('hermod.core.auto_debugger.logger.info')
    @patch('hermod.core.auto_debugger.logger.error')
    def test_analyze_logs_and_suggest_fixes_success(self, mock_logger_error, mock_logger_info, mock_generate_code, mock_file):
        log_file = 'logs/hermod.log'
        analyze_logs_and_suggest_fixes(log_file)

        mock_file.assert_called_once_with(log_file, 'r', encoding='utf-8')
        mock_generate_code.assert_called_once()
        mock_logger_info.assert_any_call("Suggested fix:")
        mock_logger_info.assert_called_with('def fix(): pass')

    @patch('builtins.open', side_effect=FileNotFoundError)
    @patch('hermod.core.auto_debugger.logger.error')
    def test_analyze_logs_and_suggest_fixes_file_not_found(self, mock_logger_error, mock_file):
        log_file = 'logs/nonexistent.log'
        analyze_logs_and_suggest_fixes(log_file)

        mock_file.assert_called_once_with(log_file, 'r', encoding='utf-8')
        mock_logger_error.assert_called_once_with("Log file %s not found.", log_file)

    @patch('builtins.open', new_callable=mock_open, read_data="Error: Invalid input")
    @patch('hermod.core.auto_debugger.generate_code', return_value=None)
    @patch('hermod.core.auto_debugger.logger.error')
    def test_analyze_logs_and_suggest_fixes_generate_code_failure(self, mock_logger_error, mock_generate_code, mock_file):
        log_file = 'logs/hermod.log'
        analyze_logs_and_suggest_fixes(log_file)

        mock_file.assert_called_once_with(log_file, 'r', encoding='utf-8')
        mock_generate_code.assert_called_once()
        mock_logger_error.assert_called_once_with("Failed to generate a fix suggestion.")

if __name__ == '__main__':
    unittest.main()
