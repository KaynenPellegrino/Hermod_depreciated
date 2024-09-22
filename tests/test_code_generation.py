# tests/test_code_generation.py

"""
Unit tests for the code_generation module.
"""

import unittest
from unittest.mock import patch, mock_open
import os

from hermod.core.code_generation import generate_code, save_code
from hermod.utils.logger import logger


class TestCodeGeneration(unittest.TestCase):

    @patch('hermod.core.code_generation.openai.ChatCompletion.create')
    def test_generate_code_success(self, mock_chat_completion):
        # Setup the mock response
        mock_chat_completion.return_value = {
            'choices': [{'message': {'content': 'print("Hello, World!")'}}]
        }

        result = generate_code("Generate a simple hello world program.")
        self.assertEqual(result, 'print("Hello, World!")')

    @patch('hermod.core.code_generation.openai.ChatCompletion.create', side_effect=Exception("API Error"))
    def test_generate_code_api_error(self, mock_chat_completion):
        result = generate_code("Generate a simple hello world program.")
        self.assertIsNone(result)

    @patch('os.makedirs')
    @patch('builtins.open', new_callable=mock_open)
    @patch('hermod.core.code_generation.logger.info')
    def test_save_code_success(self, mock_logger_info, mock_file, mock_makedirs):
        code = 'print("Hello, World!")'
        filename = 'hello.py'
        directory = 'generated_code'

        save_code(code, filename, directory)

        mock_makedirs.assert_called_once_with(directory, exist_ok=True)
        mock_file.assert_called_once_with(os.path.join(directory, filename), 'w', encoding='utf-8')
        mock_logger_info.assert_called_once_with("Code saved to %s", os.path.join(directory, filename))

    @patch('os.makedirs', side_effect=OSError("File system error"))
    @patch('hermod.core.code_generation.logger.error')
    def test_save_code_os_error(self, mock_logger_error, mock_makedirs):
        code = 'print("Hello, World!")'
        filename = 'hello.py'
        directory = 'generated_code'

        with self.assertRaises(OSError):
            save_code(code, filename, directory)

        mock_makedirs.assert_called_once_with(directory, exist_ok=True)
        mock_logger_error.assert_called_once_with("Error saving code to %s: %s", os.path.join(directory, filename),
                                                  "File system error")


if __name__ == '__main__':
    unittest.main()
