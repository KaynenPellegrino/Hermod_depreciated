# hermod/core/code_generation.py

"""
Module: code_generation.py

Provides functions to generate code using OpenAI's API and to save the generated code to files.
"""

import os
import re
import openai
from dotenv import load_dotenv
from hermod.utils.logger import setup_logger

# Initialize logger
logger = setup_logger()

# Load environment variables from .env file
load_dotenv()

# Set up OpenAI API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    logger.error("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")
    raise ValueError("OpenAI API key is required.")

openai.api_key = OPENAI_API_KEY


def generate_code(prompt, model="gpt-4o-mini", max_tokens=1500):
    """
    Generates code based on the provided prompt using OpenAI's API.

    Args:
        prompt (str): The instruction or specification for code generation.
        model (str): The model to use for code generation.
        max_tokens (int): The maximum number of tokens to generate.

    Returns:
        str or None: The generated code, or None if an error occurred.
    """
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a code generator. Provide only the code without any explanations or formatting."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            n=1,
            stop=None,
            temperature=0,
        )
        raw_output = response['choices'][0]['message']['content'].strip()
        code = extract_code(raw_output)
        logger.info("Code generation successful.")
        return code
    except openai.error.OpenAIError as openai_error:
        logger.error("OpenAI API error: %s", openai_error)
        return None
    except (TimeoutError, ConnectionError) as e:
        logger.error("Network-related error: %s", e)
        return None
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        return None


def extract_code(text):
    """
    Extracts code from text, handling code blocks and mark down formatting.

    Args:
        text (str): The text containing code.

    Returns:
        str: Extracted code.
    """
    # Regex to extract code within triple backticks
    match = re.search(r'(?:\w+)?\n(.*?)', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return text.strip()


def save_code(code, filename, directory="generated_code"):
    """
    Saves the generated code to a file.

    Args:
        code (str): The code to save.
        filename (str): The name of the file.
        directory (str): The directory to save the file in.
    """
    os.makedirs(directory, exist_ok=True)  # Create directory if it doesn't exist
    filepath = os.path.join(directory, filename)

    try:
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(code)
        logger.info("Code saved to %s", filepath)
    except OSError as e:
        logger.error("Error saving code to %s: %s", filepath, e)
        raise