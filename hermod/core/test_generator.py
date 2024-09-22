import os
from hermod.core.code_generation import generate_code
from hermod.utils.logger import setup_logger

# Initialize logger
logger = setup_logger()

def generate_tests_for_module(module_path, runtime_behavior=None):
    """
    Generates tests either based on the module's code or its runtime behavior.

    Args:
        module_path (str): The path to the module to generate tests for.
        runtime_behavior (dict, optional): Runtime behavior metrics for the module.
    """
    logger.info(f"Generating tests for {module_path}...")

    if runtime_behavior:
        prompt = f"Generate unit tests based on the following runtime behavior:\n{runtime_behavior}"
    else:
        code = open(module_path, 'r').read()
        prompt = f"Generate unit tests for the following code:\n\n{code}"

    test_code = generate_code(prompt)

    if test_code:
        test_filename = os.path.join('tests', f"test_{os.path.basename(module_path)}.py")
        with open(test_filename, 'w', encoding='utf-8') as f:
            f.write(test_code)
        logger.info(f"Generated and saved tests for {module_path}.")
        return test_code
    else:
        logger.error(f"Failed to generate tests for {module_path}")
        return None
