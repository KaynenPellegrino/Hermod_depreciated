def store_and_update_tests(test_code, module_path):
    """
    Stores and updates the test cases for a module.

    Args:
        test_code (str): The generated test code.
        module_path (str): The module for which the tests were generated.
    """
    logger.info(f"Storing and updating tests for module: {module_path}")

    # Ensure the tests directory exists
    if not os.path.exists('tests'):
        os.makedirs('tests')

    # Save or update the test code to the appropriate test file
    module_name = os.path.basename(module_path)
    test_filename = os.path.join('tests', f"test_{module_name}")

    # If the test file already exists, append new test cases
    if os.path.exists(test_filename):
        with open(test_filename, 'a', encoding='utf-8') as f:
            f.write(test_code)
        logger.info(f"Updated test cases for {module_name} in {test_filename}")
    else:
        with open(test_filename, 'w', encoding='utf-8') as f:
            f.write(test_code)
        logger.info(f"Saved new test cases for {module_name} in {test_filename}")
