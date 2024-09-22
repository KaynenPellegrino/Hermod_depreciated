import subprocess
from hermod.utils.logger import setup_logger
from hermod.core.security import run_security_scan

# Initialize logger
logger = setup_logger()

def run_tests_and_validate(module_path):
    """
    Runs the tests for the specified module and validates the results.
    """
    logger.info(f"Running tests for module: {module_path}")
    test_results = run_tests_in_project(module_path)

    if test_results['passed']:
        logger.info(f"All tests passed for module: {module_path}")
    else:
        logger.error(f"Tests failed for {module_path}. Rolling back changes.")
        rollback_changes(module_path)

def run_tests_in_project(project_name):
    """
    Runs tests on all test files in the project directory and returns the results.
    """
    logger.info(f"Running tests for project: {project_name}")
    tests_dir = os.path.join(project_name, 'tests')

    if os.path.exists(tests_dir):
        result = subprocess.run(['python', '-m', 'unittest', 'discover', tests_dir], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("All tests passed.")
            return {"passed": True, "output": result.stdout}
        else:
            logger.error("Some tests failed.")
            return {"passed": False, "output": result.stderr}
    else:
        logger.warning(f"Tests directory {tests_dir} does not exist.")
        return {"passed": False, "output": "Tests directory not found."}
