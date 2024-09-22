import psutil
import time
from hermod.core.test_manager import run_tests_and_validate
import os
from radon.complexity import cc_visit
from hermod.utils.logger import setup_logger

# Initialize logger
logger = setup_logger()

def count_lines_of_code(module_path):
    """
    Counts the total lines of code in the module.

    Args:
        module_path (str): The path to the module to analyze.

    Returns:
        int: Total number of lines of code.
    """
    logger.info(f"Counting lines of code for {module_path}")
    try:
        with open(module_path, 'r', encoding='utf-8') as f:
            code_lines = f.readlines()
        return len(code_lines)
    except FileNotFoundError:
        logger.error(f"Module {module_path} not found.")
        return 0

def get_module_metrics(module_path):
    """
    Collects initial metrics from a module, including cyclomatic complexity and test results.

    Args:
        module_path (str): The path to the module to analyze.

    Returns:
        dict: A dictionary containing the collected metrics.
    """
    logger.info(f"Collecting metrics for {module_path}...")

    # Measure cyclomatic complexity using Radon
    with open(module_path, 'r') as f:
        code = f.read()
    complexity_scores = cc_visit(code)
    total_complexity = sum(score.complexity for score in complexity_scores)

    # Run tests on the module and gather test results
    test_passed = run_tests_and_validate(module_path)

    # Collect additional metrics (e.g., file size)
    file_size = os.path.getsize(module_path)

    # Aggregate metrics
    metrics = {
        "complexity": total_complexity,
        "test_passed": test_passed,
        "file_size": file_size
    }

    logger.info(f"Collected metrics for {module_path}: {metrics}")
    return metrics


def collect_performance_metrics(module_path):
    """
    Collect performance metrics, including CPU, memory usage, and cyclomatic complexity.

    Args:
        module_path (str): The path to the module being analyzed.

    Returns:
        dict: A dictionary of performance metrics.
    """
    logger.info(f"Collecting performance metrics for {module_path}...")

    # Measure CPU and memory usage
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()

    # Measure cyclomatic complexity using Radon
    with open(module_path, 'r') as f:
        code = f.read()
    complexity_scores = cc_visit(code)
    complexity = sum(score.complexity for score in complexity_scores)

    # Measure test results and time taken
    start_time = time.time()
    test_passed = run_tests_and_validate(module_path)
    test_duration = time.time() - start_time

    # Aggregate metrics
    metrics = {
        "cpu_usage": cpu_usage,
        "memory_available": memory_info.available,
        "complexity": complexity,
        "test_passed": test_passed,
        "test_duration": test_duration
    }

    logger.info(f"Collected metrics for {module_path}: {metrics}")
    return metrics
