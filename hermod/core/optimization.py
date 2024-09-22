import psutil
import time
from radon.complexity import cc_visit
import os

from tensorflow.tools.pip_package.setup import project_name

from hermod.core.performance_monitor import monitor_performance_after_refactor
from hermod.core.self_refactor import refactor_module
from hermod.utils.logger import setup_logger

# Initialize logger
logger = setup_logger()


def analyze_performance():
    """
    Analyze current system performance, including CPU and memory usage.
    """
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()

    logger.info(f"Current CPU usage: {cpu_usage}%")
    logger.info(f"Available memory: {memory_info.available / (1024 * 1024)} MB")

    return cpu_usage, memory_info


def analyze_complexity(module_path):
    """
    Analyze the complexity of the given module using Radon.
    """
    with open(module_path, 'r') as f:
        code = f.read()
    complexity_scores = cc_visit(code)
    logger.info(f"Complexity scores for {module_path}: {complexity_scores}")
    return complexity_scores


def apply_optimization(module_path):
    """
    Apply optimizations based on complexity and performance analysis.
    """
    logger.info(f"Applying optimization to {module_path}...")
    refactor_module(module_path, iterations=1, approval_required=False)
    logger.info(f"Optimization completed for {module_path}.")


import os
import logging


def self_optimize_and_correct(project_path):
    logger.info(f"Starting self-optimization and correction for {project_path}")

    # Define the log file location
    log_file = os.path.join('logs', 'hermod.log')

    # Define the modules directory
    modules_dir = os.path.join(project_path, 'core')

    # Check if the modules directory exists
    if not os.path.exists(modules_dir):
        logger.error("Modules directory not found.")
        return

    try:
        # Analyze logs for errors
        error_message = analyze_logs(log_file)

        # If an error is found, try to generate a fix
        if error_message:
            logger.info(f"Identified error: {error_message}")

            # Generate a code fix based on the error
            prompt = f"Fix the following error in the code: {error_message}"
            code_fix = generate_code(prompt)

            if code_fix:
                save_code(code_fix, 'main.py', directory=project_path)
                logger.info("Code fix applied successfully.")
            else:
                logger.error("Failed to generate a code fix. AI code generation might be returning null results.")
        else:
            logger.info("No errors found in logs.")

    except Exception as e:
        logger.error(f"Error during self-optimization: {e}")

    # Analyze system performance (CPU usage, memory)
    cpu_usage, memory_info = analyze_performance()

    # Define threshold values for triggering optimizations
    cpu_threshold = 80  # Optimize if CPU usage > 80%
    complexity_threshold = 10  # Optimize if cyclomatic complexity > 10

    # Iterate over Python files in the 'core' directory
    for module_file in os.listdir(modules_dir):
        if module_file.endswith('.py') and not module_file.startswith('__'):
            module_path = os.path.join(modules_dir, module_file)

            # Analyze the code complexity of the module
            complexity_scores = analyze_complexity(module_path)

            # Trigger optimization based on CPU usage or high complexity
            if cpu_usage > cpu_threshold or any(score.complexity > complexity_threshold for score in complexity_scores):
                logger.info(f"Optimization triggered for {module_path}")
                apply_optimization(module_path)
            else:
                logger.info(f"No optimization needed for {module_path} at this time.")

    # Log the completion of the optimization process
    logger.info("Self-optimization and correction completed.")


def self_optimize(module_path, auto_approve=False):
    """
    Self-optimizes the module based on performance and complexity metrics.

    Args:
        module_path (str): The path to the module to optimize.
        auto_approve (bool): Whether to automatically apply optimizations without approval.
    """
    logger.info(f"Starting self-optimization for {module_path}...")

    # Step 1: Monitor performance
    performance_metrics = monitor_performance_after_refactor(module_path)

    # Step 2: Check if optimization is needed
    if performance_metrics['degraded']:
        logger.warning(f"Performance degradation detected in {module_path}. Applying optimizations.")
        refactor_module(module_path, iterations=2, approval_required=not auto_approve)
    else:
        logger.info(f"No optimization needed for {module_path}. Performance is stable.")
