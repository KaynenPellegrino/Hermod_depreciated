from hermod.core.metrics_collector import collect_performance_metrics
from hermod.core.self_refactor import refactor_module
from hermod.utils.logger import setup_logger

# Initialize logger
logger = setup_logger()


def analyze_and_adjust_strategy(module_path, current_strategy):
    """
    Analyze performance metrics and adjust the refactoring strategy accordingly.

    Args:
        module_path (str): The path to the module being analyzed.
        current_strategy (dict): The current refactoring strategy (e.g., number of iterations).

    Returns:
        dict: The updated refactoring strategy.
    """
    logger.info(f"Analyzing performance metrics for {module_path} to adjust strategy...")

    # Collect performance metrics
    metrics = collect_performance_metrics(module_path)

    # Define thresholds for adjustment
    cpu_threshold = 80  # CPU usage threshold
    complexity_threshold = 10  # Cyclomatic complexity threshold
    test_duration_threshold = 10  # Time to run tests in seconds

    # Adjust the refactoring strategy based on the metrics
    if metrics['cpu_usage'] > cpu_threshold:
        logger.info("CPU usage is high, reducing the number of refactoring iterations.")
        current_strategy['iterations'] = max(1, current_strategy['iterations'] - 1)  # Reduce iterations

    if metrics['complexity'] > complexity_threshold:
        logger.info("Code complexity is high, increasing the number of refactoring iterations.")
        current_strategy['iterations'] += 1  # Increase iterations

    if not metrics['test_passed'] or metrics['test_duration'] > test_duration_threshold:
        logger.info("Tests failed or took too long, considering more aggressive optimizations.")
        current_strategy['aggressive'] = True  # Apply more aggressive optimizations

    logger.info(f"Updated strategy: {current_strategy}")
    return current_strategy
