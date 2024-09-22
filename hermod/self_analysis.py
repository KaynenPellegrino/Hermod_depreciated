# hermod/self_analysis.py

"""
Module: self_analysis.py

Performs static code analysis on the Hermod codebase using pylint.
"""

import subprocess
import os

from hermod.core.approval_manager import apply_changes
from hermod.core.metrics_collector import collect_performance_metrics
from hermod.utils.logger import setup_logger

# Initialize logger
logger = setup_logger()


def detect_code_improvements(module_path):
    """
    Detects areas in the code that could benefit from improvements, such as performance optimizations or code refactoring.

    Args:
        module_path (str): The path to the module.

    Returns:
        str: Suggested code improvements or refactor suggestions.
    """
    logger.info(f"Detecting code improvements for {module_path}...")

    # Collect performance metrics
    metrics = collect_performance_metrics(module_path)

    suggestions = []

    # Check for performance improvements
    if metrics['cpu_usage'] > 80:
        suggestions.append("Consider optimizing high CPU usage sections.")

    if metrics['complexity'] > 10:
        suggestions.append("Code complexity is high, consider refactoring.")

    if not metrics['test_passed']:
        suggestions.append("Test failures detected, investigate possible bugs.")

    if not suggestions:
        return "No immediate improvements detected."

    return '\n'.join(suggestions)


def run_static_code_analysis():
    """
    Runs static code analysis on the Hermod codebase using pylint.
    """
    logger.info("Starting static code analysis using pylint...")
    code_directories = ['hermod/core', 'hermod/utils']

    for directory in code_directories:
        logger.info(f"Analyzing directory: {directory}")
        result = subprocess.run(['pylint', directory], capture_output=True, text=True)
        # Save the report to a file
        report_file = os.path.join('logs', f'pylint_report_{os.path.basename(directory)}.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(result.stdout)
        logger.info(f"Static analysis report saved to {report_file}")

        # Check for issues
        if result.returncode != 0:
            logger.warning(f"Issues found in {directory}. Check the report for details.")
        else:
            logger.info(f"No issues found in {directory}.")

def self_optimize_and_upgrade():
    """
    Hermod attempts to upgrade itself based on past refactorings and logged improvements.
    """
    changes = detect_code_improvements()
    if changes:
        apply_changes(changes)
        logger.info("Self-optimization applied.")


if __name__ == "__main__":
    run_static_code_analysis()
