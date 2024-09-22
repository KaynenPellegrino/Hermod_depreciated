import os
from hermod.core.security import run_security_scan
from hermod.core.version_control import rollback_changes
from hermod.utils.logger import setup_logger
from hermod.core.performance_monitor import monitor_performance_after_refactor
from hermod.core.test_manager import run_tests_and_validate
from hermod.core.code_generation import generate_code
from hermod.core.approval_manager import assess_risk_and_approve

# Initialize logger
logger = setup_logger()

def refactor_module(module_path, iterations=1, approval_required=False):
    """
    Refactors the specified Python module, optionally multiple times.
    Automatically applies changes if no approval is required.

    Args:
        module_path (str): The path to the module to refactor.
        iterations (int): Number of refactoring iterations to perform.
        approval_required (bool): Whether human approval is required for changes.

    Returns:
        str: The refactored code, or None if refactoring fails.
    """
    logger.info(f"Starting refactoring for module: {module_path}")
    try:
        with open(module_path, 'r', encoding='utf-8') as f:
            original_code = f.read()
    except FileNotFoundError:
        logger.error(f"Module {module_path} not found.")
        return None

    refactored_code = original_code
    for _ in range(iterations):
        prompt = f"Refactor the following Python code for performance and readability:\n\n{refactored_code}"
        refactored_code = generate_code(prompt)

        if not refactored_code:
            logger.error(f"Failed to refactor module {module_path}.")
            return None

    return refactored_code

def refactor_module_with_validation(module_path, auto_approve=False):
    """
    Refactors a module and validates the changes through tests and performance monitoring.
    """
    logger.info(f"Starting refactor process for {module_path}...")

    # Step 1: Refactor the module
    refactored_code = refactor_module(module_path, iterations=1, approval_required=not auto_approve)

    if not refactored_code:
        logger.error(f"Failed to refactor {module_path}")
        return

    # Step 2: Run tests on refactored code
    test_passed = run_tests_and_validate(module_path)

    if not test_passed:
        logger.error(f"Tests failed after refactoring {module_path}. Rolling back.")
        rollback_changes(module_path)
        return

    # Step 3: Monitor performance after refactor
    performance_metrics = monitor_performance_after_refactor(module_path)

    if performance_metrics.get('degraded'):
        logger.warning(f"Performance degradation detected in {module_path}. Rolling back changes.")
        rollback_changes(module_path)
    else:
        logger.info(f"Performance improvement detected in {module_path}. Refactor successful.")

def self_optimize_and_modify(module_path, auto_approve=False):
    """
    Combines optimization and self-modification into a unified function.
    """
    logger.info(f"Starting self-optimization for {module_path}...")

    if run_security_scan(module_path):
        refactored_code = refactor_module(module_path, iterations=2, approval_required=not auto_approve)

        if refactored_code:
            test_passed = run_tests_and_validate(module_path)
            if not test_passed:
                logger.error(f"Tests failed for {module_path}. Rolling back.")
                rollback_changes(module_path)
            else:
                logger.info(f"Optimization completed for {module_path}.")
        else:
            logger.error(f"Refactoring failed for {module_path}.")
    else:
        logger.warning(f"Security scan failed for {module_path}. Aborting.")

def refactor_module_with_optimization(module_path, auto_approve=False):
    """
    AI-based refactoring for the module with optional automatic approval for optimizations.
    """
    logger.info(f"Optimizing and refactoring module: {module_path}")
    refactored_code = refactor_module(module_path)

    if refactored_code:
        # Assess risk before applying changes
        critical_sections = []  # Define critical sections for your project
        risk_approved = assess_risk_and_approve(module_path, refactored_code, critical_sections)

        if risk_approved or auto_approve:
            logger.info(f"Automatically approved refactoring for {module_path}")
            with open(module_path, 'w', encoding='utf-8') as f:
                f.write(refactored_code)
        else:
            logger.info(f"Refactoring for {module_path} was not approved, skipping.")
    else:
        logger.error(f"Failed to generate refactored code for {module_path}")
