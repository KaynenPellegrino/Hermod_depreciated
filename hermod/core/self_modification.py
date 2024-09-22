from hermod.core.self_refactor import refactor_module_with_optimization
from hermod.core.security import security_check, run_security_scan
from hermod.core.test_manager import run_tests_and_validate
from hermod.core.performance_monitor import monitor_performance_after_refactor
import os
from hermod.core.version_control import rollback_changes
from hermod.utils.logger import setup_logger

# Initialize logger
logger = setup_logger()

def apply_code_changes(module_path, refactored_code):
    """
    Applies the code changes to the module and rolls back if there is an issue.

    Args:
        module_path (str): The path to the module.
        refactored_code (str): The refactored code to apply.
    """
    logger.info(f"Applying code changes to {module_path}...")
    try:
        with open(module_path, 'w', encoding='utf-8') as f:
            f.write(refactored_code)
        logger.info(f"Changes applied to {module_path}.")
    except Exception as e:
        logger.error(f"Failed to apply code changes to {module_path}: {e}")
        rollback_changes(module_path)



def can_modify_file(module_path):
    """
    Determines if a module is safe to modify. Returns True if safe, False otherwise.

    Args:
        module_path (str): The path to the module to analyze.

    Returns:
        bool: True if the file is safe to modify, False if it is a core module.
    """
    # Example: Protect core modules or files critical to Hermod's functionality
    protected_modules = ['hermod/main.py', 'hermod/core/security.py']
    if module_path in protected_modules:
        logger.warning(f"Modification of {module_path} is restricted.")
        return False
    return True


def attempt_self_modification(module_path, refactored_code, auto_approve=False):
    """
    Attempts to apply self-modification. If the changes fail, roll back.

    Args:
        module_path (str): The path to the module to modify.
        refactored_code (str): The new refactored code.
        auto_approve (bool): If True, apply changes without approval.
    """
    if not can_modify_file(module_path):
        logger.warning(f"Modification of {module_path} is not allowed.")
        return

    logger.info(f"Attempting self-modification on {module_path}...")

    try:
        with open(module_path, 'w', encoding='utf-8') as f:
            f.write(refactored_code)
        logger.info(f"Applied self-modification to {module_path}.")
    except Exception as e:
        logger.error(f"Failed to apply self-modification to {module_path}: {e}")
        rollback_changes(module_path)

def controlled_self_modification(module_path, auto_approve=False):
    """
    Hermod modifies a module with control mechanisms to ensure compliance and safety.
    """
    if not security_check(module_path):
        logger.warning(f"Modification blocked for {module_path}. Manual approval required.")
        return

    logger.info(f"Starting controlled self-modification for {module_path}...")
    refactor_module_with_optimization(module_path, auto_approve)

    # After modification, run security and performance checks
    if run_security_scan(module_path) and not monitor_performance_after_refactor(module_path)['degraded']:
        logger.info(f"Modification of {module_path} successful.")
    else:
        logger.error(f"Security or performance check failed. Rolling back changes.")
        rollback_changes(module_path)
