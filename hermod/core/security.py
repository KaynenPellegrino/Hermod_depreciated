# hermod/core/security.py

"""
Module: security.py

Performs security scans on the project codebase and checks for sensitive information.
"""

import subprocess
import os
from hermod.utils.logger import setup_logger

# Initialize logger
logger = setup_logger()


def is_core_module(module_path):
    """
    Determines whether the given module is considered a core module.
    Core modules should not be modified automatically.
    """
    core_modules = ['hermod/core/security.py', 'hermod/core/version_control.py', 'hermod/core/internet_access.py']
    return module_path in core_modules


def module_path_contains_sensitive_info(module_path):
    """
    Checks if the module handles sensitive information. This function checks
    against a predefined list of sensitive modules that deal with security, data access, or
    compliance-related information.

    Args:
        module_path (str): Path to the module file to check.

    Returns:
        bool: True if the module contains or handles sensitive information, False otherwise.
    """
    sensitive_modules = [
        'hermod/core/data_collection.py',
        'hermod/core/security.py',
        'hermod/core/encryption.py',
        'hermod/core/authentication.py'
    ]

    return any(module_path.endswith(sensitive_module) for sensitive_module in sensitive_modules)


def security_check(module_path):
    """
    Perform a security check before allowing Hermod to modify code.
    Ensures that critical components or compliance boundaries are not violated.

    Args:
        module_path (str): The path to the module being modified.

    Returns:
        bool: True if the module can be modified safely, False otherwise.
    """
    # Check if the module is a core component or handles sensitive data
    if is_core_module(module_path) or module_path_contains_sensitive_info(module_path):
        logger.warning(f"Modification to critical or sensitive module {module_path} requires manual approval.")
        return False  # Require manual approval
    return True  # Safe to modify


def run_security_scan(project_path):
    """
    Runs security scans on the project using Bandit.

    Args:
        project_path (str): The path to the project directory.
    """
    logger.info("Running security scan on project: %s", project_path)

    try:
        result = subprocess.run(['bandit', '-r', project_path], capture_output=True, text=True, check=True)
        report_path = os.path.join('logs', 'security_scan_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(result.stdout)
        logger.info("Security scan completed. Report saved to %s", report_path)
    except subprocess.CalledProcessError as e:
        logger.error("Security scan failed: %s", e)
        report_path = os.path.join('logs', 'security_scan_error.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(e.output)
        logger.info("Security scan error report saved to %s", report_path)


def run_dependency_security_scan():
    result = subprocess.run(['safety', 'check'], capture_output=True, text=True)
    logger.info(f"Dependency security scan result: {result.stdout}")
