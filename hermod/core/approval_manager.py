# In hermod/core/approval_manager.py

from hermod.utils.logger import setup_logger
from hermod.core.security import is_core_module, module_path_contains_sensitive_info

# Initialize logger
logger = setup_logger()

def assess_risk(changes, critical_sections):
    """
    Assesses the risk of the proposed changes. If the changes affect critical sections or sensitive data,
    they are considered high-risk.

    Args:
        changes (str): The proposed code changes.
        critical_sections (list): List of critical sections where approval is required.

    Returns:
        bool: True if the changes are considered low-risk and can be automatically approved, False otherwise.
    """
    logger.info("Assessing risk of changes...")

    # Placeholder logic for risk assessment. In a real implementation, this could be more complex,
    # involving static code analysis, security checks, or complexity analysis.

    # Example: Check if the changes affect critical sections
    for section in critical_sections:
        if section in changes:
            logger.warning("Changes affect critical section. Manual approval required.")
            return False  # High-risk, manual approval required

    # If the changes do not affect critical sections, consider them low-risk
    logger.info("Changes are low-risk. No manual approval needed.")
    return True

def assess_risk_and_approve(module_path, changes, critical_sections=[]):
    """
    Automatically approves low-risk changes, but requests approval for critical changes.

    Args:
        module_path (str): Path to the module.
        changes (str): The proposed code changes.
        critical_sections (list): List of critical sections where approval is required.

    Returns:
        bool: True if the changes are approved, False otherwise.
    """
    logger.info(f"Assessing risk for changes in {module_path}")

    # Perform risk assessment
    low_risk = assess_risk(changes, critical_sections)

    if low_risk:
        logger.info(f"Low-risk changes detected. Automatically approving changes for {module_path}")
        apply_changes(module_path)  # Apply the changes directly if low-risk
        return True
    else:
        logger.info(f"High-risk changes detected in critical section. Requesting manual approval for {module_path}")
        return request_manual_approval(module_path)  # Request manual approval for high-risk changes

def apply_changes(module_path):
    """
    Applies the proposed changes to the module. This could involve writing the new code to the module file.

    Args:
        module_path (str): The path to the module to apply changes to.
    """
    # Example: This function would normally write the changes to the file system.
    logger.info(f"Applying changes to {module_path}")
    # Here you would actually write the changes to the file, for now we simulate it.
    with open(module_path, 'a') as f:
        f.write("\n# Changes applied")  # Simulate applying changes

def request_manual_approval(module_path):
    """
    Requests manual approval for critical changes.

    Args:
        module_path (str): The path to the module where the changes are being proposed.

    Returns:
        bool: True if the changes are approved manually, False otherwise.
    """
    logger.info(f"Requesting manual approval for changes to {module_path}.")
    approval = input(f"Approve changes to {module_path}? (yes/no): ").strip().lower()
    if approval == 'yes':
        logger.info(f"Changes approved for {module_path}.")
        return True
    else:
        logger.warning(f"Changes not approved for {module_path}.")
        return False
