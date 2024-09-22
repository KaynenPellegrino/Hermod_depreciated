import difflib
import os
from hermod.utils.logger import setup_logger
from hermod.core.version_control import rollback_changes

# Initialize logger
logger = setup_logger()

def compare_code(original_code, refactored_code):
    """
    Compares the original and refactored code and returns a unified diff.
    """
    diff = difflib.unified_diff(
        original_code.splitlines(),
        refactored_code.splitlines(),
        lineterm='',
        fromfile='Original Code',
        tofile='Refactored Code'
    )
    return '\n'.join(diff)

def audit_modification(module_path, refactored_code):
    """
    Logs the details of Hermod's modification, including test results and performance metrics.
    """
    logger.info(f"Auditing modifications for {module_path}.")
    original_code = open(module_path, 'r').read()
    diff = compare_code(original_code, refactored_code)

    audit_log_path = f'logs/audit_{module_path}.log'
    logger.info(f"Writing audit log to {audit_log_path}")
    with open(audit_log_path, 'w') as f:
        f.write(f"Modification audit for {module_path}\n")
        f.write(f"Code Diff:\n{diff}\n")

    logger.info(f"Audit complete for {module_path}")
