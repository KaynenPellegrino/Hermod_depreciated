from hermod.core.self_modification import controlled_self_modification
from hermod.core.human_approval import request_manual_approval
from hermod.core.version_control import commit_changes, merge_to_main_branch
from hermod.utils.audit_logger import audit_modification
from hermod.self_analysis import detect_code_improvements
from hermod.core.approval_manager import apply_changes
from hermod.utils.logger import setup_logger

# Initialize logger
logger = setup_logger()

def execute_self_upgrade(module_path, auto_approve=False):
    """
    Executes Hermod's self-upgrade process in a controlled, secure, and auditable manner.
    """
    logger.info(f"Executing self-upgrade for {module_path}...")

    # Step 1: Run controlled self-modification
    controlled_self_modification(module_path, auto_approve)

    # Step 2: Detect code improvements
    improvements = detect_code_improvements(module_path)
    logger.info(f"Code improvements detected: {improvements}")

    # Step 3: Commit changes if successful
    commit_changes(module_path)

    # Step 4: Request manual approval for critical modules
    if not auto_approve and not request_manual_approval(module_path):
        logger.warning(f"Manual approval not granted for {module_path}. Rollback initiated.")
        return

    # Step 5: Merge to main branch if approved
    merge_to_main_branch(module_path)

    # Step 6: Log the upgrade in the audit trail
    audit_modification(module_path)
