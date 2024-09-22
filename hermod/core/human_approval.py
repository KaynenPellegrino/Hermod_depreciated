from hermod.utils.logger import setup_logger

# Initialize logger
logger = setup_logger()

def request_manual_approval(module_path):
    """
    Requests manual approval for critical changes.
    """
    logger.info(f"Requesting manual approval for changes to {module_path}.")
    approval = input(f"Approve changes to {module_path}? (yes/no): ").strip().lower()
    if approval == 'yes':
        logger.info(f"Changes approved for {module_path}.")
        return True
    else:
        logger.warning(f"Changes not approved for {module_path}.")
        return False
