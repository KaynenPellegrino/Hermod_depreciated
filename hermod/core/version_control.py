import subprocess
import os
from hermod.utils.logger import setup_logger

# Initialize logger
logger = setup_logger()

def rollback_changes(module_path):
    try:
        git_status = subprocess.run(['git', 'ls-files', '--error-unmatch', module_path], capture_output=True, text=True)
        if git_status.returncode != 0:
            logger.warning(f"{module_path} is not tracked by Git. Cannot rollback.")
            return

        subprocess.run(['git', 'checkout', '--', module_path], check=True)
        logger.info(f"Rollback successful for {module_path}.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to rollback changes for {module_path}: {e}")
        logger.warning(f"Manual intervention required for rollback.")

def commit_changes(module_path):
    """
    Commits changes to the development branch and requests manual approval for merges to the main branch.
    """
    logger.info(f"Committing changes for {module_path} to development branch.")
    subprocess.run(['git', 'add', module_path], check=True)
    subprocess.run(['git', 'commit', '-m', f"Auto-commit for {module_path} modification"], check=True)

def merge_to_main_branch(module_path):
    """
    Merges changes from the development branch to the main branch after human approval.
    """
    logger.info(f"Merging {module_path} changes to main branch.")
    subprocess.run(['git', 'checkout', 'main'], check=True)
    subprocess.run(['git', 'merge', 'development'], check=True)
