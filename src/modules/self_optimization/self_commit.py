# src/modules/self_optimization/self_commit.py

import os
import logging
from typing import Optional

from utils.logger import get_logger
from utils.configuration_manager import ConfigurationManager
from src.modules.collaboration.version_control import VersionControl

class SelfCommit:
    """
    Automates the process of committing code changes once they pass testing and validation.
    Integrates with the VersionControl module to handle branching, committing, and pushing changes.
    """

    def __init__(self, project_id: str):
        self.logger = get_logger(__name__)
        self.project_id = project_id
        self.config_manager = ConfigurationManager()
        self.config = self.config_manager.get_configuration(project_id)

        # Initialize VersionControl instance
        self.repo_path = self.config.get('repository_path', '/path/to/repo')  # Update as per configuration
        self.version_control = VersionControl(repo_path=self.repo_path)

    def commit_changes(self, message: str) -> bool:
        """
        Commits all staged changes with the provided commit message.

        Args:
            message (str): The commit message.

        Returns:
            bool: True if commit is successful, False otherwise.
        """
        self.logger.info("Starting commit process.")
        try:
            if not self.version_control.has_changes():
                self.logger.info("No changes to commit.")
                return False

            self.version_control.stage_all_changes()
            self.logger.debug("All changes staged.")

            self.version_control.commit(message=message)
            self.logger.info(f"Committed changes with message: '{message}'.")

            return True
        except Exception as e:
            self.logger.error(f"Failed to commit changes: {e}", exc_info=True)
            return False

    def push_changes(self, branch: Optional[str] = None) -> bool:
        """
        Pushes committed changes to the remote repository.

        Args:
            branch (Optional[str]): The branch to push to. Defaults to the current branch.

        Returns:
            bool: True if push is successful, False otherwise.
        """
        self.logger.info("Starting push process.")
        try:
            self.version_control.push(branch=branch)
            self.logger.info(f"Pushed changes to branch: '{branch or 'current branch'}'.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to push changes: {e}", exc_info=True)
            return False

    def create_new_branch(self, branch_name: str) -> bool:
        """
        Creates a new branch in the repository.

        Args:
            branch_name (str): The name of the new branch.

        Returns:
            bool: True if branch creation is successful, False otherwise.
        """
        self.logger.info(f"Creating new branch: '{branch_name}'.")
        try:
            self.version_control.create_branch(branch_name=branch_name)
            self.logger.info(f"Branch '{branch_name}' created successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to create branch '{branch_name}': {e}", exc_info=True)
            return False

    def run_commit_process(self, commit_message: str, branch_name: Optional[str] = None) -> bool:
        """
        Runs the complete commit and push process.

        Args:
            commit_message (str): The commit message.
            branch_name (Optional[str]): The branch to push to. If not specified, uses the current branch.

        Returns:
            bool: True if the entire process is successful, False otherwise.
        """
        self.logger.info("Running complete commit process.")

        if branch_name:
            if not self.create_new_branch(branch_name):
                self.logger.error("Branch creation failed. Aborting commit process.")
                return False
            self.version_control.checkout_branch(branch_name=branch_name)
            self.logger.info(f"Checked out to branch '{branch_name}'.")

        commit_success = self.commit_changes(message=commit_message)
        if not commit_success:
            self.logger.warning("Commit process did not proceed due to no changes or failure.")
            return False

        push_success = self.push_changes(branch=branch_name)
        if not push_success:
            self.logger.error("Push process failed.")
            return False

        self.logger.info("Commit and push processes completed successfully.")
        return True

    def run_sample_operations(self):
        """
        Demonstrates sample commit and push operations.
        """
        self.logger.info("Running sample commit operations.")

        # Example: Commit changes with a message
        commit_message = "Auto-commit: Optimized performance parameters."
        commit_success = self.commit_changes(message=commit_message)
        if commit_success:
            self.logger.info("Sample commit successful.")

        # Example: Push changes to the current branch
        push_success = self.push_changes()
        if push_success:
            self.logger.info("Sample push successful.")

        # Example: Create a new branch, commit, and push
        new_branch = "auto-optimization-branch"
        if self.create_new_branch(new_branch):
            self.version_control.checkout_branch(new_branch)
            self.logger.info(f"Switched to new branch '{new_branch}'.")
            new_commit_message = "Auto-commit on new branch: Further optimizations."
            if self.commit_changes(message=new_commit_message):
                self.push_changes(branch=new_branch)


# Example Usage and Test Cases
if __name__ == "__main__":
    import os

    # Initialize SelfCommit
    project_id = os.getenv('PROJECT_ID', 'default_project')  # Ensure PROJECT_ID is set
    self_commit = SelfCommit(project_id=project_id)

    # Run sample operations
    self_commit.run_sample_operations()
