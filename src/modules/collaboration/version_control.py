# src/modules/collaboration/version_control.py

import os
import logging
from typing import List, Dict, Any, Optional
from git import Repo, GitCommandError, Git
from utils.logger import get_logger
from utils.configuration_manager import ConfigurationManager
from src.modules.code_generation.project_manager import VersionControlInterface

class VersionControl(VersionControlInterface):
    """
    Integrates version control systems (like Git) into Hermod, allowing users to manage code versions,
    track changes, and collaborate on codebases effectively. Provides functionalities like commit history viewing,
    branching, merging, and conflict resolution within the application.
    """

    def __init__(self, repo_path: str):
        """
        Initializes the VersionControl with the specified repository path.

        Args:
            repo_path (str): Path to the Git repository.
        """
        self.logger = get_logger(__name__)
        self.repo_path = repo_path

        if not os.path.exists(repo_path):
            self.logger.error(f"Repository path '{repo_path}' does not exist.")
            raise FileNotFoundError(f"Repository path '{repo_path}' does not exist.")

        try:
            self.repo = Repo(repo_path)
            self.git = self.repo.git
            self.logger.info(f"Initialized Git repository at '{repo_path}'.")
        except GitCommandError as e:
            self.logger.error(f"Failed to initialize Git repository: {e}")
            raise e

    # Existing methods...

    def get_commit_history(self, max_count: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves the commit history.

        Args:
            max_count (int, optional): Maximum number of commits to retrieve. Defaults to 10.

        Returns:
            List[Dict[str, Any]]: List of commits with details.
        """
        try:
            commits = list(self.repo.iter_commits(max_count=max_count))
            commit_history = []
            for commit in commits:
                commit_info = {
                    'hexsha': commit.hexsha,
                    'author_name': commit.author.name,
                    'author_email': commit.author.email,
                    'message': commit.message.strip(),
                    'committed_datetime': commit.committed_datetime.isoformat()
                }
                commit_history.append(commit_info)
            self.logger.info(f"Retrieved the last {len(commit_history)} commits.")
            return commit_history
        except GitCommandError as e:
            self.logger.error(f"Failed to retrieve commit history: {e}", exc_info=True)
            raise e

    # ----------------------------
    # VersionControlInterface Methods
    # ----------------------------

    def initialize_repo(self, project_path: str) -> None:
        """
        Initializes a new Git repository at the specified project path.

        Args:
            project_path (str): Path to initialize the Git repository.
        """
        try:
            repo = Repo.init(project_path)
            self.logger.debug(f"Initialized new Git repository at '{project_path}'.")
        except Exception as e:
            self.logger.error(f"Failed to initialize Git repository at '{project_path}': {e}")
            raise e

    def commit_changes(self, project_path: str, commit_message: str) -> None:
        """
        Commits all changes in the specified project path with the provided commit message.

        Args:
            project_path (str): Path of the project to commit changes.
            commit_message (str): The commit message.
        """
        try:
            repo = Repo(project_path)
            repo.git.add(A=True)
            repo.index.commit(commit_message)
            self.logger.debug(f"Committed changes with message: '{commit_message}'.")
        except Exception as e:
            self.logger.error(f"Failed to commit changes in '{project_path}': {e}")
            raise e

    # ----------------------------
    # Repository Initialization
    # ----------------------------

    @classmethod
    def init_repo(cls, repo_path: str) -> 'VersionControl':
        """
        Initializes a new Git repository at the specified path.

        Args:
            repo_path (str): Path to initialize the Git repository.

        Returns:
            VersionControl: An instance of the VersionControl class.
        """
        if not os.path.exists(repo_path):
            os.makedirs(repo_path, exist_ok=True)
        try:
            repo = Repo.init(repo_path)
            logging.info(f"Initialized new Git repository at '{repo_path}'.")
            return cls(repo_path)
        except GitCommandError as e:
            logging.error(f"Failed to initialize Git repository: {e}")
            raise e

    # ----------------------------
    # Git Operations
    # ----------------------------

    def add_files(self, file_paths: Optional[List[str]] = None):
        """
        Adds files to the staging area.

        Args:
            file_paths (Optional[List[str]]): List of file paths to add. If None, adds all changes.
        """
        try:
            if file_paths:
                self.repo.index.add(file_paths)
                self.logger.info(f"Added files to staging: {file_paths}")
            else:
                self.repo.git.add(A=True)
                self.logger.info("Added all changes to staging.")
        except GitCommandError as e:
            self.logger.error(f"Failed to add files: {e}", exc_info=True)
            raise e

    def commit(self, message: str, author: Optional[Dict[str, str]] = None):
        """
        Commits staged changes with the provided commit message.

        Args:
            message (str): The commit message.
            author (Optional[Dict[str, str]]): Dictionary containing 'name' and 'email' of the author.
        """
        try:
            if author:
                self.repo.index.commit(message, author=author)
                self.logger.info(f"Committed changes with message: '{message}' by {author}")
            else:
                self.repo.index.commit(message)
                self.logger.info(f"Committed changes with message: '{message}'")
        except GitCommandError as e:
            self.logger.error(f"Failed to commit changes: {e}", exc_info=True)
            raise e

    def push(self, remote_name: str = 'origin', branch: Optional[str] = None):
        """
        Pushes commits to the remote repository.

        Args:
            remote_name (str): Name of the remote. Defaults to 'origin'.
            branch (Optional[str]): Branch to push to. Defaults to the current branch.
        """
        try:
            if not branch:
                branch = self.repo.active_branch.name
            self.repo.git.push(remote_name, branch)
            self.logger.info(f"Pushed commits to '{remote_name}/{branch}'.")
        except GitCommandError as e:
            self.logger.error(f"Failed to push commits: {e}", exc_info=True)
            raise e

    def pull(self, remote_name: str = 'origin', branch: Optional[str] = None):
        """
        Pulls commits from the remote repository.

        Args:
            remote_name (str): Name of the remote. Defaults to 'origin'.
            branch (Optional[str]): Branch to pull from. Defaults to the current branch.
        """
        try:
            if not branch:
                branch = self.repo.active_branch.name
            self.repo.git.pull(remote_name, branch)
            self.logger.info(f"Pulled commits from '{remote_name}/{branch}'.")
        except GitCommandError as e:
            self.logger.error(f"Failed to pull commits: {e}", exc_info=True)
            raise e

    def create_branch(self, branch_name: str):
        """
        Creates a new branch.

        Args:
            branch_name (str): Name of the new branch.
        """
        try:
            self.repo.git.branch(branch_name)
            self.logger.info(f"Created new branch '{branch_name}'.")
        except GitCommandError as e:
            self.logger.error(f"Failed to create branch '{branch_name}': {e}", exc_info=True)
            raise e

    def checkout_branch(self, branch_name: str):
        """
        Switches to the specified branch.

        Args:
            branch_name (str): The branch to switch to.
        """
        try:
            self.repo.git.checkout(branch_name)
            self.logger.info(f"Checked out to branch '{branch_name}'.")
        except GitCommandError as e:
            self.logger.error(f"Failed to checkout branch '{branch_name}': {e}", exc_info=True)
            raise e

    def merge_branch(self, source_branch: str, target_branch: Optional[str] = None):
        """
        Merges the source branch into the target branch.

        Args:
            source_branch (str): The branch to merge from.
            target_branch (Optional[str]): The branch to merge into. Defaults to current branch.
        """
        try:
            if target_branch:
                self.checkout_branch(target_branch)
            self.repo.git.merge(source_branch)
            self.logger.info(f"Merged branch '{source_branch}' into '{self.repo.active_branch.name}'.")
        except GitCommandError as e:
            self.logger.error(f"Failed to merge branch '{source_branch}': {e}", exc_info=True)
            raise e

    def get_commit_history(self, max_count: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves the commit history.

        Args:
            max_count (int, optional): Maximum number of commits to retrieve. Defaults to 10.

        Returns:
            List[Dict[str, Any]]: List of commits with details.
        """
        try:
            commits = list(self.repo.iter_commits(max_count=max_count))
            commit_history = []
            for commit in commits:
                commit_info = {
                    'hexsha': commit.hexsha,
                    'author_name': commit.author.name,
                    'author_email': commit.author.email,
                    'message': commit.message.strip(),
                    'committed_datetime': commit.committed_datetime.isoformat()
                }
                commit_history.append(commit_info)
            self.logger.info(f"Retrieved the last {len(commit_history)} commits.")
            return commit_history
        except GitCommandError as e:
            self.logger.error(f"Failed to retrieve commit history: {e}", exc_info=True)
            raise e

    def get_branches(self) -> List[str]:
        """
        Retrieves all branches in the repository.

        Returns:
            List[str]: List of branch names.
        """
        try:
            branches = [head.name for head in self.repo.heads]
            self.logger.info(f"Retrieved branches: {branches}")
            return branches
        except GitCommandError as e:
            self.logger.error(f"Failed to retrieve branches: {e}", exc_info=True)
            raise e

    def get_current_branch(self) -> str:
        """
        Retrieves the current active branch.

        Returns:
            str: Name of the current branch.
        """
        try:
            current_branch = self.repo.active_branch.name
            self.logger.info(f"Current branch: {current_branch}")
            return current_branch
        except Exception as e:
            self.logger.error(f"Failed to get current branch: {e}", exc_info=True)
            raise e

    def resolve_conflicts(self, branch_name: str):
        """
        Resolves merge conflicts by favoring the incoming changes.

        Args:
            branch_name (str): The branch to resolve conflicts from.
        """
        try:
            self.repo.git.merge(branch_name, strategy_option='theirs')
            self.logger.info(f"Resolved conflicts by favoring changes from '{branch_name}'.")
        except GitCommandError as e:
            self.logger.error(f"Failed to resolve conflicts with branch '{branch_name}': {e}", exc_info=True)
            raise e

    # ----------------------------
    # Helper Methods
    # ----------------------------

    def _get_user_email(self, username: str) -> str:
        """
        Retrieves the email address for a given username.
        Placeholder method; integrate with actual user management system.

        Args:
            username (str): Username of the user.

        Returns:
            str: Email address of the user.
        """
        # Placeholder: Replace with actual implementation
        email_mapping = {
            'user1': 'user1@example.com',
            'user2': 'user2@example.com',
            # Add more mappings as needed
        }
        return email_mapping.get(username, f"{username}@example.com")


# Example Usage and Test Cases
if __name__ == "__main__":
    import os

    # Initialize VersionControl
    repo_path = os.getenv('REPO_PATH', '/path/to/your/repo')  # Ensure REPO_PATH is set
    try:
        version_control = VersionControl(repo_path=repo_path)
    except FileNotFoundError:
        # Initialize a new repository if not existing
        version_control = VersionControl.init_repo(repo_path=repo_path)

    # Run sample operations
    try:
        # Add all files
        version_control.add_files()

        # Commit changes
        version_control.commit("Initial commit by Hermod.", author={'name': 'Hermod', 'email': 'hermod@example.com'})

        # Push to remote
        version_control.push()

        # Create a new branch
        version_control.create_branch('feature/new-feature')

        # Checkout to the new branch
        version_control.checkout_branch('feature/new-feature')

        # Make changes in the repository manually or via scripts here...

        # Add and commit changes
        version_control.add_files()
        version_control.commit("Added new feature.", author={'name': 'Hermod', 'email': 'hermod@example.com'})

        # Push the new branch
        version_control.push(branch='feature/new-feature')

        # Merge the new branch into main
        version_control.checkout_branch('main')
        version_control.merge_branch('feature/new-feature')

        # View commit history
        history = version_control.get_commit_history()
        for commit in history:
            print(commit)

    except Exception as e:
        version_control.logger.error(f"An error occurred during version control operations: {e}")
