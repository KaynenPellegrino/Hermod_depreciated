# src/modules/collaboration/project_sharing_manager.py

import os
import logging
from typing import List, Dict, Any
from utils.logger import get_logger
from utils.configuration_manager import ConfigurationManager
from src.modules.collaboration.version_control import VersionControl
from src.modules.collaboration.collaboration_tools import CollaborationTools


class ProjectSharingManager:
    """
    Manages the sharing of projects between users, ensuring access control and permissions are properly enforced.
    Integrates with version control systems, such as Git, to track shared project history and contributions.
    """

    def __init__(self, project_id: str):
        """
        Initializes the ProjectSharingManager with necessary configurations and tools.

        Args:
            project_id (str): Unique identifier for the project.
        """
        self.logger = get_logger(__name__)
        self.project_id = project_id
        self.config_manager = ConfigurationManager()
        self.config = self.config_manager.get_configuration(project_id)

        # Initialize collaboration tools
        self.collaboration_tools = CollaborationTools(project_id=project_id)

        # Initialize version control
        repo_path = self.config.get('repository_path', '/path/to/repo')  # Ensure correct path
        self.version_control = VersionControl(repo_path=repo_path)

        # Access control list: {username: [permissions]}
        self.access_control: Dict[str, List[str]] = {}

        self.logger.info(f"ProjectSharingManager initialized for project '{project_id}'.")

    # ----------------------------
    # Access Control Management
    # ----------------------------

    def add_collaborator(self, username: str, permissions: List[str] = ['read', 'write']):
        """
        Adds a collaborator to the project with specified permissions.

        Args:
            username (str): Username of the collaborator.
            permissions (List[str], optional): List of permissions ('read', 'write', 'admin'). Defaults to ['read', 'write'].
        """
        try:
            if username in self.access_control:
                self.access_control[username].extend([perm for perm in permissions if perm not in self.access_control[username]])
            else:
                self.access_control[username] = permissions
            self.logger.info(f"Added collaborator '{username}' with permissions {permissions}.")

            # Optionally, integrate with external user management or notification systems
            # For example, sending Slack invitations via collaboration_tools
            user_emails = [self._get_user_email(username)]
            self.collaboration_tools.share_project(self.project_id, user_emails)

        except Exception as e:
            self.logger.error(f"Failed to add collaborator '{username}': {e}", exc_info=True)

    def remove_collaborator(self, username: str):
        """
        Removes a collaborator from the project.

        Args:
            username (str): Username of the collaborator to remove.
        """
        try:
            if username in self.access_control:
                del self.access_control[username]
                self.logger.info(f"Removed collaborator '{username}' from project.")

                # Optionally, revoke access via external systems
                # For example, removing from Slack channels

            else:
                self.logger.warning(f"Attempted to remove non-existent collaborator '{username}'.")
        except Exception as e:
            self.logger.error(f"Failed to remove collaborator '{username}': {e}", exc_info=True)

    def set_permissions(self, username: str, permissions: List[str]):
        """
        Sets permissions for a collaborator.

        Args:
            username (str): Username of the collaborator.
            permissions (List[str]): List of permissions ('read', 'write', 'admin').
        """
        try:
            if username in self.access_control:
                self.access_control[username] = permissions
                self.logger.info(f"Set permissions for collaborator '{username}' to {permissions}.")
            else:
                self.logger.warning(f"Attempted to set permissions for non-existent collaborator '{username}'.")
        except Exception as e:
            self.logger.error(f"Failed to set permissions for collaborator '{username}': {e}", exc_info=True)

    def get_permissions(self, username: str) -> List[str]:
        """
        Retrieves permissions for a collaborator.

        Args:
            username (str): Username of the collaborator.

        Returns:
            List[str]: List of permissions. Empty list if user not found.
        """
        return self.access_control.get(username, [])

    # ----------------------------
    # Project Sharing
    # ----------------------------

    def share_project_with_users(self, usernames: List[str], permissions: List[str] = ['read', 'write']):
        """
        Shares the project with a list of users, assigning specified permissions.

        Args:
            usernames (List[str]): List of usernames to share the project with.
            permissions (List[str], optional): Permissions to assign. Defaults to ['read', 'write'].
        """
        for username in usernames:
            self.add_collaborator(username, permissions)

    def revoke_project_access(self, usernames: List[str]):
        """
        Revokes access to the project for a list of users.

        Args:
            usernames (List[str]): List of usernames to revoke access from.
        """
        for username in usernames:
            self.remove_collaborator(username)

    # ----------------------------
    # Integration with Version Control
    # ----------------------------

    def track_contributions(self, commit_message: str):
        """
        Tracks contributions by committing changes to version control with appropriate messages.

        Args:
            commit_message (str): Commit message describing the changes.
        """
        try:
            self.version_control.commit(commit_message)
            self.version_control.push()
            self.logger.info(f"Tracked contributions with commit message: '{commit_message}'.")
        except Exception as e:
            self.logger.error(f"Failed to track contributions: {e}", exc_info=True)

    def get_project_history(self) -> List[Dict[str, Any]]:
        """
        Retrieves the project's commit history from version control.

        Returns:
            List[Dict[str, Any]]: List of commits with details.
        """
        try:
            history = self.version_control.get_commit_history()
            self.logger.info(f"Retrieved project history with {len(history)} commits.")
            return history
        except Exception as e:
            self.logger.error(f"Failed to retrieve project history: {e}", exc_info=True)
            return []

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

    # Initialize ProjectSharingManager
    project_id = os.getenv('PROJECT_ID', 'default_project')  # Ensure PROJECT_ID is set
    project_sharing_manager = ProjectSharingManager(project_id=project_id)

    # Run sample operations
    # Example 1: Share project with users
    collaborators = ['user1', 'user2']
    project_sharing_manager.share_project_with_users(collaborators, permissions=['read', 'write'])

    # Example 2: Set permissions for a user
    project_sharing_manager.set_permissions('user1', ['read', 'write', 'admin'])

    # Example 3: Revoke access for a user
    project_sharing_manager.revoke_project_access(['user2'])

    # Example 4: Track contributions
    project_sharing_manager.track_contributions("Initial commit by user1.")

    # Example 5: Get project history
    history = project_sharing_manager.get_project_history()
    for commit in history:
        print(commit)
