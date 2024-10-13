import logging
import os
import shutil
from typing import Dict, Any
from abc import ABC, abstractmethod
import git  # GitPython library for Git integration

from src.modules.collaboration.version_control import VersionControl
from src.modules.data_management.metadata_storage import MetadataStorage

# Configure logging
logging.basicConfig(
    filename='hermod_project_manager.log',
    level=logging.DEBUG,
    format='%(asctime)s:%(levelname)s:%(message)s'
)


# Interfaces for dependencies
class VersionControlInterface(ABC):
    """
    Interface for version control systems.
    This should be replaced with the actual implementation as needed.
    """

    @abstractmethod
    def initialize_repo(self, project_path: str) -> None:
        pass

    @abstractmethod
    def commit_changes(self, project_path: str, commit_message: str) -> None:
        pass


class ProjectManager:
    def __init__(self,
                 version_control: VersionControlInterface,
                 metadata_storage: MetadataStorage,
                 base_directory: str = 'generated_projects'):
        """
        Initializes the ProjectManager with necessary dependencies.

        :param version_control: Instance of VersionControlInterface
        :param metadata_storage: Instance of MetadataStorage
        :param base_directory: Base directory where projects are stored
        """
        self.version_control = version_control
        self.metadata_storage = metadata_storage
        self.base_directory = base_directory
        logging.info("ProjectManager initialized.")

        # Ensure base directory exists
        os.makedirs(self.base_directory, exist_ok=True)
        logging.debug(f"Ensured base directory exists at '{self.base_directory}'.")

    def create_project(self, project_info: Dict[str, Any]) -> str:
        """
        Creates a new project with the specified information.

        :param project_info: Dictionary containing project details
        :return: The unique project ID
        """
        try:
            project_id = project_info.get('project_id')
            if not project_id:
                raise ValueError("Project information must include a 'project_id'.")

            project_path = self.get_project_path(project_id)
            if os.path.exists(project_path):
                raise FileExistsError(f"Project directory '{project_path}' already exists.")

            # Create project directory
            os.makedirs(project_path)
            logging.debug(f"Created project directory at '{project_path}'.")

            # Organize file structure based on project type
            self.organize_file_structure(project_path, project_info)

            # Save metadata
            self.metadata_storage.save_metadata(project_id, project_info)
            logging.info(f"Project '{project_id}' created successfully.")

            # Initialize version control
            self.version_control.initialize_repo(project_path)
            self.version_control.commit_changes(project_path, "Initial commit")

            return project_id
        except Exception as e:
            logging.error(f"Failed to create project: {e}")
            raise e

    def delete_project(self, project_id: str) -> None:
        """
        Deletes an existing project.

        :param project_id: Unique identifier for the project
        """
        try:
            project_path = self.get_project_path(project_id)
            if not os.path.exists(project_path):
                raise FileNotFoundError(f"Project directory '{project_path}' does not exist.")

            shutil.rmtree(project_path)
            logging.debug(f"Deleted project directory at '{project_path}'.")

            # Remove metadata
            self.metadata_storage.delete_metadata(project_id)
            logging.debug(f"Removed metadata for project_id='{project_id}'.")

            logging.info(f"Project '{project_id}' deleted successfully.")
        except Exception as e:
            logging.error(f"Failed to delete project '{project_id}': {e}")
            raise e

    def get_project_metadata(self, project_id: str) -> Dict[str, Any]:
        """
        Retrieves metadata for a specified project.

        :param project_id: Unique identifier for the project
        :return: Dictionary containing project metadata
        """
        try:
            metadata = self.metadata_storage.load_metadata(project_id)
            if not metadata:
                raise ValueError(f"No metadata found for project_id='{project_id}'.")
            return metadata
        except Exception as e:
            logging.error(f"Failed to retrieve metadata for project_id='{project_id}': {e}")
            raise e

    def update_project_metadata(self, project_id: str, metadata: Dict[str, Any]) -> None:
        """
        Updates metadata for a specified project.

        :param project_id: Unique identifier for the project
        :param metadata: Dictionary containing metadata updates
        """
        try:
            self.metadata_storage.update_metadata(project_id, metadata)
            logging.info(f"Metadata updated for project_id='{project_id}': {metadata}")
        except Exception as e:
            logging.error(f"Failed to update metadata for project_id='{project_id}': {e}")
            raise e

    def initialize_version_control(self, project_id: str) -> None:
        """
        Initializes version control for a project.

        :param project_id: Unique identifier for the project
        """
        try:
            project_path = self.get_project_path(project_id)
            self.version_control.initialize_repo(project_path)
            self.version_control.commit_changes(project_path, "Initialize version control")
            logging.info(f"Version control initialized for project_id='{project_id}'.")
        except Exception as e:
            logging.error(f"Failed to initialize version control for project_id='{project_id}': {e}")
            raise e

    def commit_changes(self, project_id: str, commit_message: str) -> None:
        """
        Commits changes to the project's version control repository.

        :param project_id: Unique identifier for the project
        :param commit_message: Commit message describing the changes
        """
        try:
            project_path = self.get_project_path(project_id)
            self.version_control.commit_changes(project_path, commit_message)
            logging.info(f"Committed changes for project_id='{project_id}' with message: '{commit_message}'.")
        except Exception as e:
            logging.error(f"Failed to commit changes for project_id='{project_id}': {e}")
            raise e

    def get_project_path(self, project_id: str) -> str:
        """
        Retrieves the file system path of a specified project.

        :param project_id: Unique identifier for the project
        :return: Path to the project's directory
        """
        return os.path.join(self.base_directory, project_id)

    def organize_file_structure(self, project_path: str, project_info: Dict[str, Any]) -> None:
        """
        Organizes the directory structure based on project type and language.

        :param project_path: Path to the project's directory
        :param project_info: Dictionary containing project details
        """
        try:
            language = project_info.get("language", "Python")
            project_type = project_info.get("project_type", "web_app")
            features = project_info.get("features", [])

            # Create directories
            os.makedirs(os.path.join(project_path, "src"), exist_ok=True)
            os.makedirs(os.path.join(project_path, "src", "modules"), exist_ok=True)
            os.makedirs(os.path.join(project_path, "tests"), exist_ok=True)
            logging.debug("Created standard directories: src/, src/modules/, tests/")

            # Create feature-specific modules
            for feature in features:
                module_path = os.path.join(project_path, "src", "modules", feature)
                os.makedirs(module_path, exist_ok=True)
                # Create an empty __init__.py file
                init_file = os.path.join(module_path, "__init__.py")
                with open(init_file, 'w') as f:
                    f.write("# Initialize the module\n")
                logging.debug(f"Created module directory and __init__.py for feature '{feature}'.")

            logging.info(f"Organized file structure for project at '{project_path}'.")
        except Exception as e:
            logging.error(f"Failed to organize file structure at '{project_path}': {e}")
            raise e


# Example usage and test cases
if __name__ == "__main__":
    # Initialize real dependencies
    version_control = VersionControl()  # Real VersionControl implementation
    metadata_storage = MetadataStorage(storage_path='metadata_storage')  # Real MetadataStorage implementation

    # Initialize ProjectManager
    project_manager = ProjectManager(version_control, metadata_storage)

    # Define project information
    project_info = {
        "project_id": "proj_12345",
        "name": "AI Chatbot",
        "description": "A chatbot using NLP techniques.",
        "language": "Python",
        "project_type": "web_app",
        "created_at": "2023-01-15",
        "last_updated": "2023-06-10",
        "features": ["authentication", "database_integration"],
        "performance_metrics": {
            "response_time_ms": 200,
            "memory_usage_mb": 150
        }
    }

    # Create a new project
    try:
        project_id = project_manager.create_project(project_info)
        print(f"Project '{project_id}' created successfully.")
    except Exception as e:
        print(f"Failed to create project: {e}")

    # Retrieve project metadata
    try:
        metadata = project_manager.get_project_metadata("proj_12345")
        print(f"Retrieved Metadata for 'proj_12345': {metadata}")
    except Exception as e:
        print(f"Failed to retrieve metadata: {e}")

    # Update project metadata
    try:
        project_manager.update_project_metadata("proj_12345", {"last_updated": "2024-01-01"})
        updated_metadata = project_manager.get_project_metadata("proj_12345")
        print(f"Updated Metadata for 'proj_12345': {updated_metadata}")
    except Exception as e:
        print(f"Failed to update metadata: {e}")

    # Commit changes to version control
    try:
        project_manager.commit_changes("proj_12345", "Added authentication module")
        print("Committed changes successfully.")
    except Exception as e:
        print(f"Failed to commit changes: {e}")

    # Uncomment the following lines to delete the project
    # try:
    #     project_manager.delete_project("proj_12345")
    #     print("Project 'proj_12345' deleted successfully.")
    # except Exception as e:
    #     print(f"Failed to delete project: {e}")

