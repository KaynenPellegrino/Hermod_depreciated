import os
import json
import logging
from typing import Dict, Any
from abc import ABC, abstractmethod


class MetadataStorageInterface(ABC):
    """
    Interface for metadata storage management.
    Provides methods to save, load, and update project metadata.
    """

    @abstractmethod
    def save_metadata(self, project_id: str, metadata: Dict[str, Any]) -> None:
        """
        Saves metadata for a given project.

        :param project_id: Unique identifier for the project
        :param metadata: Metadata to be saved
        """
        pass

    @abstractmethod
    def load_metadata(self, project_id: str) -> Dict[str, Any]:
        """
        Loads metadata for a given project.

        :param project_id: Unique identifier for the project
        :return: Metadata of the project
        """
        pass

    @abstractmethod
    def update_metadata(self, project_id: str, metadata: Dict[str, Any]) -> None:
        """
        Updates metadata for a given project.

        :param project_id: Unique identifier for the project
        :param metadata: Metadata to be updated
        """
        pass


class MetadataStorage(MetadataStorageInterface):
    def __init__(self, storage_path: str = 'metadata_storage'):
        self.storage_path = storage_path
        os.makedirs(self.storage_path, exist_ok=True)

    def _get_metadata_file_path(self, project_id: str) -> str:
        return os.path.join(self.storage_path, f"{project_id}.json")

    def save_metadata(self, project_id: str, metadata: Dict[str, Any]) -> None:
        try:
            file_path = self._get_metadata_file_path(project_id)
            with open(file_path, 'w') as file:
                json.dump(metadata, file)
            logging.debug(f"Saved metadata for project_id='{project_id}': {metadata}")
        except Exception as e:
            logging.error(f"Failed to save metadata for project_id='{project_id}': {e}")
            raise e

    def load_metadata(self, project_id: str) -> Dict[str, Any]:
        try:
            file_path = self._get_metadata_file_path(project_id)
            if os.path.exists(file_path):
                with open(file_path, 'r') as file:
                    metadata = json.load(file)
                logging.debug(f"Loaded metadata for project_id='{project_id}': {metadata}")
                return metadata
            else:
                logging.warning(f"No metadata found for project_id='{project_id}'.")
                return {}
        except Exception as e:
            logging.error(f"Failed to load metadata for project_id='{project_id}': {e}")
            raise e

    def update_metadata(self, project_id: str, metadata: Dict[str, Any]) -> None:
        try:
            existing_metadata = self.load_metadata(project_id)
            existing_metadata.update(metadata)
            self.save_metadata(project_id, existing_metadata)
            logging.debug(f"Updated metadata for project_id='{project_id}': {metadata}")
        except Exception as e:
            logging.error(f"Failed to update metadata for project_id='{project_id}': {e}")
            raise e
