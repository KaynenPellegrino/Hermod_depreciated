# src/modules/self_optimization/persistent_memory.py

import os
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

from utils.logger import get_logger
from utils.configuration_manager import ConfigurationManager

Base = declarative_base()


class Knowledge(Base):
    """
    SQLAlchemy model for storing knowledge entries.
    """
    __tablename__ = 'knowledge'

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(String, nullable=False)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    tags = Column(String, nullable=True)  # Comma-separated tags
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<Knowledge(title='{self.title}', project_id='{self.project_id}')>"


class PersistentMemory:
    """
    Manages persistent memory storage and retrieval for Hermod.
    """

    def __init__(self, project_id: str):
        """
        Initializes the PersistentMemory with necessary configurations.

        Args:
            project_id (str): Unique identifier for the project.
        """
        self.logger = get_logger(__name__)
        self.project_id = project_id
        self.config_manager = ConfigurationManager()
        self.config = self.config_manager.get_configuration(project_id)

        self.memory_dir = self.config.get('persistent_memory_dir', f'memory/{project_id}')
        os.makedirs(self.memory_dir, exist_ok=True)

        # Initialize SQLite database for knowledge storage
        self.db_path = os.path.join(self.memory_dir, 'persistent_memory.db')
        self.engine = create_engine(f'sqlite:///{self.db_path}')
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

        self.logger.info(f"PersistentMemory initialized for project '{project_id}'. Database at '{self.db_path}'.")

    def add_knowledge(self, title: str, content: str, tags: Optional[List[str]] = None) -> bool:
        """
        Adds a new knowledge entry to the persistent memory.

        Args:
            title (str): Title of the knowledge entry.
            content (str): Detailed content of the knowledge.
            tags (Optional[List[str]]): List of tags associated with the knowledge.

        Returns:
            bool: True if addition is successful, False otherwise.
        """
        self.logger.info(f"Adding knowledge entry titled '{title}'.")
        try:
            session = self.Session()
            tags_str = ','.join(tags) if tags else None
            knowledge = Knowledge(
                project_id=self.project_id,
                title=title,
                content=content,
                tags=tags_str
            )
            session.add(knowledge)
            session.commit()
            session.close()
            self.logger.info(f"Knowledge entry '{title}' added successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to add knowledge entry '{title}': {e}", exc_info=True)
            return False

    def get_knowledge(self, title: Optional[str] = None, tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Retrieves knowledge entries based on title or tags.

        Args:
            title (Optional[str]): Title of the knowledge entry to retrieve.
            tags (Optional[List[str]]): List of tags to filter knowledge entries.

        Returns:
            List[Dict[str, Any]]: List of knowledge entries matching the criteria.
        """
        self.logger.info(f"Retrieving knowledge with title='{title}' and tags='{tags}'.")
        try:
            session = self.Session()
            query = session.query(Knowledge).filter(Knowledge.project_id == self.project_id)
            if title:
                query = query.filter(Knowledge.title.ilike(f"%{title}%"))
            if tags:
                for tag in tags:
                    query = query.filter(Knowledge.tags.ilike(f"%{tag}%"))
            results = query.all()
            session.close()

            knowledge_list = []
            for knowledge in results:
                knowledge_dict = {
                    'id': knowledge.id,
                    'title': knowledge.title,
                    'content': knowledge.content,
                    'tags': knowledge.tags.split(',') if knowledge.tags else [],
                    'created_at': knowledge.created_at.isoformat(),
                    'updated_at': knowledge.updated_at.isoformat()
                }
                knowledge_list.append(knowledge_dict)

            self.logger.debug(f"Retrieved {len(knowledge_list)} knowledge entries.")
            return knowledge_list
        except Exception as e:
            self.logger.error(f"Failed to retrieve knowledge entries: {e}", exc_info=True)
            return []

    def update_knowledge(self, knowledge_id: int, title: Optional[str] = None,
                         content: Optional[str] = None, tags: Optional[List[str]] = None) -> bool:
        """
        Updates an existing knowledge entry.

        Args:
            knowledge_id (int): ID of the knowledge entry to update.
            title (Optional[str]): New title.
            content (Optional[str]): New content.
            tags (Optional[List[str]]): New list of tags.

        Returns:
            bool: True if update is successful, False otherwise.
        """
        self.logger.info(f"Updating knowledge entry ID '{knowledge_id}'.")
        try:
            session = self.Session()
            knowledge = session.query(Knowledge).filter(Knowledge.id == knowledge_id,
                                                       Knowledge.project_id == self.project_id).first()
            if not knowledge:
                self.logger.warning(f"Knowledge entry ID '{knowledge_id}' not found.")
                session.close()
                return False

            if title:
                knowledge.title = title
            if content:
                knowledge.content = content
            if tags is not None:
                knowledge.tags = ','.join(tags) if tags else None

            session.commit()
            session.close()
            self.logger.info(f"Knowledge entry ID '{knowledge_id}' updated successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to update knowledge entry ID '{knowledge_id}': {e}", exc_info=True)
            return False

    def delete_knowledge(self, knowledge_id: int) -> bool:
        """
        Deletes a knowledge entry from persistent memory.

        Args:
            knowledge_id (int): ID of the knowledge entry to delete.

        Returns:
            bool: True if deletion is successful, False otherwise.
        """
        self.logger.info(f"Deleting knowledge entry ID '{knowledge_id}'.")
        try:
            session = self.Session()
            knowledge = session.query(Knowledge).filter(Knowledge.id == knowledge_id,
                                                       Knowledge.project_id == self.project_id).first()
            if not knowledge:
                self.logger.warning(f"Knowledge entry ID '{knowledge_id}' not found.")
                session.close()
                return False

            session.delete(knowledge)
            session.commit()
            session.close()
            self.logger.info(f"Knowledge entry ID '{knowledge_id}' deleted successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete knowledge entry ID '{knowledge_id}': {e}", exc_info=True)
            return False

    def list_all_knowledge(self) -> List[Dict[str, Any]]:
        """
        Lists all knowledge entries for the project.

        Returns:
            List[Dict[str, Any]]: List of all knowledge entries.
        """
        self.logger.info("Listing all knowledge entries.")
        try:
            session = self.Session()
            results = session.query(Knowledge).filter(Knowledge.project_id == self.project_id).all()
            session.close()

            knowledge_list = []
            for knowledge in results:
                knowledge_dict = {
                    'id': knowledge.id,
                    'title': knowledge.title,
                    'content': knowledge.content,
                    'tags': knowledge.tags.split(',') if knowledge.tags else [],
                    'created_at': knowledge.created_at.isoformat(),
                    'updated_at': knowledge.updated_at.isoformat()
                }
                knowledge_list.append(knowledge_dict)

            self.logger.debug(f"Total knowledge entries: {len(knowledge_list)}.")
            return knowledge_list
        except Exception as e:
            self.logger.error(f"Failed to list all knowledge entries: {e}", exc_info=True)
            return []

    def run_sample_operations(self):
        """
        Runs sample operations to demonstrate usage of PersistentMemory.
        """
        self.logger.info("Running sample operations on PersistentMemory.")

        # Example: Add a new knowledge entry
        success = self.add_knowledge(
            title='Understanding SQLAlchemy ORM',
            content='SQLAlchemy ORM allows for high-level database interactions using Python objects.',
            tags=['SQLAlchemy', 'ORM', 'Database']
        )
        if success:
            self.logger.info("Sample knowledge entry added.")

        # Example: List all knowledge entries
        all_knowledge = self.list_all_knowledge()
        self.logger.info(f"All Knowledge Entries: {all_knowledge}")

        # Example: Retrieve specific knowledge by tag
        db_knowledge = self.get_knowledge(tags=['Database'])
        self.logger.info(f"Knowledge Entries with tag 'Database': {db_knowledge}")

        # Example: Update a knowledge entry
        if all_knowledge:
            first_entry = all_knowledge[0]
            update_success = self.update_knowledge(
                knowledge_id=first_entry['id'],
                content='Updated content: SQLAlchemy ORM provides a full suite of well-known enterprise-level persistence patterns.',
                tags=['SQLAlchemy', 'ORM', 'Persistence']
            )
            if update_success:
                self.logger.info(f"Knowledge entry ID '{first_entry['id']}' updated.")

        # Example: Delete a knowledge entry
        if all_knowledge:
            first_entry = all_knowledge[0]
            delete_success = self.delete_knowledge(knowledge_id=first_entry['id'])
            if delete_success:
                self.logger.info(f"Knowledge entry ID '{first_entry['id']}' deleted.")


# Example Usage and Test Cases
if __name__ == "__main__":
    # Initialize PersistentMemory
    project_id = os.getenv('PROJECT_ID', 'default_project')  # Ensure PROJECT_ID is set
    persistent_memory = PersistentMemory(project_id=project_id)

    # Run sample operations
    persistent_memory.run_sample_operations()
