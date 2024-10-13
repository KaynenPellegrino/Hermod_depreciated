# data_management/metadata_storage.py

import logging
import os
from typing import Dict, Any, Optional, List
import pandas as pd
import json
from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData, DateTime, Float, Text
from sqlalchemy.exc import SQLAlchemyError
from pymongo import MongoClient, errors as pymongo_errors
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging with RotatingFileHandler to prevent log files from becoming too large
from logging.handlers import RotatingFileHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a rotating file handler
handler = RotatingFileHandler('logs/hermod_metadata_storage.log', maxBytes=10**6, backupCount=5)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)


class MetadataStorage:
    """
    Manages project-specific metadata, such as configuration details, project properties,
    version history, and optimization metrics. Provides methods to save and retrieve
    metadata from persistent storage systems.
    """

    def __init__(self):
        """
        Initializes the MetadataStorage with necessary configurations.
        """
        self.database_engines = {}
        self.mongo_clients = {}
        self.metadata_table = None  # For SQL storage
        self._initialize_database_connections()
        self._initialize_metadata_storage()
        logger.info("MetadataStorage initialized successfully.")

    def _initialize_database_connections(self):
        """
        Initializes connections to SQL and NoSQL databases using environment variables.
        """
        # Initialize PostgreSQL connection
        postgres_host = os.getenv('POSTGRES_HOST')
        postgres_port = os.getenv('POSTGRES_PORT', '5432')
        postgres_db = os.getenv('POSTGRES_DB')
        postgres_user = os.getenv('POSTGRES_USER')
        postgres_password = os.getenv('POSTGRES_PASSWORD')

        if all([postgres_host, postgres_port, postgres_db, postgres_user, postgres_password]):
            try:
                postgres_engine = create_engine(
                    f"postgresql+psycopg2://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_db}"
                )
                self.database_engines['postgresql'] = postgres_engine
                logger.info("PostgreSQL engine initialized successfully for MetadataStorage.")
            except SQLAlchemyError as e:
                logger.error(f"Failed to initialize PostgreSQL engine for MetadataStorage: {e}")
        else:
            logger.warning("PostgreSQL credentials are incomplete. Skipping PostgreSQL initialization for MetadataStorage.")

        # Initialize MongoDB connection
        mongo_uri = os.getenv('MONGODB_URI')
        if mongo_uri:
            try:
                mongo_client = MongoClient(mongo_uri)
                # Test connection
                mongo_client.admin.command('ping')
                self.mongo_clients['mongodb'] = mongo_client
                logger.info("MongoDB client initialized successfully for MetadataStorage.")
            except pymongo_errors.ConnectionFailure as e:
                logger.error(f"Failed to connect to MongoDB for MetadataStorage: {e}")
        else:
            logger.warning("MongoDB URI not provided. Skipping MongoDB initialization for MetadataStorage.")

    def _initialize_metadata_storage(self):
        """
        Sets up the metadata storage structure based on available storage backends.
        """
        # Initialize SQL metadata table if PostgreSQL is available
        postgres_engine = self.database_engines.get('postgresql')
        if postgres_engine:
            try:
                metadata = MetaData()
                self.metadata_table = Table('project_metadata', metadata,
                                            Column('id', Integer, primary_key=True, autoincrement=True),
                                            Column('project_id', String, unique=True, nullable=False),
                                            Column('description', Text),
                                            Column('language', String),
                                            Column('project_type', String),
                                            Column('version_history', Text),  # JSON string
                                            Column('optimization_metrics', Text),  # JSON string
                                            Column('created_at', DateTime, default=datetime.utcnow),
                                            Column('updated_at', DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
                                            )
                metadata.create_all(postgres_engine)
                logger.info("Metadata table 'project_metadata' created successfully in PostgreSQL.")
            except SQLAlchemyError as e:
                logger.error(f"Failed to create metadata table in PostgreSQL: {e}")
        else:
            logger.warning("PostgreSQL engine not available. Skipping SQL metadata storage setup.")

        # MongoDB can be used as an alternative or supplementary storage
        # No explicit setup is needed as MongoDB is schemaless

    def save_metadata_sql(self, metadata: Dict[str, Any]):
        """
        Saves project metadata to the SQL database.

        :param metadata: Dictionary containing metadata fields
        """
        postgres_engine = self.database_engines.get('postgresql')
        if not postgres_engine or not self.metadata_table:
            logger.error("PostgreSQL engine or metadata table not initialized. Cannot save metadata to SQL.")
            return

        try:
            ins = self.metadata_table.insert().values(
                project_id=metadata['project_id'],
                description=metadata.get('description', ''),
                language=metadata.get('language', ''),
                project_type=metadata.get('project_type', ''),
                version_history=json.dumps(metadata.get('version_history', [])),
                optimization_metrics=json.dumps(metadata.get('optimization_metrics', {})),
                created_at=metadata.get('created_at', datetime.utcnow()),
                updated_at=metadata.get('updated_at', datetime.utcnow())
            )
            postgres_engine.execute(ins)
            logger.info(f"Metadata for project '{metadata['project_id']}' saved successfully to SQL.")
        except SQLAlchemyError as e:
            logger.error(f"Failed to save metadata to SQL: {e}")

    def update_metadata_sql(self, project_id: str, update_fields: Dict[str, Any]):
        """
        Updates existing project metadata in the SQL database.

        :param project_id: Unique identifier of the project
        :param update_fields: Dictionary of fields to update
        """
        postgres_engine = self.database_engines.get('postgresql')
        if not postgres_engine or not self.metadata_table:
            logger.error("PostgreSQL engine or metadata table not initialized. Cannot update metadata in SQL.")
            return

        try:
            update_values = {}
            if 'version_history' in update_fields:
                update_values['version_history'] = json.dumps(update_fields['version_history'])
            if 'optimization_metrics' in update_fields:
                update_values['optimization_metrics'] = json.dumps(update_fields['optimization_metrics'])
            for key in ['description', 'language', 'project_type']:
                if key in update_fields:
                    update_values[key] = update_fields[key]
            update_values['updated_at'] = datetime.utcnow()

            upd = self.metadata_table.update().where(
                self.metadata_table.c.project_id == project_id
            ).values(**update_values)

            result = postgres_engine.execute(upd)
            if result.rowcount > 0:
                logger.info(f"Metadata for project '{project_id}' updated successfully in SQL.")
            else:
                logger.warning(f"No metadata found for project '{project_id}' to update in SQL.")
        except SQLAlchemyError as e:
            logger.error(f"Failed to update metadata in SQL: {e}")

    def get_metadata_sql(self, project_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves project metadata from the SQL database.

        :param project_id: Unique identifier of the project
        :return: Dictionary containing metadata or None if not found
        """
        postgres_engine = self.database_engines.get('postgresql')
        if not postgres_engine or not self.metadata_table:
            logger.error("PostgreSQL engine or metadata table not initialized. Cannot retrieve metadata from SQL.")
            return None

        try:
            sel = self.metadata_table.select().where(
                self.metadata_table.c.project_id == project_id
            )
            result = postgres_engine.execute(sel).fetchone()
            if result:
                metadata = {
                    'project_id': result['project_id'],
                    'description': result['description'],
                    'language': result['language'],
                    'project_type': result['project_type'],
                    'version_history': json.loads(result['version_history']),
                    'optimization_metrics': json.loads(result['optimization_metrics']),
                    'created_at': result['created_at'],
                    'updated_at': result['updated_at']
                }
                logger.info(f"Metadata for project '{project_id}' retrieved successfully from SQL.")
                return metadata
            else:
                logger.warning(f"No metadata found for project '{project_id}' in SQL.")
                return None
        except SQLAlchemyError as e:
            logger.error(f"Failed to retrieve metadata from SQL: {e}")
            return None

    def save_metadata_mongodb(self, metadata: Dict[str, Any]):
        """
        Saves project metadata to MongoDB.

        :param metadata: Dictionary containing metadata fields
        """
        mongo_client = self.mongo_clients.get('mongodb')
        if not mongo_client:
            logger.error("MongoDB client not initialized. Cannot save metadata to MongoDB.")
            return

        try:
            db = mongo_client['hermod_metadata_db']
            collection = db['project_metadata']
            # Convert datetime objects to ISO format strings for JSON serialization
            metadata_copy = metadata.copy()
            for key in ['created_at', 'updated_at']:
                if key in metadata_copy and isinstance(metadata_copy[key], datetime):
                    metadata_copy[key] = metadata_copy[key].isoformat()
            # Convert lists and dicts to JSON strings if necessary
            if 'version_history' in metadata_copy and isinstance(metadata_copy['version_history'], list):
                metadata_copy['version_history'] = json.dumps(metadata_copy['version_history'])
            if 'optimization_metrics' in metadata_copy and isinstance(metadata_copy['optimization_metrics'], dict):
                metadata_copy['optimization_metrics'] = json.dumps(metadata_copy['optimization_metrics'])
            collection.insert_one(metadata_copy)
            logger.info(f"Metadata for project '{metadata['project_id']}' saved successfully to MongoDB.")
        except pymongo_errors.PyMongoError as e:
            logger.error(f"Failed to save metadata to MongoDB: {e}")

    def update_metadata_mongodb(self, project_id: str, update_fields: Dict[str, Any]):
        """
        Updates existing project metadata in MongoDB.

        :param project_id: Unique identifier of the project
        :param update_fields: Dictionary of fields to update
        """
        mongo_client = self.mongo_clients.get('mongodb')
        if not mongo_client:
            logger.error("MongoDB client not initialized. Cannot update metadata in MongoDB.")
            return

        try:
            db = mongo_client['hermod_metadata_db']
            collection = db['project_metadata']
            update_values = {}
            if 'version_history' in update_fields:
                update_values['version_history'] = json.dumps(update_fields['version_history'])
            if 'optimization_metrics' in update_fields:
                update_values['optimization_metrics'] = json.dumps(update_fields['optimization_metrics'])
            for key in ['description', 'language', 'project_type']:
                if key in update_fields:
                    update_values[key] = update_fields[key]
            update_values['updated_at'] = datetime.utcnow().isoformat()

            result = collection.update_one(
                {'project_id': project_id},
                {'$set': update_values}
            )
            if result.modified_count > 0:
                logger.info(f"Metadata for project '{project_id}' updated successfully in MongoDB.")
            else:
                logger.warning(f"No metadata found for project '{project_id}' to update in MongoDB.")
        except pymongo_errors.PyMongoError as e:
            logger.error(f"Failed to update metadata in MongoDB: {e}")

    def get_metadata_mongodb(self, project_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves project metadata from MongoDB.

        :param project_id: Unique identifier of the project
        :return: Dictionary containing metadata or None if not found
        """
        mongo_client = self.mongo_clients.get('mongodb')
        if not mongo_client:
            logger.error("MongoDB client not initialized. Cannot retrieve metadata from MongoDB.")
            return None

        try:
            db = mongo_client['hermod_metadata_db']
            collection = db['project_metadata']
            result = collection.find_one({'project_id': project_id})
            if result:
                metadata = {
                    'project_id': result.get('project_id'),
                    'description': result.get('description'),
                    'language': result.get('language'),
                    'project_type': result.get('project_type'),
                    'version_history': json.loads(result.get('version_history', '[]')),
                    'optimization_metrics': json.loads(result.get('optimization_metrics', '{}')),
                    'created_at': result.get('created_at'),
                    'updated_at': result.get('updated_at')
                }
                logger.info(f"Metadata for project '{project_id}' retrieved successfully from MongoDB.")
                return metadata
            else:
                logger.warning(f"No metadata found for project '{project_id}' in MongoDB.")
                return None
        except pymongo_errors.PyMongoError as e:
            logger.error(f"Failed to retrieve metadata from MongoDB: {e}")
            return None

    def save_metadata(self, metadata: Dict[str, Any], storage_type: str = 'sql'):
        """
        Saves project metadata to the specified storage backend.

        :param metadata: Dictionary containing metadata fields
        :param storage_type: Type of storage backend ('sql', 'mongodb')
        """
        logger.info(f"Saving metadata for project '{metadata.get('project_id')}' to '{storage_type}'.")
        if storage_type.lower() == 'sql':
            self.save_metadata_sql(metadata)
        elif storage_type.lower() == 'mongodb':
            self.save_metadata_mongodb(metadata)
        else:
            logger.error(f"Unsupported storage type for metadata: {storage_type}")

    def update_metadata(self, project_id: str, update_fields: Dict[str, Any], storage_type: str = 'sql'):
        """
        Updates existing project metadata in the specified storage backend.

        :param project_id: Unique identifier of the project
        :param update_fields: Dictionary of fields to update
        :param storage_type: Type of storage backend ('sql', 'mongodb')
        """
        logger.info(f"Updating metadata for project '{project_id}' in '{storage_type}'.")
        if storage_type.lower() == 'sql':
            self.update_metadata_sql(project_id, update_fields)
        elif storage_type.lower() == 'mongodb':
            self.update_metadata_mongodb(project_id, update_fields)
        else:
            logger.error(f"Unsupported storage type for metadata: {storage_type}")

    def get_metadata(self, project_id: str, storage_type: str = 'sql') -> Optional[Dict[str, Any]]:
        """
        Retrieves project metadata from the specified storage backend.

        :param project_id: Unique identifier of the project
        :param storage_type: Type of storage backend ('sql', 'mongodb')
        :return: Dictionary containing metadata or None if not found
        """
        logger.info(f"Retrieving metadata for project '{project_id}' from '{storage_type}'.")
        if storage_type.lower() == 'sql':
            return self.get_metadata_sql(project_id)
        elif storage_type.lower() == 'mongodb':
            return self.get_metadata_mongodb(project_id)
        else:
            logger.error(f"Unsupported storage type for metadata: {storage_type}")
            return None

# Example usage and test cases
if __name__ == "__main__":
    # Initialize MetadataStorage
    metadata_storage = MetadataStorage()

    # Example metadata to save
    project_metadata = {
        'project_id': 'project_123',
        'description': 'Natural Language Processing for sentiment analysis.',
        'language': 'Python',
        'project_type': 'Classification',
        'version_history': [
            {'version': '1.0', 'changes': 'Initial project setup.'},
            {'version': '1.1', 'changes': 'Added data preprocessing steps.'}
        ],
        'optimization_metrics': {
            'accuracy': 0.95,
            'precision': 0.94,
            'recall': 0.96,
            'f1_score': 0.95
        },
        'created_at': datetime.utcnow(),
        'updated_at': datetime.utcnow()
    }

    # Save metadata to SQL
    metadata_storage.save_metadata(project_metadata, storage_type='sql')

    # Save metadata to MongoDB
    metadata_storage.save_metadata(project_metadata, storage_type='mongodb')

    # Retrieve metadata from SQL
    retrieved_metadata_sql = metadata_storage.get_metadata('project_123', storage_type='sql')
    if retrieved_metadata_sql:
        print("Retrieved Metadata from SQL:")
        print(retrieved_metadata_sql)

    # Retrieve metadata from MongoDB
    retrieved_metadata_mongo = metadata_storage.get_metadata('project_123', storage_type='mongodb')
    if retrieved_metadata_mongo:
        print("\nRetrieved Metadata from MongoDB:")
        print(retrieved_metadata_mongo)

    # Update metadata in SQL
    update_fields_sql = {
        'version_history': [
            {'version': '1.0', 'changes': 'Initial project setup.'},
            {'version': '1.1', 'changes': 'Added data preprocessing steps.'},
            {'version': '1.2', 'changes': 'Implemented model training pipeline.'}
        ],
        'optimization_metrics': {
            'accuracy': 0.96,
            'precision': 0.95,
            'recall': 0.97,
            'f1_score': 0.96
        }
    }
    metadata_storage.update_metadata('project_123', update_fields_sql, storage_type='sql')

    # Update metadata in MongoDB
    update_fields_mongo = {
        'version_history': [
            {'version': '1.0', 'changes': 'Initial project setup.'},
            {'version': '1.1', 'changes': 'Added data preprocessing steps.'},
            {'version': '1.2', 'changes': 'Implemented model training pipeline.'}
        ],
        'optimization_metrics': {
            'accuracy': 0.96,
            'precision': 0.95,
            'recall': 0.97,
            'f1_score': 0.96
        }
    }
    metadata_storage.update_metadata('project_123', update_fields_mongo, storage_type='mongodb')

    # Retrieve updated metadata from SQL
    updated_metadata_sql = metadata_storage.get_metadata('project_123', storage_type='sql')
    if updated_metadata_sql:
        print("\nUpdated Metadata from SQL:")
        print(updated_metadata_sql)

    # Retrieve updated metadata from MongoDB
    updated_metadata_mongo = metadata_storage.get_metadata('project_123', storage_type='mongodb')
    if updated_metadata_mongo:
        print("\nUpdated Metadata from MongoDB:")
        print(updated_metadata_mongo)
