# data_management/metadata_storage.py

import logging
import os
from typing import Dict, Any, Optional, List
import json
from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData, DateTime, Text
from sqlalchemy.exc import SQLAlchemyError
from pymongo import MongoClient, errors as pymongo_errors
from datetime import datetime
from dotenv import load_dotenv
import hashlib

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
        self.compliance_table = None  # For SQL storage
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
                self.compliance_table = Table('compliance_reports', metadata,
                                              Column('id', Integer, primary_key=True, autoincrement=True),
                                              Column('entity', String(50), nullable=False),
                                              Column('identifier', String(255), nullable=False),
                                              Column('report_path', Text, nullable=False),
                                              Column('checked_at', DateTime, nullable=False),
                                              Column('created_at', DateTime, default=datetime.utcnow),
                                              Column('updated_at', DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
                                              )
                metadata.create_all(postgres_engine)
                logger.info("Metadata table 'compliance_reports' created successfully in PostgreSQL.")
            except SQLAlchemyError as e:
                logger.error(f"Failed to create metadata table in PostgreSQL: {e}")
        else:
            logger.warning("PostgreSQL engine not available. Skipping SQL metadata storage setup.")

        # MongoDB can be used as an alternative or supplementary storage
        # No explicit setup is needed as MongoDB is schemaless

    def save_metadata_sql(self, metadata: Dict[str, Any]):
        """
        Saves compliance report metadata to the SQL database.

        :param metadata: Dictionary containing metadata fields
        """
        postgres_engine = self.database_engines.get('postgresql')
        if not postgres_engine or not self.compliance_table:
            logger.error("PostgreSQL engine or compliance_reports table not initialized. Cannot save metadata to SQL.")
            return

        try:
            ins = self.compliance_table.insert().values(
                entity=metadata['entity'],
                identifier=metadata['identifier'],
                report_path=metadata['report_path'],
                checked_at=datetime.fromisoformat(metadata['checked_at']) if isinstance(metadata['checked_at'], str) else metadata['checked_at'],
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            postgres_engine.execute(ins)
            logger.info(f"Metadata for {metadata['entity']} '{metadata['identifier']}' saved successfully to SQL.")
        except SQLAlchemyError as e:
            logger.error(f"Failed to save metadata to SQL: {e}")

    def update_metadata_sql(self, report_id: int, update_fields: Dict[str, Any]):
        """
        Updates existing compliance report metadata in the SQL database.

        :param report_id: Unique identifier of the compliance report
        :param update_fields: Dictionary of fields to update
        """
        postgres_engine = self.database_engines.get('postgresql')
        if not postgres_engine or not self.compliance_table:
            logger.error("PostgreSQL engine or compliance_reports table not initialized. Cannot update metadata in SQL.")
            return

        try:
            update_values = {}
            if 'report_path' in update_fields:
                update_values['report_path'] = update_fields['report_path']
            if 'checked_at' in update_fields:
                update_values['checked_at'] = datetime.fromisoformat(update_fields['checked_at']) if isinstance(update_fields['checked_at'], str) else update_fields['checked_at']
            update_values['updated_at'] = datetime.utcnow()

            upd = self.compliance_table.update().where(
                self.compliance_table.c.id == report_id
            ).values(**update_values)

            result = postgres_engine.execute(upd)
            if result.rowcount > 0:
                logger.info(f"Metadata for report ID '{report_id}' updated successfully in SQL.")
            else:
                logger.warning(f"No metadata found for report ID '{report_id}' to update in SQL.")
        except SQLAlchemyError as e:
            logger.error(f"Failed to update metadata in SQL: {e}")

    def get_metadata_sql(self, report_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieves compliance report metadata from the SQL database.

        :param report_id: Unique identifier of the compliance report
        :return: Dictionary containing metadata or None if not found
        """
        postgres_engine = self.database_engines.get('postgresql')
        if not postgres_engine or not self.compliance_table:
            logger.error("PostgreSQL engine or compliance_reports table not initialized. Cannot retrieve metadata from SQL.")
            return None

        try:
            sel = self.compliance_table.select().where(
                self.compliance_table.c.id == report_id
            )
            result = postgres_engine.execute(sel).fetchone()
            if result:
                metadata = {
                    'id': result['id'],
                    'entity': result['entity'],
                    'identifier': result['identifier'],
                    'report_path': result['report_path'],
                    'checked_at': result['checked_at'].isoformat(),
                    'created_at': result['created_at'].isoformat(),
                    'updated_at': result['updated_at'].isoformat()
                }
                logger.info(f"Metadata for report ID '{report_id}' retrieved successfully from SQL.")
                return metadata
            else:
                logger.warning(f"No metadata found for report ID '{report_id}' in SQL.")
                return None
        except SQLAlchemyError as e:
            logger.error(f"Failed to retrieve metadata from SQL: {e}")
            return None

    def save_metadata_mongodb(self, metadata: Dict[str, Any]):
        """
        Saves compliance report metadata to MongoDB.

        :param metadata: Dictionary containing metadata fields
        """
        mongo_client = self.mongo_clients.get('mongodb')
        if not mongo_client:
            logger.error("MongoDB client not initialized. Cannot save metadata to MongoDB.")
            return

        try:
            db = mongo_client['hermod_metadata_db']
            collection = db['compliance_reports']
            # Convert datetime objects to ISO format strings for JSON serialization
            metadata_copy = metadata.copy()
            for key in ['checked_at']:
                if key in metadata_copy and isinstance(metadata_copy[key], str):
                    metadata_copy[key] = datetime.fromisoformat(metadata_copy[key])
            # MongoDB can store datetime objects directly
            collection.insert_one(metadata_copy)
            logger.info(f"Metadata for {metadata['entity']} '{metadata['identifier']}' saved successfully to MongoDB.")
        except pymongo_errors.PyMongoError as e:
            logger.error(f"Failed to save metadata to MongoDB: {e}")

    def update_metadata_mongodb(self, report_id: Any, update_fields: Dict[str, Any]):
        """
        Updates existing compliance report metadata in MongoDB.

        :param report_id: Unique identifier of the compliance report (e.g., ObjectId)
        :param update_fields: Dictionary of fields to update
        """
        mongo_client = self.mongo_clients.get('mongodb')
        if not mongo_client:
            logger.error("MongoDB client not initialized. Cannot update metadata in MongoDB.")
            return

        try:
            db = mongo_client['hermod_metadata_db']
            collection = db['compliance_reports']
            update_values = {}
            if 'report_path' in update_fields:
                update_values['report_path'] = update_fields['report_path']
            if 'checked_at' in update_fields:
                update_values['checked_at'] = datetime.fromisoformat(update_fields['checked_at']) if isinstance(update_fields['checked_at'], str) else update_fields['checked_at']
            update_values['updated_at'] = datetime.utcnow()

            result = collection.update_one(
                {'_id': report_id},
                {'$set': update_values}
            )
            if result.modified_count > 0:
                logger.info(f"Metadata for report ID '{report_id}' updated successfully in MongoDB.")
            else:
                logger.warning(f"No metadata found for report ID '{report_id}' to update in MongoDB.")
        except pymongo_errors.PyMongoError as e:
            logger.error(f"Failed to update metadata in MongoDB: {e}")

    def get_metadata_mongodb(self, report_id: Any) -> Optional[Dict[str, Any]]:
        """
        Retrieves compliance report metadata from MongoDB.

        :param report_id: Unique identifier of the compliance report (e.g., ObjectId)
        :return: Dictionary containing metadata or None if not found
        """
        mongo_client = self.mongo_clients.get('mongodb')
        if not mongo_client:
            logger.error("MongoDB client not initialized. Cannot retrieve metadata from MongoDB.")
            return None

        try:
            db = mongo_client['hermod_metadata_db']
            collection = db['compliance_reports']
            result = collection.find_one({'_id': report_id})
            if result:
                # Convert datetime objects to ISO format strings
                metadata = {
                    'id': str(result['_id']),
                    'entity': result.get('entity'),
                    'identifier': result.get('identifier'),
                    'report_path': result.get('report_path'),
                    'checked_at': result.get('checked_at').isoformat() if result.get('checked_at') else None,
                    'created_at': result.get('created_at').isoformat() if result.get('created_at') else None,
                    'updated_at': result.get('updated_at').isoformat() if result.get('updated_at') else None
                }
                logger.info(f"Metadata for report ID '{report_id}' retrieved successfully from MongoDB.")
                return metadata
            else:
                logger.warning(f"No metadata found for report ID '{report_id}' in MongoDB.")
                return None
        except pymongo_errors.PyMongoError as e:
            logger.error(f"Failed to retrieve metadata from MongoDB: {e}")
            return None

    def save_metadata(self, metadata: Dict[str, Any], storage_type: str = 'sql'):
        """
        Saves compliance report metadata to the specified storage backend.

        :param metadata: Dictionary containing metadata fields
        :param storage_type: Type of storage backend ('sql', 'mongodb')
        """
        logger.info(f"Saving metadata for {metadata.get('entity')} '{metadata.get('identifier')}' to '{storage_type}'.")
        if storage_type.lower() == 'sql':
            self.save_metadata_sql(metadata)
        elif storage_type.lower() == 'mongodb':
            self.save_metadata_mongodb(metadata)
        else:
            logger.error(f"Unsupported storage type for metadata: {storage_type}")

    def update_metadata(self, report_id: Any, update_fields: Dict[str, Any], storage_type: str = 'sql'):
        """
        Updates existing compliance report metadata in the specified storage backend.

        :param report_id: Unique identifier of the compliance report
        :param update_fields: Dictionary of fields to update
        :param storage_type: Type of storage backend ('sql', 'mongodb')
        """
        logger.info(f"Updating metadata for report ID '{report_id}' in '{storage_type}'.")
        if storage_type.lower() == 'sql':
            self.update_metadata_sql(report_id, update_fields)
        elif storage_type.lower() == 'mongodb':
            self.update_metadata_mongodb(report_id, update_fields)
        else:
            logger.error(f"Unsupported storage type for metadata: {storage_type}")

    def get_metadata(self, report_id: Any, storage_type: str = 'sql') -> Optional[Dict[str, Any]]:
        """
        Retrieves compliance report metadata from the specified storage backend.

        :param report_id: Unique identifier of the compliance report
        :param storage_type: Type of storage backend ('sql', 'mongodb')
        :return: Dictionary containing metadata or None if not found
        """
        logger.info(f"Retrieving metadata for report ID '{report_id}' from '{storage_type}'.")
        if storage_type.lower() == 'sql':
            return self.get_metadata_sql(report_id)
        elif storage_type.lower() == 'mongodb':
            return self.get_metadata_mongodb(report_id)
        else:
            logger.error(f"Unsupported storage type for metadata: {storage_type}")
            return None


# Example usage and test cases
if __name__ == "__main__":
    # Initialize MetadataStorage
    metadata_storage = MetadataStorage()

    # Example metadata to save for a compliance report
    compliance_metadata = {
        'entity': 'Project',  # Can be 'Project', 'Deployment', 'ClientPortal'
        'identifier': 'project_example_1',  # project_id, deployment_id, client_id
        'report_path': 'compliance_reports/Project_project_example_1_compliance_report_20241012_150000.json',
        'checked_at': datetime.utcnow().isoformat()
    }

    # Save metadata to SQL
    metadata_storage.save_metadata(compliance_metadata, storage_type='sql')

    # Save metadata to MongoDB
    metadata_storage.save_metadata(compliance_metadata, storage_type='mongodb')

    # Retrieve metadata from SQL (assuming the report ID is known, e.g., 1)
    # Note: Replace '1' with the actual report ID
    retrieved_metadata_sql = metadata_storage.get_metadata(1, storage_type='sql')
    if retrieved_metadata_sql:
        print("Retrieved Metadata from SQL:")
        print(json.dumps(retrieved_metadata_sql, indent=4))

    # Retrieve metadata from MongoDB (using ObjectId)
    # Note: Replace 'ObjectId("...")' with the actual ObjectId
    # from bson import ObjectId
    # retrieved_metadata_mongo = metadata_storage.get_metadata(ObjectId("64fa3f1f2b6f4a3d9c8b4567"), storage_type='mongodb')
    # if retrieved_metadata_mongo:
    #     print("\nRetrieved Metadata from MongoDB:")
    #     print(json.dumps(retrieved_metadata_mongo, indent=4))

    # Update metadata in SQL (assuming the report ID is known, e.g., 1)
    # Note: Replace '1' with the actual report ID
    update_fields_sql = {
        'report_path': 'compliance_reports/Project_project_example_1_compliance_report_20241012_150005.json',
        'checked_at': datetime.utcnow().isoformat()
    }
    metadata_storage.update_metadata(1, update_fields_sql, storage_type='sql')

    # Update metadata in MongoDB (using ObjectId)
    # Note: Replace 'ObjectId("...")' with the actual ObjectId
    # update_fields_mongo = {
    #     'report_path': 'compliance_reports/Project_project_example_1_compliance_report_20241012_150005.json',
    #     'checked_at': datetime.utcnow().isoformat()
    # }
    # metadata_storage.update_metadata(ObjectId("64fa3f1f2b6f4a3d9c8b4567"), update_fields_mongo, storage_type='mongodb')

    # Retrieve updated metadata from SQL
    updated_metadata_sql = metadata_storage.get_metadata(1, storage_type='sql')
    if updated_metadata_sql:
        print("\nUpdated Metadata from SQL:")
        print(json.dumps(updated_metadata_sql, indent=4))

    # Retrieve updated metadata from MongoDB
    # retrieved_updated_metadata_mongo = metadata_storage.get_metadata(ObjectId("64fa3f1f2b6f4a3d9c8b4567"), storage_type='mongodb')
    # if retrieved_updated_metadata_mongo:
    #     print("\nUpdated Metadata from MongoDB:")
    #     print(json.dumps(retrieved_updated_metadata_mongo, indent=4))
