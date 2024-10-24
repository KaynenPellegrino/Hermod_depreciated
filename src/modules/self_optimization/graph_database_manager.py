# src/modules/self_optimization/graph_database_manager.py

import logging
from typing import Dict, Any, Optional, List

from neo4j import GraphDatabase, basic_auth

from src.utils.logger import get_logger
from src.utils.configuration_manager import ConfigurationManager


class GraphDatabaseManager:
    """
    Manages storage and querying of complex relationships between entities using Neo4j.
    Facilitates efficient querying and retrieval of data with relational complexities.
    """

    def __init__(self, project_id: str):
        """
        Initializes the GraphDatabaseManager with necessary configurations.

        Args:
            project_id (str): Unique identifier for the project.
        """
        self.logger = get_logger(__name__)
        self.project_id = project_id
        self.config_manager = ConfigurationManager()
        self.config = self.config_manager.get_configuration(project_id)

        self.uri = self.config.get('neo4j_uri', 'bolt://localhost:7687')
        self.user = self.config.get('neo4j_user', 'neo4j')
        self.password = self.config.get('neo4j_password', 'password')

        self.driver = None
        self.connect()

    def connect(self) -> None:
        """
        Establishes a connection to the Neo4j database.
        """
        self.logger.info(f"Connecting to Neo4j at '{self.uri}' with user '{self.user}'.")
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=basic_auth(self.user, self.password))
            self.logger.info("Connected to Neo4j successfully.")
        except Exception as e:
            self.logger.error(f"Failed to connect to Neo4j: {e}", exc_info=True)

    def close(self) -> None:
        """
        Closes the connection to the Neo4j database.
        """
        if self.driver:
            self.driver.close()
            self.logger.info("Neo4j connection closed.")

    def create_entity(self, label: str, properties: Dict[str, Any]) -> bool:
        """
        Creates an entity (node) in the graph database.

        Args:
            label (str): Label of the node (e.g., 'Project', 'Collaborator').
            properties (Dict[str, Any]): Properties of the node.

        Returns:
            bool: True if the entity was created successfully, False otherwise.
        """
        self.logger.info(f"Creating entity with label '{label}' and properties {properties}.")
        try:
            with self.driver.session() as session:
                session.write_transaction(self._create_entity_tx, label, properties)
            self.logger.info(f"Entity '{label}' created successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to create entity '{label}': {e}", exc_info=True)
            return False

    @staticmethod
    def _create_entity_tx(tx, label: str, properties: Dict[str, Any]):
        """
        Transaction function to create an entity.

        Args:
            tx: Neo4j transaction.
            label (str): Label of the node.
            properties (Dict[str, Any]): Properties of the node.
        """
        tx.run(f"CREATE (n:{label} $props)", props=properties)

    def create_relationship(self, from_label: str, from_props: Dict[str, Any],
                            to_label: str, to_props: Dict[str, Any],
                            relationship: str) -> bool:
        """
        Creates a relationship between two entities in the graph database.

        Args:
            from_label (str): Label of the source node.
            from_props (Dict[str, Any]): Properties of the source node to identify it.
            to_label (str): Label of the target node.
            to_props (Dict[str, Any]): Properties of the target node to identify it.
            relationship (str): Type of the relationship.

        Returns:
            bool: True if the relationship was created successfully, False otherwise.
        """
        self.logger.info(f"Creating relationship '{relationship}' between {from_label} and {to_label}.")
        try:
            with self.driver.session() as session:
                session.write_transaction(self._create_relationship_tx, from_label, from_props, to_label, to_props,
                                          relationship)
            self.logger.info(f"Relationship '{relationship}' created successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to create relationship '{relationship}': {e}", exc_info=True)
            return False

    @staticmethod
    def _create_relationship_tx(tx, from_label: str, from_props: Dict[str, Any],
                                to_label: str, to_props: Dict[str, Any],
                                relationship: str):
        """
        Transaction function to create a relationship.

        Args:
            tx: Neo4j transaction.
            from_label (str): Label of the source node.
            from_props (Dict[str, Any]): Properties of the source node.
            to_label (str): Label of the target node.
            to_props (Dict[str, Any]): Properties of the target node.
            relationship (str): Type of the relationship.
        """
        query = (
            f"MATCH (a:{from_label}), (b:{to_label}) "
            f"WHERE a.id = $from_id AND b.id = $to_id "
            f"CREATE (a)-[r:{relationship}]->(b)"
        )
        tx.run(query, from_id=from_props.get('id'), to_id=to_props.get('id'))

    def query_relationships(self, query: str, parameters: Dict[str, Any] = {}) -> List[Dict[str, Any]]:
        """
        Queries the graph database for relationships.

        Args:
            query (str): Cypher query string.
            parameters (Dict[str, Any], optional): Parameters for the query. Defaults to {}.

        Returns:
            List[Dict[str, Any]]: List of query results.
        """
        self.logger.info(f"Executing query: {query} with parameters {parameters}.")
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters)
                records = [record.data() for record in result]
            self.logger.debug(f"Query returned {len(records)} records.")
            return records
        except Exception as e:
            self.logger.error(f"Failed to execute query: {e}", exc_info=True)
            return []

    def add_project(self, project_id: str, name: str, description: str) -> bool:
        """
        Adds a new project to the graph database.

        Args:
            project_id (str): Unique identifier for the project.
            name (str): Name of the project.
            description (str): Description of the project.

        Returns:
            bool: True if the project was added successfully, False otherwise.
        """
        self.logger.info(f"Adding project '{project_id}' to graph database.")
        properties = {
            'id': project_id,
            'name': name,
            'description': description
        }
        return self.create_entity('Project', properties)

    def add_collaborator(self, collaborator_id: str, name: str, role: str) -> bool:
        """
        Adds a new collaborator to the graph database.

        Args:
            collaborator_id (str): Unique identifier for the collaborator.
            name (str): Name of the collaborator.
            role (str): Role of the collaborator.

        Returns:
            bool: True if the collaborator was added successfully, False otherwise.
        """
        self.logger.info(f"Adding collaborator '{collaborator_id}' to graph database.")
        properties = {
            'id': collaborator_id,
            'name': name,
            'role': role
        }
        return self.create_entity('Collaborator', properties)

    def log_ai_decision(self, decision_id: str, description: str, project_id: str,
                        collaborator_id: Optional[str] = None) -> bool:
        """
        Logs an AI decision in the graph database.

        Args:
            decision_id (str): Unique identifier for the decision.
            description (str): Description of the decision.
            project_id (str): ID of the associated project.
            collaborator_id (Optional[str], optional): ID of the collaborator involved. Defaults to None.

        Returns:
            bool: True if the decision was logged successfully, False otherwise.
        """
        self.logger.info(f"Logging AI decision '{decision_id}' to graph database.")
        properties = {
            'id': decision_id,
            'description': description
        }
        success = self.create_entity('AIDecision', properties)
        if not success:
            return False

        # Create relationship with project
        project_props = {'id': project_id}
        success = self.create_relationship('Project', project_props, 'AIDecision', {'id': decision_id}, 'HAS_DECISION')
        if not success:
            return False

        # If collaborator is involved, create relationship
        if collaborator_id:
            collaborator_props = {'id': collaborator_id}
            success = self.create_relationship('Collaborator', collaborator_props, 'AIDecision', {'id': decision_id},
                                               'MADE_DECISION')
            if not success:
                return False

        return True

    def get_project_decisions(self, project_id: str) -> List[Dict[str, Any]]:
        """
        Retrieves all AI decisions associated with a specific project.

        Args:
            project_id (str): Unique identifier for the project.

        Returns:
            List[Dict[str, Any]]: List of AI decision records.
        """
        self.logger.info(f"Retrieving AI decisions for project '{project_id}'.")
        query = (
            "MATCH (p:Project)-[:HAS_DECISION]->(d:AIDecision) "
            "RETURN d.id AS decision_id, d.description AS description, d.timestamp AS timestamp"
        )
        parameters = {'project_id': project_id}
        results = self.query_relationships(query, parameters)
        return results

    def add_relationship_between_projects(self, from_project_id: str, to_project_id: str, relationship: str) -> bool:
        """
        Creates a relationship between two projects.

        Args:
            from_project_id (str): ID of the source project.
            to_project_id (str): ID of the target project.
            relationship (str): Type of the relationship.

        Returns:
            bool: True if the relationship was created successfully, False otherwise.
        """
        self.logger.info(
            f"Creating relationship '{relationship}' between projects '{from_project_id}' and '{to_project_id}'.")
        from_project_props = {'id': from_project_id}
        to_project_props = {'id': to_project_id}
        return self.create_relationship('Project', from_project_props, 'Project', to_project_props, relationship)

    def run_sample_operations(self):
        """
        Runs sample operations to demonstrate usage of GraphDatabaseManager.
        """
        self.logger.info("Running sample operations on graph database.")
        # Add sample projects
        self.add_project('proj_12345', 'Hermod AI', 'AI system for project management.')
        self.add_project('proj_67890', 'Phoenix AI', 'AI system for resource allocation.')

        # Add sample collaborators
        self.add_collaborator('collab_1', 'Alice Smith', 'Data Scientist')
        self.add_collaborator('collab_2', 'Bob Johnson', 'AI Engineer')

        # Log AI decisions
        self.log_ai_decision('decision_1', 'Selected Logistic Regression for NLU model.', 'proj_12345', 'collab_1')
        self.log_ai_decision('decision_2', 'Automated code refactoring performed.', 'proj_12345')

        # Create relationship between projects
        self.add_relationship_between_projects('proj_12345', 'proj_67890', 'RELATED_TO')

        # Retrieve project decisions
        decisions = self.get_project_decisions('proj_12345')
        self.logger.info(f"Decisions for project 'proj_12345': {decisions}")


# Example Usage and Test Cases
if __name__ == "__main__":
    # Initialize GraphDatabaseManager
    project_id = "proj_12345"  # Replace with your actual project ID
    graph_manager = GraphDatabaseManager(project_id)

    # Run sample operations
    graph_manager.run_sample_operations()

    # Close the connection
    graph_manager.close()
