# src/modules/self_optimization/multi_agent_manager.py

import os
import logging
from typing import Dict, Any, Optional, List
import multiprocessing
import time
import uuid

from utils.logger import get_logger
from utils.configuration_manager import ConfigurationManager


class Agent(multiprocessing.Process):
    """
    Represents an individual AI agent.
    Each agent runs in its own process.
    """

    def __init__(self, agent_id: str, role: str, task_queue: multiprocessing.Queue, result_queue: multiprocessing.Queue):
        """
        Initializes the Agent.

        Args:
            agent_id (str): Unique identifier for the agent.
            role (str): Role or specialty of the agent.
            task_queue (multiprocessing.Queue): Queue from which the agent receives tasks.
            result_queue (multiprocessing.Queue): Queue to which the agent sends results.
        """
        super().__init__()
        self.agent_id = agent_id
        self.role = role
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.logger = get_logger(__name__)
        self.active = True

    def run(self):
        """
        The main loop of the agent.
        Listens for tasks and processes them.
        """
        self.logger.info(f"Agent '{self.agent_id}' with role '{self.role}' started.")
        while self.active:
            try:
                if not self.task_queue.empty():
                    task = self.task_queue.get()
                    if task is None:
                        self.logger.info(f"Agent '{self.agent_id}' received shutdown signal.")
                        break
                    self.logger.info(f"Agent '{self.agent_id}' received task: {task}")
                    result = self.process_task(task)
                    self.result_queue.put({'agent_id': self.agent_id, 'result': result})
            except Exception as e:
                self.logger.error(f"Agent '{self.agent_id}' encountered an error: {e}", exc_info=True)
            time.sleep(0.1)  # Prevent busy waiting
        self.logger.info(f"Agent '{self.agent_id}' with role '{self.role}' terminated.")

    def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a given task based on the agent's role.

        Args:
            task (Dict[str, Any]): The task to process.

        Returns:
            Dict[str, Any]: The result of the task processing.
        """
        self.logger.info(f"Agent '{self.agent_id}' processing task: {task}")
        # Placeholder for actual task processing logic
        # Replace with real processing based on role
        if self.role == 'DataCleaner':
            # Simulate data cleaning
            cleaned_data = f"Cleaned data for {task.get('data_id')}"
            return {'status': 'success', 'data': cleaned_data}
        elif self.role == 'FeatureEngineer':
            # Simulate feature engineering
            engineered_features = f"Engineered features for {task.get('data_id')}"
            return {'status': 'success', 'features': engineered_features}
        elif self.role == 'ModelTrainer':
            # Simulate model training
            trained_model = f"Trained model for {task.get('model_name')}"
            return {'status': 'success', 'model': trained_model}
        else:
            self.logger.warning(f"Agent '{self.agent_id}' has an undefined role: '{self.role}'")
            return {'status': 'failure', 'reason': 'Undefined role'}

    def stop_agent(self):
        """
        Stops the agent's main loop.
        """
        self.active = False
        self.task_queue.put(None)  # Send shutdown signal


class MultiAgentManager:
    """
    Manages multiple AI agents, handling their creation, communication, and lifecycle.
    """

    def __init__(self, project_id: str):
        """
        Initializes the MultiAgentManager with necessary configurations.

        Args:
            project_id (str): Unique identifier for the project.
        """
        self.logger = get_logger(__name__)
        self.project_id = project_id
        self.config_manager = ConfigurationManager()
        self.config = self.config_manager.get_configuration(project_id)

        self.agents: Dict[str, Agent] = {}
        self.task_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()
        self.agent_roles = self.config.get('agent_roles', {'DataCleaner': 2, 'FeatureEngineer': 2, 'ModelTrainer': 1})
        self.logger.info(f"MultiAgentManager initialized for project '{project_id}' with roles: {self.agent_roles}.")

    def start_agents(self):
        """
        Starts the agents based on predefined roles and counts.
        """
        self.logger.info("Starting agents based on configured roles.")
        for role, count in self.agent_roles.items():
            for _ in range(count):
                agent_id = str(uuid.uuid4())
                agent = Agent(agent_id=agent_id, role=role, task_queue=self.task_queue, result_queue=self.result_queue)
                agent.start()
                self.agents[agent_id] = agent
                self.logger.info(f"Started Agent '{agent_id}' with role '{role}'.")

    def stop_agents(self):
        """
        Stops all running agents gracefully.
        """
        self.logger.info("Stopping all agents.")
        for agent_id, agent in self.agents.items():
            agent.stop_agent()
            agent.join()
            self.logger.info(f"Agent '{agent_id}' stopped.")
        self.agents.clear()

    def assign_task(self, task: Dict[str, Any]):
        """
        Assigns a task to the task queue.

        Args:
            task (Dict[str, Any]): The task to assign.
        """
        self.logger.info(f"Assigning task: {task}")
        self.task_queue.put(task)

    def collect_results(self) -> List[Dict[str, Any]]:
        """
        Collects results from the result queue.

        Returns:
            List[Dict[str, Any]]: List of results from agents.
        """
        self.logger.info("Collecting results from agents.")
        results = []
        while not self.result_queue.empty():
            result = self.result_queue.get()
            results.append(result)
            self.logger.info(f"Collected result from Agent '{result.get('agent_id')}': {result.get('result')}")
        return results

    def run_sample_operations(self):
        """
        Runs sample operations to demonstrate usage of MultiAgentManager.
        """
        self.logger.info("Running sample operations on MultiAgentManager.")

        # Start agents
        self.start_agents()

        # Assign sample tasks
        sample_tasks = [
            {'task_id': 'task_1', 'data_id': 'data_001'},
            {'task_id': 'task_2', 'data_id': 'data_002'},
            {'task_id': 'task_3', 'model_name': 'model_A'},
        ]

        for task in sample_tasks:
            self.assign_task(task)

        # Allow some time for tasks to be processed
        time.sleep(2)

        # Collect results
        results = self.collect_results()
        self.logger.info(f"Sample task results: {results}")

        # Stop agents
        self.stop_agents()


# Example Usage and Test Cases
if __name__ == "__main__":
    # Initialize MultiAgentManager
    project_id = os.getenv('PROJECT_ID', 'default_project')  # Ensure PROJECT_ID is set
    multi_agent_manager = MultiAgentManager(project_id=project_id)

    # Run sample operations
    multi_agent_manager.run_sample_operations()
