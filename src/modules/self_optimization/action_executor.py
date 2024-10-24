# src/modules/self_optimization/action_executor.py

import logging
import os
import subprocess
from typing import Dict, Any, Optional

from utils.configuration_manager import ConfigurationManager
from utils.logger import get_logger
from src.pipeline.model_training_pipeline import train_nlu_models
from src.modules.collaboration.version_control import VersionControl
from src.modules.feedback_loop.feedback_analyzer import FeedbackAnalyzer


class ActionExecutor:
    """
    Executes autonomous actions based on insights from the feedback loop to optimize Hermod's performance.
    Actions include adjusting configurations, retraining models, and modifying code.
    """

    def __init__(self, project_id: str):
        """
        Initializes the ActionExecutor with configurations specific to the project.

        Args:
            project_id (str): Unique identifier for the project.
        """
        self.logger = get_logger(__name__)
        self.config_manager = ConfigurationManager()
        self.project_id = project_id
        self.config = self.config_manager.get_configuration(project_id)

        # Initialize other necessary managers
        repo_path = self.config.get('repository_path', '/path/to/repo')  # Ensure this is correctly set in config
        self.version_control = VersionControl(repo_path=repo_path)
        self.feedback_analyzer = FeedbackAnalyzer(project_id=project_id, config_path='config.yaml')

        self.logger.info(f"ActionExecutor initialized for project '{project_id}'.")

    def adjust_configuration(self, config_changes: Dict[str, Any]) -> bool:
        """
        Adjusts system or model configurations based on the provided changes.

        Args:
            config_changes (Dict[str, Any]): Dictionary containing configuration keys and their new values.

        Returns:
            bool: True if configurations were successfully updated, False otherwise.
        """
        self.logger.info(f"Adjusting configurations with changes: {config_changes}")
        try:
            for key, value in config_changes.items():
                self.config_manager.set_value(self.project_id, key, value)
                self.logger.debug(f"Set '{key}' to '{value}' in configuration.")

            self.config_manager.save_configuration(self.project_id)
            self.logger.info("Configuration adjustments completed successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to adjust configurations: {e}", exc_info=True)
            return False

    def retrain_models(self, model_types: Optional[list] = None) -> bool:
        """
        Initiates retraining of specified models.

        Args:
            model_types (Optional[list]): List of model types to retrain (e.g., ['classification', 'qa']).
                                          If None, retrains all available models.

        Returns:
            bool: True if retraining was successfully initiated, False otherwise.
        """
        self.logger.info(f"Retraining models: {model_types if model_types else 'All Models'}")
        try:
            # Here, model_types can be used to selectively retrain models
            # For simplicity, train_nlu_models will handle retraining all NLU models
            train_nlu_models(self.project_id)
            self.logger.info("Model retraining initiated successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to retrain models: {e}", exc_info=True)
            return False

    def modify_code(self, file_path: str, modifications: Dict[str, Any]) -> bool:
        """
        Modifies code files based on specified modifications.

        Args:
            file_path (str): Path to the file to be modified.
            modifications (Dict[str, Any]): Dictionary specifying the modifications.
                                             Example: {'changes': [{'line_number': 10, 'content': 'new_line_content'}, ...]}

        Returns:
            bool: True if code was successfully modified, False otherwise.
        """
        self.logger.info(f"Modifying code at '{file_path}' with modifications: {modifications}")
        try:
            if not os.path.isfile(file_path):
                self.logger.error(f"File '{file_path}' does not exist.")
                return False

            # Read the original file
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # Apply modifications
            changes = modifications.get('changes', [])
            for mod in changes:
                line_number = mod.get('line_number')
                new_content = mod.get('content')
                if line_number is not None and new_content is not None:
                    if 0 <= line_number < len(lines):
                        self.logger.debug(f"Modifying line {line_number + 1}: {new_content}")
                        lines[line_number] = new_content + '\n'
                    else:
                        self.logger.warning(f"Line number {line_number} is out of range for file '{file_path}'.")

            # Write the modified lines back to the file
            with open(file_path, 'w') as file:
                file.writelines(lines)

            self.logger.info(f"Code modification completed successfully for '{file_path}'.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to modify code at '{file_path}': {e}", exc_info=True)
            return False

    def commit_changes(self, commit_message: str = "Autonomous code modification") -> bool:
        """
        Commits and pushes the current changes to version control.

        Args:
            commit_message (str, optional): The commit message to use. Defaults to "Autonomous code modification".

        Returns:
            bool: True if commit was successful, False otherwise.
        """
        self.logger.info("Committing changes to version control.")
        try:
            self.version_control.stage_all_changes()
            if self.version_control.has_changes():
                self.version_control.commit(commit_message)
                self.version_control.push()
                self.logger.info("Changes committed and pushed successfully.")
                return True
            else:
                self.logger.info("No changes to commit.")
                return False
        except Exception as e:
            self.logger.error(f"Failed to commit changes: {e}", exc_info=True)
            return False

    def execute_action(self, action: str, params: Dict[str, Any] = {}) -> bool:
        """
        Executes a specified action with given parameters.

        Args:
            action (str): The action to execute ('adjust_config', 'retrain_models', 'modify_code', 'commit_changes').
            params (Dict[str, Any]): Parameters required for the action.

        Returns:
            bool: True if action was successfully executed, False otherwise.
        """
        self.logger.info(f"Executing action '{action}' with parameters: {params}")
        try:
            if action == 'adjust_config':
                return self.adjust_configuration(params.get('config_changes', {}))
            elif action == 'retrain_models':
                return self.retrain_models(params.get('model_types'))
            elif action == 'modify_code':
                return self.modify_code(params.get('file_path', ''), params.get('modifications', {}))
            elif action == 'commit_changes':
                return self.commit_changes(params.get('commit_message', "Autonomous code modification"))
            else:
                self.logger.error(f"Unknown action '{action}'.")
                return False
        except Exception as e:
            self.logger.error(f"Error executing action '{action}': {e}", exc_info=True)
            return False

    def optimize_performance(self):
        """
        Orchestrates the optimization process based on feedback insights.

        This method analyzes feedback, decides which actions to take, and executes them.
        """
        self.logger.info("Starting performance optimization based on feedback insights.")
        try:
            # Run full feedback analysis to update persistent memory
            self.feedback_analyzer.run_full_analysis()

            # Retrieve insights from persistent memory
            insights = self.feedback_analyzer.run_full_analysis()  # Assuming run_full_analysis stores insights

            # Retrieve stored knowledge entries tagged for optimization
            knowledge_df = self.feedback_analyzer.persistent_memory.get_knowledge(tags=['anomaly_detection', 'feedback_analysis'])

            # Example logic based on insights
            for _, entry in knowledge_df.iterrows():
                content = entry.get('content', '')
                # Parse the content to determine required actions
                # This is a simplified example; in production, use structured data
                if 'High CPU usage detected' in content:
                    config_changes = {'cpu_usage_threshold': 70}  # Example adjustment
                    self.adjust_configuration(config_changes)

                if 'Users are experiencing slow response times' in content:
                    self.retrain_models()

                if 'code_optimization_needed' in content:
                    file_path = entry.get('file_path', 'src/modules/nlu/entity_recognizer.py')  # Example path
                    modifications = {
                        'changes': [
                            {'line_number': 42, 'content': 'self.new_attribute = True'},
                            {'line_number': 85, 'content': 'def new_method(self):'}
                        ]
                    }
                    self.modify_code(file_path, modifications)

            # Optionally, commit all changes after optimization
            self.commit_changes("Automated performance optimizations based on feedback insights.")

            self.logger.info("Performance optimization actions executed successfully.")
        except Exception as e:
            self.logger.error(f"Failed to optimize performance: {e}", exc_info=True)

    def run_sample_operations(self):
        """
        Demonstrates sample action executions.
        """
        self.logger.info("Running sample action executions.")

        # Example 1: Adjust Configuration
        config_changes = {
            'roberta_model.embedding_model_name': 'roberta-large',
            'nlu_config.some_parameter': 'new_value'
        }
        success = self.adjust_configuration(config_changes)
        print(f"Configuration Adjustment Successful: {success}")

        # Example 2: Retrain Models
        success = self.retrain_models(['classification', 'qa'])
        print(f"Model Retraining Successful: {success}")

        # Example 3: Modify Code
        file_path = 'src/modules/nlu/entity_recognizer.py'  # Example file path
        modifications = {
            'changes': [
                {'line_number': 42, 'content': 'self.new_attribute = True'},
                {'line_number': 85, 'content': 'def new_method(self):'}
            ]
        }
        success = self.modify_code(file_path, modifications)
        print(f"Code Modification Successful: {success}")

        # Example 4: Commit Changes
        success = self.commit_changes("Automated code modifications by ActionExecutor.")
        print(f"Commit Changes Successful: {success}")

        # Example 5: Execute Various Actions
        # Adjust Configuration
        action_params = {
            'config_changes': {
                'roberta_model.embedding_model_name': 'roberta-large',
                'nlu_config.some_parameter': 'new_value'
            }
        }
        success = self.execute_action('adjust_config', action_params)
        print(f"Action 'adjust_config' Successful: {success}")

        # Retrain Models
        action_params = {
            'model_types': ['classification', 'qa']
        }
        success = self.execute_action('retrain_models', action_params)
        print(f"Action 'retrain_models' Successful: {success}")

        # Modify Code
        action_params = {
            'file_path': 'src/modules/nlu/entity_recognizer.py',
            'modifications': {
                'changes': [
                    {'line_number': 42, 'content': 'self.new_attribute = True'},
                    {'line_number': 85, 'content': 'def new_method(self):'}
                ]
            }
        }
        success = self.execute_action('modify_code', action_params)
        print(f"Action 'modify_code' Successful: {success}")

        # Commit Changes via Execute Action
        action_params = {
            'commit_message': 'Automated commit by ActionExecutor for code modifications.'
        }
        success = self.execute_action('commit_changes', action_params)
        print(f"Action 'commit_changes' Successful: {success}")

        # Optimize Performance
        self.optimize_performance()


# Example Usage and Test Cases
if __name__ == "__main__":
    import os

    # Initialize ActionExecutor
    project_id = os.getenv('PROJECT_ID', 'default_project')  # Ensure PROJECT_ID is set
    action_executor = ActionExecutor(project_id)

    # Run sample operations
    action_executor.run_sample_operations()
