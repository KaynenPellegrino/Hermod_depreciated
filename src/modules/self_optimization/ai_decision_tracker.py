# src/modules/self_optimization/ai_decision_tracker.py

import logging
import json
import os
from typing import Dict, Any, List

from src.utils.logger import get_logger
from src.utils.configuration_manager import ConfigurationManager

class AIDecisionTracker:
    """
    Monitors and logs decisions made by Hermod's AI components.
    Provides transparency into the AI's decision-making process for debugging and accountability.
    """

    def __init__(self, project_id: str):
        """
        Initializes the AIDecisionTracker.

        Args:
            project_id (str): Unique identifier for the project.
        """
        self.logger = get_logger(__name__)
        self.project_id = project_id
        self.config_manager = ConfigurationManager()
        self.config = self.config_manager.get_configuration(project_id)

        self.decision_log_path = self.config.get('decision_log_path', 'logs/ai_decisions.log')

        # Ensure the decision log file exists
        os.makedirs(os.path.dirname(self.decision_log_path), exist_ok=True)
        if not os.path.exists(self.decision_log_path):
            with open(self.decision_log_path, 'w') as f:
                json.dump([], f)

        self.logger.info(f"AIDecisionTracker initialized for project '{project_id}'.")

    def log_decision(self, decision: str, justification: str, metadata: Dict[str, Any] = {}):
        """
        Logs an AI decision with justification and optional metadata.

        Args:
            decision (str): Description of the decision made.
            justification (str): Reasoning behind the decision.
            metadata (Dict[str, Any], optional): Additional data related to the decision. Defaults to {}.
        """
        self.logger.info(f"Logging AI decision: {decision}")
        decision_entry = {
            'decision': decision,
            'justification': justification,
            'metadata': metadata,
            'timestamp': self._get_current_timestamp()
        }

        try:
            with open(self.decision_log_path, 'r+') as f:
                data = json.load(f)
                data.append(decision_entry)
                f.seek(0)
                json.dump(data, f, indent=4)
            self.logger.debug(f"Decision logged: {decision_entry}")
        except Exception as e:
            self.logger.error(f"Failed to log decision: {e}", exc_info=True)

    def retrieve_decisions(self, criteria: Dict[str, Any] = {}) -> List[Dict[str, Any]]:
        """
        Retrieves logged decisions based on given criteria.

        Args:
            criteria (Dict[str, Any], optional): Filters to apply on the decisions. Defaults to {}.

        Returns:
            List[Dict[str, Any]]: List of decision entries matching the criteria.
        """
        self.logger.info(f"Retrieving decisions with criteria: {criteria}")
        try:
            with open(self.decision_log_path, 'r') as f:
                data = json.load(f)

            if not criteria:
                return data

            filtered_decisions = []
            for entry in data:
                match = True
                for key, value in criteria.items():
                    if entry.get(key) != value:
                        match = False
                        break
                if match:
                    filtered_decisions.append(entry)

            self.logger.debug(f"Retrieved {len(filtered_decisions)} decisions matching criteria.")
            return filtered_decisions
        except Exception as e:
            self.logger.error(f"Failed to retrieve decisions: {e}", exc_info=True)
            return []

    def export_decisions(self, export_path: str, format: str = 'json') -> bool:
        """
        Exports logged decisions to a specified format.

        Args:
            export_path (str): Path to export the decisions.
            format (str, optional): Format to export ('json', 'csv'). Defaults to 'json'.

        Returns:
            bool: True if export was successful, False otherwise.
        """
        self.logger.info(f"Exporting decisions to '{export_path}' in format '{format}'.")
        try:
            with open(self.decision_log_path, 'r') as f:
                data = json.load(f)

            if format == 'json':
                with open(export_path, 'w') as f_export:
                    json.dump(data, f_export, indent=4)
            elif format == 'csv':
                import csv
                keys = data[0].keys() if data else []
                with open(export_path, 'w', newline='') as f_export:
                    dict_writer = csv.DictWriter(f_export, fieldnames=keys)
                    dict_writer.writeheader()
                    dict_writer.writerows(data)
            else:
                self.logger.error(f"Unsupported export format: {format}")
                return False

            self.logger.info("Decisions exported successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to export decisions: {e}", exc_info=True)
            return False

    def _get_current_timestamp(self) -> str:
        """
        Retrieves the current timestamp in ISO format.

        Returns:
            str: Current timestamp.
        """
        from datetime import datetime
        return datetime.utcnow().isoformat() + 'Z'

# Example Usage and Test Cases
if __name__ == "__main__":
    # Initialize AIDecisionTracker
    project_id = "proj_12345"  # Replace with your actual project ID
    decision_tracker = AIDecisionTracker(project_id)

    # Log a decision
    decision_tracker.log_decision(
        decision="Refactor NLU Engine",
        justification="Detected high complexity and frequent errors in nlu_engine.py",
        metadata={"complexity": 10, "error_rate": 0.05}
    )

    # Retrieve all decisions
    all_decisions = decision_tracker.retrieve_decisions()
    print(f"All Decisions: {all_decisions}")

    # Retrieve decisions with specific criteria
    high_risk_decisions = decision_tracker.retrieve_decisions({"metadata": {"complexity": 10}})
    print(f"High Risk Decisions: {high_risk_decisions}")

    # Export decisions to JSON
    export_success = decision_tracker.export_decisions('exports/ai_decisions_export.json', 'json')
    print(f"Export to JSON Successful: {export_success}")

    # Export decisions to CSV
    export_success = decision_tracker.export_decisions('exports/ai_decisions_export.csv', 'csv')
    print(f"Export to CSV Successful: {export_success}")
