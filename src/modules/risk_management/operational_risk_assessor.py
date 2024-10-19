# src/modules/risk_management/operational_risk_assessor.py

import os
import logging
import json
from typing import List, Dict, Any
from src.utils.configuration_manager import ConfigurationManager
from src.modules.notifications.notification_manager import NotificationManager

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('logs/operational_risk_assessor.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class OperationalRiskAssessor:
    """
    Operational Risk Assessment
    Evaluates risks associated with operational processes, including system failures,
    process inefficiencies, or human errors. Helps in mitigating potential operational risks.
    """

    def __init__(self):
        """
        Initializes the OperationalRiskAssessor with necessary configurations.
        """
        try:
            self.config_manager = ConfigurationManager()
            self.notification_manager = NotificationManager()
            self.load_assessor_config()
            logger.info("OperationalRiskAssessor initialized successfully.")
        except Exception as e:
            logger.exception(f"Failed to initialize OperationalRiskAssessor: {e}")
            raise e

    def load_assessor_config(self):
        """
        Loads assessor configurations from the configuration manager or environment variables.
        """
        logger.info("Loading assessor configurations.")
        try:
            self.assessor_config = {
                'risk_definitions_file': self.config_manager.get('RISK_DEFINITIONS_FILE', 'risks/risk_definitions.json'),
                'systems_to_assess': self.config_manager.get('SYSTEMS_TO_ASSESS', ['system1', 'system2']),
                'risk_assessment_report_path': self.config_manager.get('RISK_ASSESSMENT_REPORT_PATH', 'reports/risk_assessment_report.json'),
                'notification_recipients': self.config_manager.get('NOTIFICATION_RECIPIENTS', '').split(','),
            }
            logger.info(f"Assessor configurations loaded: {self.assessor_config}")
        except Exception as e:
            logger.error(f"Failed to load assessor configurations: {e}")
            raise e

    def perform_risk_assessment(self):
        """
        Performs operational risk assessment on the specified systems.
        """
        logger.info("Starting operational risk assessment.")
        try:
            risk_definitions = self.load_risk_definitions()
            assessment_results = []

            for system in self.assessor_config['systems_to_assess']:
                logger.info(f"Assessing risks for system: {system}")
                system_data = self.get_system_data(system)
                risk_result = self.evaluate_risks(system_data, risk_definitions)
                assessment_results.append({
                    'system': system,
                    'risk_result': risk_result,
                })

            self.generate_risk_assessment_report(assessment_results)
            self.send_notification(
                subject="Operational Risk Assessment Completed",
                message="The operational risk assessment has been completed successfully. Please review the risk assessment report."
            )
            logger.info("Operational risk assessment completed successfully.")
        except Exception as e:
            logger.error(f"Operational risk assessment failed: {e}")
            self.send_notification(
                subject="Operational Risk Assessment Failed",
                message=f"Operational risk assessment failed with the following error:\n\n{e}"
            )
            raise e

    def load_risk_definitions(self) -> List[Dict[str, Any]]:
        """
        Loads risk definitions from the risk definitions file.

        :return: List of risk definitions.
        """
        logger.info("Loading risk definitions.")
        try:
            risk_definitions_file = self.assessor_config['risk_definitions_file']
            if not os.path.exists(risk_definitions_file):
                raise FileNotFoundError(f"Risk definitions file not found at '{risk_definitions_file}'.")

            with open(risk_definitions_file, 'r') as file:
                risk_definitions = json.load(file)
            logger.info("Risk definitions loaded successfully.")
            return risk_definitions
        except Exception as e:
            logger.error(f"Failed to load risk definitions: {e}")
            raise e

    def get_system_data(self, system: str) -> Dict[str, Any]:
        """
        Retrieves operational data of a system.

        :param system: Name of the system.
        :return: Operational data dictionary of the system.
        """
        logger.info(f"Retrieving operational data for system '{system}'.")
        try:
            data_directory = self.assessor_config.get('data_directory', 'data')
            system_data_path = os.path.join(data_directory, f"{system}_data.json")

            if not os.path.exists(system_data_path):
                raise FileNotFoundError(f"Data file for system '{system}' not found at '{system_data_path}'.")

            with open(system_data_path, 'r') as data_file:
                system_data = json.load(data_file)
            logger.info(f"Operational data for system '{system}' loaded successfully.")
            return system_data
        except Exception as e:
            logger.error(f"Failed to retrieve system data: {e}")
            raise e

    def evaluate_risks(self, system_data: Dict[str, Any], risk_definitions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluates the system data against the risk definitions.

        :param system_data: Operational data of the system.
        :param risk_definitions: List of risk definitions.
        :return: Risk evaluation results with recommendations.
        """
        logger.info(f"Evaluating risks for system '{system_data['system_name']}'.")
        try:
            risk_results = {
                'risks_identified': [],
                'recommendations': []
            }

            for risk in risk_definitions:
                risk_check = self.evaluate_risk(system_data, risk)
                if risk_check['risk_identified']:
                    risk_results['risks_identified'].append(risk_check['risk'])
                    risk_results['recommendations'].append(risk_check['recommendation'])

            return risk_results
        except Exception as e:
            logger.error(f"Failed to evaluate risks: {e}")
            raise e

    def evaluate_risk(self, system_data: Dict[str, Any], risk: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates a single risk definition against the system data.

        :param system_data: Operational data of the system.
        :param risk: Risk definition.
        :return: Result of the risk evaluation.
        """
        try:
            risk_expression = risk.get('expression')
            risk_description = risk.get('description', 'No description provided.')
            recommendation = risk.get('recommendation', 'No recommendation provided.')

            variables = system_data.get('metrics', {})

            # Evaluate the expression safely
            result = self.evaluate_expression(risk_expression, variables)

            if result:
                return {
                    'risk_identified': True,
                    'risk': risk_description,
                    'recommendation': recommendation
                }
            else:
                return {
                    'risk_identified': False
                }
        except Exception as e:
            logger.error(f"Failed to evaluate risk: {e}")
            raise e

    def evaluate_expression(self, expression: str, variables: Dict[str, Any]) -> Any:
        """
        Safely evaluates an expression using the given variables.

        :param expression: The expression to evaluate.
        :param variables: A dictionary of variables to use in the evaluation.
        :return: The result of the evaluation.
        """
        try:
            # Use the same safe evaluation as in compliance_auditor.py
            import ast

            # Parse the expression into an AST node
            expression_ast = ast.parse(expression, mode='eval')

            # Define a visitor to ensure the AST contains only safe nodes
            class SafeEval(ast.NodeVisitor):
                SAFE_NODES = (
                    ast.Expression, ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Compare,
                    ast.Num, ast.Str, ast.NameConstant, ast.Name, ast.Load,
                    ast.List, ast.Tuple, ast.Dict, ast.Set,
                    ast.And, ast.Or, ast.Not, ast.Eq, ast.NotEq, ast.Lt, ast.LtE,
                    ast.Gt, ast.GtE, ast.Is, ast.IsNot, ast.In, ast.NotIn,
                    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
                    ast.UAdd, ast.USub, ast.Call, ast.keyword, ast.IfExp
                )

                def visit(self, node):
                    if not isinstance(node, self.SAFE_NODES):
                        raise ValueError(f"Unsafe expression: {ast.dump(node)}")
                    return super().visit(node)

                def visit_Call(self, node):
                    # Disallow all function calls
                    raise ValueError("Function calls are not allowed.")

            # Visit the AST to ensure it's safe
            SafeEval().visit(expression_ast)

            # Evaluate the expression
            result = eval(compile(expression_ast, filename="", mode="eval"), {"__builtins__": {}}, variables)
            return result
        except Exception as e:
            logger.error(f"Error evaluating expression '{expression}': {e}")
            raise ValueError(f"Error evaluating expression '{expression}': {e}")

    def generate_risk_assessment_report(self, assessment_results: List[Dict[str, Any]]):
        """
        Generates the risk assessment report and saves it to a file.

        :param assessment_results: Results of the risk assessment.
        """
        logger.info("Generating risk assessment report.")
        try:
            report_path = self.assessor_config['risk_assessment_report_path']
            os.makedirs(os.path.dirname(report_path), exist_ok=True)

            with open(report_path, 'w') as report_file:
                json.dump(assessment_results, report_file, indent=4)
            logger.info(f"Risk assessment report saved to '{report_path}'.")
        except Exception as e:
            logger.error(f"Failed to generate risk assessment report: {e}")
            raise e

    def send_notification(self, subject: str, message: str):
        """
        Sends a notification to the configured recipients.

        :param subject: Subject of the notification.
        :param message: Body of the notification.
        """
        try:
            recipients = self.assessor_config['notification_recipients']
            if recipients:
                self.notification_manager.send_notification(
                    recipients=recipients,
                    subject=subject,
                    message=message
                )
                logger.info("Notification sent successfully.")
            else:
                logger.warning("No notification recipients configured.")
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")

# --------------------- Example Usage --------------------- #

def example_usage():
    """
    Demonstrates example usage of the OperationalRiskAssessor class.
    """
    try:
        # Initialize OperationalRiskAssessor
        assessor = OperationalRiskAssessor()

        # Perform the operational risk assessment
        assessor.perform_risk_assessment()

    except Exception as e:
        logger.exception(f"Error in example usage: {e}")

# --------------------- Main Execution --------------------- #

if __name__ == "__main__":
    # Run the operational risk assessor example
    example_usage()
