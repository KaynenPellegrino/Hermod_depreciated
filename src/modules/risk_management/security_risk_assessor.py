# src/modules/risk_management/security_risk_assessor.py

import os
import logging
import json
from typing import List, Dict, Any
from src.utils.configuration_manager import ConfigurationManager
from src.modules.notifications.notification_manager import NotificationManager

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('logs/security_risk_assessor.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class SecurityRiskAssessor:
    """
    Security Risk Analysis
    Assesses security risks by analyzing threats, vulnerabilities, and potential impacts.
    Informs security planning and risk mitigation strategies.
    """

    def __init__(self):
        """
        Initializes the SecurityRiskAssessor with necessary configurations.
        """
        try:
            self.config_manager = ConfigurationManager()
            self.notification_manager = NotificationManager()
            self.load_assessor_config()
            logger.info("SecurityRiskAssessor initialized successfully.")
        except Exception as e:
            logger.exception(f"Failed to initialize SecurityRiskAssessor: {e}")
            raise e

    def load_assessor_config(self):
        """
        Loads assessor configurations from the configuration manager or environment variables.
        """
        logger.info("Loading assessor configurations.")
        try:
            self.assessor_config = {
                'threat_model_file': self.config_manager.get('THREAT_MODEL_FILE', 'security/threat_model.json'),
                'systems_to_assess': self.config_manager.get('SYSTEMS_TO_ASSESS', ['system1', 'system2']),
                'security_assessment_report_path': self.config_manager.get('SECURITY_ASSESSMENT_REPORT_PATH', 'reports/security_assessment_report.json'),
                'notification_recipients': self.config_manager.get('NOTIFICATION_RECIPIENTS', '').split(','),
            }
            logger.info(f"Assessor configurations loaded: {self.assessor_config}")
        except Exception as e:
            logger.error(f"Failed to load assessor configurations: {e}")
            raise e

    def perform_security_assessment(self):
        """
        Performs security risk assessment on the specified systems.
        """
        logger.info("Starting security risk assessment.")
        try:
            threat_model = self.load_threat_model()
            assessment_results = []

            for system in self.assessor_config['systems_to_assess']:
                logger.info(f"Assessing security risks for system: {system}")
                system_data = self.get_system_data(system)
                risk_result = self.evaluate_security_risks(system_data, threat_model)
                assessment_results.append({
                    'system': system,
                    'risk_result': risk_result,
                })

            self.generate_security_assessment_report(assessment_results)
            self.send_notification(
                subject="Security Risk Assessment Completed",
                message="The security risk assessment has been completed successfully. Please review the security assessment report."
            )
            logger.info("Security risk assessment completed successfully.")
        except Exception as e:
            logger.error(f"Security risk assessment failed: {e}")
            self.send_notification(
                subject="Security Risk Assessment Failed",
                message=f"Security risk assessment failed with the following error:\n\n{e}"
            )
            raise e

    def load_threat_model(self) -> List[Dict[str, Any]]:
        """
        Loads the threat model from the threat model file.

        :return: List of threats and vulnerabilities.
        """
        logger.info("Loading threat model.")
        try:
            threat_model_file = self.assessor_config['threat_model_file']
            if not os.path.exists(threat_model_file):
                raise FileNotFoundError(f"Threat model file not found at '{threat_model_file}'.")

            with open(threat_model_file, 'r') as file:
                threat_model = json.load(file)
            logger.info("Threat model loaded successfully.")
            return threat_model
        except Exception as e:
            logger.error(f"Failed to load threat model: {e}")
            raise e

    def get_system_data(self, system: str) -> Dict[str, Any]:
        """
        Retrieves security data of a system.

        :param system: Name of the system.
        :return: Security data dictionary of the system.
        """
        logger.info(f"Retrieving security data for system '{system}'.")
        try:
            data_directory = self.assessor_config.get('data_directory', 'data')
            system_data_path = os.path.join(data_directory, f"{system}_security_data.json")

            if not os.path.exists(system_data_path):
                raise FileNotFoundError(f"Security data file for system '{system}' not found at '{system_data_path}'.")

            with open(system_data_path, 'r') as data_file:
                system_data = json.load(data_file)
            logger.info(f"Security data for system '{system}' loaded successfully.")
            return system_data
        except Exception as e:
            logger.error(f"Failed to retrieve system data: {e}")
            raise e

    def evaluate_security_risks(self, system_data: Dict[str, Any], threat_model: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluates the system data against the threat model.

        :param system_data: Security data of the system.
        :param threat_model: List of threats and vulnerabilities.
        :return: Risk evaluation results with recommendations.
        """
        logger.info(f"Evaluating security risks for system '{system_data['system_name']}'.")
        try:
            risk_results = {
                'risks_identified': [],
                'recommendations': []
            }

            for threat in threat_model:
                risk_check = self.evaluate_threat(system_data, threat)
                if risk_check['risk_identified']:
                    risk_results['risks_identified'].append(risk_check['risk'])
                    risk_results['recommendations'].append(risk_check['recommendation'])

            return risk_results
        except Exception as e:
            logger.error(f"Failed to evaluate security risks: {e}")
            raise e

    def evaluate_threat(self, system_data: Dict[str, Any], threat: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates a single threat against the system data.

        :param system_data: Security data of the system.
        :param threat: Threat definition.
        :return: Result of the threat evaluation.
        """
        try:
            threat_expression = threat.get('expression')
            threat_description = threat.get('description', 'No description provided.')
            recommendation = threat.get('recommendation', 'No recommendation provided.')

            variables = system_data.get('security_metrics', {})

            # Evaluate the expression safely
            result = self.evaluate_expression(threat_expression, variables)

            if result:
                return {
                    'risk_identified': True,
                    'risk': threat_description,
                    'recommendation': recommendation
                }
            else:
                return {
                    'risk_identified': False
                }
        except Exception as e:
            logger.error(f"Failed to evaluate threat: {e}")
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

    def generate_security_assessment_report(self, assessment_results: List[Dict[str, Any]]):
        """
        Generates the security assessment report and saves it to a file.

        :param assessment_results: Results of the security risk assessment.
        """
        logger.info("Generating security assessment report.")
        try:
            report_path = self.assessor_config['security_assessment_report_path']
            os.makedirs(os.path.dirname(report_path), exist_ok=True)

            with open(report_path, 'w') as report_file:
                json.dump(assessment_results, report_file, indent=4)
            logger.info(f"Security assessment report saved to '{report_path}'.")
        except Exception as e:
            logger.error(f"Failed to generate security assessment report: {e}")
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
    Demonstrates example usage of the SecurityRiskAssessor class.
    """
    try:
        # Initialize SecurityRiskAssessor
        assessor = SecurityRiskAssessor()

        # Perform the security risk assessment
        assessor.perform_security_assessment()

    except Exception as e:
        logger.exception(f"Error in example usage: {e}")

# --------------------- Main Execution --------------------- #

if __name__ == "__main__":
    # Run the security risk assessor example
    example_usage()
