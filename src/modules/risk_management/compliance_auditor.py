# src/modules/risk_management/compliance_auditor.py
import ast
import os
import logging
import json
from typing import List, Dict, Any
from src.utils.configuration_manager import ConfigurationManager
from src.modules.notifications.notification_manager import NotificationManager

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('logs/compliance_auditor.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class ComplianceAuditor:
    """
    Compliance Auditing
    Audits systems and processes to ensure compliance with regulations and internal policies.
    Identifies compliance gaps and recommends corrective actions.
    """

    def __init__(self):
        """
        Initializes the ComplianceAuditor with necessary configurations.
        """
        try:
            self.config_manager = ConfigurationManager()
            self.notification_manager = NotificationManager()
            self.load_auditor_config()
            logger.info("ComplianceAuditor initialized successfully.")
        except Exception as e:
            logger.exception(f"Failed to initialize ComplianceAuditor: {e}")
            raise e

    def load_auditor_config(self):
        """
        Loads auditor configurations from the configuration manager or environment variables.
        """
        logger.info("Loading auditor configurations.")
        try:
            self.auditor_config = {
                'policies_directory': self.config_manager.get('POLICIES_DIRECTORY', 'policies'),
                'systems_to_audit': self.config_manager.get('SYSTEMS_TO_AUDIT', ['system1', 'system2']),
                'audit_report_path': self.config_manager.get('AUDIT_REPORT_PATH', 'reports/audit_report.json'),
                'notification_recipients': self.config_manager.get('NOTIFICATION_RECIPIENTS', '').split(','),
            }
            logger.info(f"Auditor configurations loaded: {self.auditor_config}")
        except Exception as e:
            logger.error(f"Failed to load auditor configurations: {e}")
            raise e

    def perform_audit(self):
        """
        Performs the compliance audit on the specified systems.
        """
        logger.info("Starting compliance audit.")
        try:
            policies = self.load_policies()
            audit_results = []

            for system in self.auditor_config['systems_to_audit']:
                logger.info(f"Auditing system: {system}")
                system_config = self.get_system_configuration(system)
                compliance_result = self.check_compliance(system_config, policies)
                audit_results.append({
                    'system': system,
                    'compliance_result': compliance_result,
                })

            self.generate_audit_report(audit_results)
            self.send_notification(
                subject="Compliance Audit Completed",
                message="The compliance audit has been completed successfully. Please review the audit report."
            )
            logger.info("Compliance audit completed successfully.")
        except Exception as e:
            logger.error(f"Compliance audit failed: {e}")
            self.send_notification(
                subject="Compliance Audit Failed",
                message=f"Compliance audit failed with the following error:\n\n{e}"
            )
            raise e

    def load_policies(self) -> Dict[str, Any]:
        """
        Loads compliance policies from the policies directory.

        :return: Dictionary of policies.
        """
        logger.info("Loading compliance policies.")
        try:
            policies_directory = self.auditor_config['policies_directory']
            policies = {}

            for filename in os.listdir(policies_directory):
                if filename.endswith('.json'):
                    policy_path = os.path.join(policies_directory, filename)
                    with open(policy_path, 'r') as policy_file:
                        policy_name = os.path.splitext(filename)[0]
                        policies[policy_name] = json.load(policy_file)
                        logger.info(f"Policy '{policy_name}' loaded.")

            return policies
        except Exception as e:
            logger.error(f"Failed to load policies: {e}")
            raise e

    def get_system_configuration(self, system: str) -> Dict[str, Any]:
        """
        Retrieves the configuration of a system.

        :param system: Name of the system.
        :return: Configuration dictionary of the system.
        """
        logger.info(f"Retrieving configuration for system '{system}'.")
        try:
            configurations_directory = self.auditor_config.get('configurations_directory', 'configurations')
            system_config_path = os.path.join(configurations_directory, f"{system}.json")

            if not os.path.exists(system_config_path):
                raise FileNotFoundError(
                    f"Configuration file for system '{system}' not found at '{system_config_path}'.")

            with open(system_config_path, 'r') as config_file:
                system_config = json.load(config_file)
            logger.info(f"Configuration for system '{system}' loaded successfully.")
            return system_config
        except Exception as e:
            logger.error(f"Failed to retrieve system configuration: {e}")
            raise e

    def check_compliance(self, system_config: Dict[str, Any], policies: Dict[str, Any]) -> Dict[str, Any]:
        """
        Checks the system configuration against the compliance policies.

        :param system_config: Configuration of the system.
        :param policies: Compliance policies.
        :return: Compliance results with recommendations.
        """
        logger.info(f"Checking compliance for system '{system_config['system_name']}'.")
        try:
            compliance_results = {
                'compliant': True,
                'violations': [],
                'recommendations': []
            }

            for policy_name, policy in policies.items():
                policy_rules = policy.get('rules', [])
                for rule in policy_rules:
                    rule_check = self.evaluate_rule(system_config, rule)
                    if not rule_check['compliant']:
                        compliance_results['compliant'] = False
                        compliance_results['violations'].append(rule_check['violation'])
                        compliance_results['recommendations'].append(rule_check['recommendation'])

            return compliance_results
        except Exception as e:
            logger.error(f"Failed to check compliance: {e}")
            raise e

    import ast

    def evaluate_rule(self, system_config: Dict[str, Any], rule: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates a single compliance rule against the system configuration.

        :param system_config: Configuration of the system.
        :param rule: Compliance rule.
        :return: Result of the rule evaluation.
        """
        try:
            rule_expression = rule.get('expression')
            violation_message = rule.get('violation_message', 'Policy violation detected.')
            recommendation = rule.get('recommendation', 'No recommendation provided.')

            settings = system_config.get('settings', {})

            # Evaluate the expression safely
            result = self.evaluate_expression(rule_expression, settings)

            if not result:
                return {
                    'compliant': False,
                    'violation': violation_message,
                    'recommendation': recommendation
                }
            else:
                return {
                    'compliant': True
                }
        except Exception as e:
            logger.error(f"Failed to evaluate rule: {e}")
            raise e

    def evaluate_expression(self, expression: str, variables: Dict[str, Any]) -> Any:
        """
        Safely evaluates an expression using the given variables.

        :param expression: The expression to evaluate.
        :param variables: A dictionary of variables to use in the evaluation.
        :return: The result of the evaluation.
        """
        try:
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

    def generate_audit_report(self, audit_results: List[Dict[str, Any]]):
        """
        Generates the audit report and saves it to a file.

        :param audit_results: Results of the compliance audit.
        """
        logger.info("Generating audit report.")
        try:
            report_path = self.auditor_config['audit_report_path']
            os.makedirs(os.path.dirname(report_path), exist_ok=True)

            with open(report_path, 'w') as report_file:
                json.dump(audit_results, report_file, indent=4)
            logger.info(f"Audit report saved to '{report_path}'.")
        except Exception as e:
            logger.error(f"Failed to generate audit report: {e}")
            raise e

    def send_notification(self, subject: str, message: str):
        """
        Sends a notification to the configured recipients.

        :param subject: Subject of the notification.
        :param message: Body of the notification.
        """
        try:
            recipients = self.auditor_config['notification_recipients']
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
    Demonstrates example usage of the ComplianceAuditor class.
    """
    try:
        # Initialize ComplianceAuditor
        auditor = ComplianceAuditor()

        # Perform the compliance audit
        auditor.perform_audit()

    except Exception as e:
        logger.exception(f"Error in example usage: {e}")

# --------------------- Main Execution --------------------- #

if __name__ == "__main__":
    # Run the compliance auditor example
    example_usage()
