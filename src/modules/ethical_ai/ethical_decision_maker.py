# src/modules/ethical_ai/ethical_decision_maker.py

import logging
from typing import Any, Dict, Optional
from src.utils.configuration_manager import ConfigurationManager
from src.modules.notifications.notification_manager import NotificationManager

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('logs/ethical_decision_maker.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class EthicalDecisionMaker:
    """
    Ethical Decision Framework
    Implements frameworks and algorithms that guide AI systems to make decisions aligned with ethical principles.
    Ensures that AI behaviors adhere to predefined ethical standards.
    """

    def __init__(self):
        """
        Initializes the EthicalDecisionMaker with necessary configurations.
        """
        try:
            self.config_manager = ConfigurationManager()
            self.notification_manager = NotificationManager()
            self.load_ethics_config()
            logger.info("EthicalDecisionMaker initialized successfully.")
        except Exception as e:
            logger.exception(f"Failed to initialize EthicalDecisionMaker: {e}")
            raise e

    def load_ethics_config(self):
        """
        Loads ethical decision configurations from the configuration manager or environment variables.
        """
        logger.info("Loading ethical decision configurations.")
        try:
            self.ethics_config = {
                'ethical_principles_file': self.config_manager.get('ETHICAL_PRINCIPLES_FILE', 'data/ethical_principles.json'),
                'notification_recipients': self.config_manager.get('NOTIFICATION_RECIPIENTS', '').split(','),
            }
            # Load ethical principles
            self.ethical_principles = self.load_ethical_principles()
            logger.info(f"Ethical decision configurations loaded: {self.ethics_config}")
        except Exception as e:
            logger.error(f"Failed to load ethical decision configurations: {e}")
            raise e

    def load_ethical_principles(self) -> Dict[str, Any]:
        """
        Loads the ethical principles from a JSON file.

        :return: Dictionary of ethical principles.
        """
        logger.info("Loading ethical principles.")
        try:
            import json
            ethical_principles_file = self.ethics_config['ethical_principles_file']
            with open(ethical_principles_file, 'r') as f:
                principles = json.load(f)
            logger.info("Ethical principles loaded successfully.")
            return principles
        except Exception as e:
            logger.error(f"Failed to load ethical principles: {e}")
            raise e

    def evaluate_decision(self, decision_context: Dict[str, Any]) -> bool:
        """
        Evaluates a decision based on the ethical principles.

        :param decision_context: The context of the decision to be evaluated.
        :return: True if the decision adheres to ethical principles, False otherwise.
        """
        logger.info("Evaluating decision against ethical principles.")
        try:
            # Iterate over each ethical principle
            for principle in self.ethical_principles.get('principles', []):
                rule = principle.get('rule')
                if not self.apply_rule(rule, decision_context):
                    logger.info(f"Decision violates the principle: {principle.get('name')}")
                    return False
            logger.info("Decision adheres to all ethical principles.")
            return True
        except Exception as e:
            logger.error(f"Failed to evaluate decision: {e}")
            raise e

    def apply_rule(self, rule: Dict[str, Any], decision_context: Dict[str, Any]) -> bool:
        """
        Applies an ethical rule to the decision context.

        :param rule: The ethical rule to apply.
        :param decision_context: The context of the decision.
        :return: True if the rule is satisfied, False otherwise.
        """
        try:
            condition = rule.get('condition')
            if condition:
                # Safely evaluate the condition using a custom evaluator
                result = self.evaluate_condition(condition, decision_context)
                return result
            else:
                logger.warning("No condition specified in the rule.")
                return True  # If no condition, assume rule is satisfied
        except Exception as e:
            logger.error(f"Failed to apply rule: {e}")
            raise e

    def evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """
        Safely evaluates a condition expression against the decision context.

        :param condition: The condition expression to evaluate.
        :param context: The decision context with variables.
        :return: The result of the condition evaluation.
        """
        try:
            # Define allowed names and operators
            allowed_names = {
                'True': True,
                'False': False,
                'and': lambda a, b: a and b,
                'or': lambda a, b: a or b,
                'not': lambda a: not a,
            }

            # Add context variables to allowed names
            for key, value in context.items():
                allowed_names[key] = value

            # Parse the condition using ast
            import ast

            class SafeEvaluator(ast.NodeVisitor):
                def visit_Name(self, node):
                    if node.id in allowed_names:
                        return allowed_names[node.id]
                    else:
                        raise ValueError(f"Use of undefined variable '{node.id}' in condition.")

                def visit_BoolOp(self, node):
                    op_type = type(node.op)
                    if op_type is ast.And:
                        return all(self.visit(value) for value in node.values)
                    elif op_type is ast.Or:
                        return any(self.visit(value) for value in node.values)
                    else:
                        raise ValueError("Unsupported boolean operator.")

                def visit_UnaryOp(self, node):
                    if isinstance(node.op, ast.Not):
                        return not self.visit(node.operand)
                    else:
                        raise ValueError("Unsupported unary operator.")

                def visit_Compare(self, node):
                    left = self.visit(node.left)
                    comparisons = []
                    for op, comparator in zip(node.ops, node.comparators):
                        right = self.visit(comparator)
                        if isinstance(op, ast.Eq):
                            comparisons.append(left == right)
                        elif isinstance(op, ast.NotEq):
                            comparisons.append(left != right)
                        elif isinstance(op, ast.Lt):
                            comparisons.append(left < right)
                        elif isinstance(op, ast.LtE):
                            comparisons.append(left <= right)
                        elif isinstance(op, ast.Gt):
                            comparisons.append(left > right)
                        elif isinstance(op, ast.GtE):
                            comparisons.append(left >= right)
                        else:
                            raise ValueError("Unsupported comparison operator.")
                        left = right
                    return all(comparisons)

                def visit_Constant(self, node):
                    return node.value

                def visit_NameConstant(self, node):
                    return node.value

                def visit_Call(self, node):
                    raise ValueError("Function calls are not allowed in conditions.")

                def generic_visit(self, node):
                    raise ValueError(f"Unsupported expression: {ast.dump(node)}")

            expr_ast = ast.parse(condition, mode='eval')
            evaluator = SafeEvaluator()
            result = evaluator.visit(expr_ast.body)
            return result
        except Exception as e:
            logger.error(f"Failed to evaluate condition '{condition}': {e}")
            raise e

    def make_decision(self, decision_context: Dict[str, Any]) -> Any:
        """
        Makes a decision based on the context and ethical principles.

        :param decision_context: The context of the decision to be made.
        :return: The decision output, or None if the decision is unethical.
        """
        logger.info("Making decision based on ethical principles.")
        try:
            if self.evaluate_decision(decision_context):
                # Proceed with the decision
                decision_output = self.execute_decision(decision_context)
                logger.info("Decision made successfully.")
                return decision_output
            else:
                # Decision is unethical
                logger.warning("Decision is unethical and has been blocked.")
                self.send_notification(
                    subject="Unethical Decision Blocked",
                    message="An unethical decision was blocked by the EthicalDecisionMaker."
                )
                return None
        except Exception as e:
            logger.error(f"Failed to make decision: {e}")
            raise e

    def execute_decision(self, decision_context: Dict[str, Any]) -> Any:
        """
        Executes the decision logic.

        :param decision_context: The context of the decision.
        :return: The decision output.
        """
        try:
            action = decision_context.get('action')
            if action == 'share_user_data':
                # Implement logic to share user data
                user_data = decision_context.get('user_data')
                target = decision_context.get('target')
                if user_data and target:
                    # Simulate sharing data
                    logger.info(f"Sharing user data with {target}")
                    # In a real implementation, include code to share data securely
                    return f"User data shared with {target}"
                else:
                    logger.warning("User data or target not specified.")
                    return None
            elif action == 'provide_service':
                # Implement logic to provide a service
                service = decision_context.get('service')
                if service:
                    logger.info(f"Providing service: {service}")
                    # In a real implementation, include code to provide the service
                    return f"Service '{service}' provided"
                else:
                    logger.warning("Service not specified.")
                    return None
            else:
                logger.warning(f"Unknown action: {action}")
                return None
        except Exception as e:
            logger.error(f"Failed to execute decision: {e}")
            raise e

    def send_notification(self, subject: str, message: str):
        """
        Sends a notification to the configured recipients.

        :param subject: Subject of the notification.
        :param message: Body of the notification.
        """
        try:
            recipients = self.ethics_config['notification_recipients']
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
    Demonstrates example usage of the EthicalDecisionMaker class.
    """
    try:
        # Initialize EthicalDecisionMaker
        decision_maker = EthicalDecisionMaker()

        # Example decision context
        decision_context = {
            'action': 'share_user_data',
            'causes_harm': False,
            'user_consent': True,
            'user_data': {'name': 'John Doe', 'email': 'john@example.com'},
            'target': 'partner_service'
        }

        # Make a decision
        decision_output = decision_maker.make_decision(decision_context)

        if decision_output:
            print("Decision Output:")
            print(decision_output)
        else:
            print("Decision was deemed unethical and was not executed.")

    except Exception as e:
        logger.exception(f"Error in example usage: {e}")

# --------------------- Main Execution --------------------- #

if __name__ == "__main__":
    # Run the ethical decision maker example
    example_usage()
