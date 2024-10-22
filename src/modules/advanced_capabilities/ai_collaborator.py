# src/modules/advanced_capabilities/ai_collaborator.py

import logging
from typing import List, Dict, Any

from src.utils.logger import get_logger
from src.modules.code_generation.code_generator import CodeGenerator
from src.modules.nlu.nlu_engine import NLUEngine
from src.modules.code_generation.template_manager import TemplateManager
from src.utils.helpers import format_code_snippet

class AICollaborator:
    """
    AICollaborator provides developers with code suggestions, debugging assistance,
    and intelligent code reviews to enhance the development process.
    """

    def __init__(self):
        # Initialize the logger
        self.logger = get_logger(__name__)

        # Initialize the Code Generator for suggestions and analysis
        self.code_generator = CodeGenerator()

        # Initialize the NLU Engine for understanding developer queries
        self.nlu_engine = NLUEngine()

        # Initialize the Template Manager for accessing code templates
        self.template_manager = TemplateManager()

        self.logger.info("AICollaborator initialized successfully.")

    def suggest_code(self, context: str, partial_code: str) -> Dict[str, Any]:
        """
        Provides code suggestions based on the current context and partial code.

        Args:
            context (str): The context or description of what the developer is trying to achieve.
            partial_code (str): The current partial code written by the developer.

        Returns:
            Dict[str, Any]: A dictionary containing suggested code snippets and explanations.
        """
        self.logger.debug(f"Suggesting code with context: '{context}' and partial_code: '{partial_code}'")

        try:
            # Use the Code Generator to generate suggestions
            suggestions = self.code_generator.generate_suggestions(context, partial_code)
            self.logger.info(f"Generated {len(suggestions)} code suggestions.")

            # Format the suggestions for better readability
            formatted_suggestions = [
                {
                    "snippet": format_code_snippet(suggestion['code']),
                    "explanation": suggestion.get('explanation', 'No explanation provided.')
                }
                for suggestion in suggestions
            ]

            return {
                "status": "success",
                "suggestions": formatted_suggestions
            }

        except Exception as e:
            self.logger.error(f"Error in suggesting code: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": "An error occurred while generating code suggestions."
            }

    def debug_code(self, code: str) -> Dict[str, Any]:
        """
        Assists in debugging by analyzing the provided code and identifying potential issues.

        Args:
            code (str): The code snippet to be debugged.

        Returns:
            Dict[str, Any]: A dictionary containing identified issues and suggested fixes.
        """
        self.logger.debug(f"Debugging code:\n{code}")

        try:
            # Use the Code Generator to analyze code
            issues = self.code_generator.analyze_code(code)

            if not issues:
                self.logger.info("No issues found in the provided code.")
                return {
                    "status": "success",
                    "issues": [],
                    "message": "No issues found in the code."
                }

            # Format the issues with explanations and possible fixes
            formatted_issues = [
                {
                    "issue": issue.get('description', 'No description provided.'),
                    "location": issue.get('location', 'Unknown'),
                    "suggested_fix": issue.get('suggested_fix', 'No fix available.')
                }
                for issue in issues
            ]

            self.logger.info(f"Found {len(formatted_issues)} issues in the code.")

            return {
                "status": "success",
                "issues": formatted_issues
            }

        except Exception as e:
            self.logger.error(f"Error in debugging code: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": "An error occurred while debugging the code."
            }

    def review_code(self, code: str) -> Dict[str, Any]:
        """
        Conducts an intelligent code review, highlighting best practices, potential optimizations,
        and adherence to coding standards.

        Args:
            code (str): The code snippet to be reviewed.

        Returns:
            Dict[str, Any]: A dictionary containing review comments and improvement suggestions.
        """
        self.logger.debug(f"Reviewing code:\n{code}")

        try:
            # Use the Code Generator to review code
            review_comments = self.code_generator.review_code(code)

            if not review_comments:
                self.logger.info("No review comments generated for the provided code.")
                return {
                    "status": "success",
                    "review_comments": [],
                    "message": "No review comments generated."
                }

            # Format the review comments
            formatted_comments = [
                {
                    "comment": comment.get('comment', 'No comment provided.'),
                    "line": comment.get('line', 'Unknown'),
                    "suggestion": comment.get('suggestion', 'No suggestion provided.')
                }
                for comment in review_comments
            ]

            self.logger.info(f"Generated {len(formatted_comments)} review comments.")

            return {
                "status": "success",
                "review_comments": formatted_comments
            }

        except Exception as e:
            self.logger.error(f"Error in reviewing code: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": "An error occurred while reviewing the code."
            }

    def intelligent_assist(self, query: str, code: str = "") -> Dict[str, Any]:
        """
        Provides intelligent assistance based on developer's natural language queries.

        Args:
            query (str): The developer's natural language query.
            code (str, optional): The current code snippet relevant to the query.

        Returns:
            Dict[str, Any]: A dictionary containing the assistance provided.
        """
        self.logger.debug(f"Handling intelligent assist with query: '{query}' and code: '{code}'")

        try:
            # Parse the query using NLU Engine
            parsed_query = self.nlu_engine.parse(query)
            self.logger.debug(f"Parsed query: {parsed_query}")

            intent = parsed_query.get('intent', {}).get('name', '').lower()

            if intent == 'suggest_code':
                context = parsed_query.get('entities', {}).get('context', '')
                return self.suggest_code(context, code)
            elif intent == 'debug_code':
                return self.debug_code(code)
            elif intent == 'review_code':
                return self.review_code(code)
            else:
                self.logger.warning(f"Unknown intent: '{intent}'")
                return {
                    "status": "failure",
                    "message": "I'm sorry, I didn't understand that request."
                }

        except Exception as e:
            self.logger.error(f"Error in intelligent assist: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "message": "An error occurred while providing assistance."
            }


if __name__ == "__main__":
    # Example usage
    collaborator = AICollaborator()

    # Example 1: Code Suggestion
    context = "Implement a REST API endpoint for user authentication."
    partial_code = "def authenticate_user(username, password):"
    suggestion_result = collaborator.suggest_code(context, partial_code)
    print("Code Suggestion Result:")
    print(suggestion_result)

    # Example 2: Debugging
    faulty_code = """
def add_numbers(a, b):
    return a + c  # 'c' is undefined
"""
    debug_result = collaborator.debug_code(faulty_code)
    print("\nDebugging Result:")
    print(debug_result)

    # Example 3: Code Review
    code_to_review = """
def calculate_area(radius):
    return 3.14 * radius * radius
"""
    review_result = collaborator.review_code(code_to_review)
    print("\nCode Review Result:")
    print(review_result)

    # Example 4: Intelligent Assist - Suggest Code
    user_query_suggest = "Can you suggest code to implement a user login system?"
    assist_result_suggest = collaborator.intelligent_assist(user_query_suggest, partial_code)
    print("\nIntelligent Assist Result - Suggest Code:")
    print(assist_result_suggest)

    # Example 5: Intelligent Assist - Debug Code
    user_query_debug = "I need help debugging my function."
    assist_result_debug = collaborator.intelligent_assist(user_query_debug, faulty_code)
    print("\nIntelligent Assist Result - Debug Code:")
    print(assist_result_debug)

    # Example 6: Intelligent Assist - Review Code
    user_query_review = "Please review my code for calculating area."
    assist_result_review = collaborator.intelligent_assist(user_query_review, code_to_review)
    print("\nIntelligent Assist Result - Review Code:")
    print(assist_result_review)
