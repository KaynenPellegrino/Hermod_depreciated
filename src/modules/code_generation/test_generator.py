import logging
import os
import ast
from typing import Dict, Any
from abc import ABC, abstractmethod
from jinja2 import Template, Environment, FileSystemLoader, select_autoescape

# Configure logging
logging.basicConfig(
    filename='hermod_test_generator.log',
    level=logging.DEBUG,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# Interfaces for dependencies
class ProjectManagerInterface(ABC):
    """
    Interface for the ProjectManager module.
    This should be replaced with the actual implementation from project_manager.py
    """

    @abstractmethod
    def get_project_metadata(self, project_id: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_project_path(self, project_id: str) -> str:
        pass


class TemplateManagerInterface(ABC):
    """
    Interface for the TemplateManager module.
    This should be replaced with the actual implementation from template_manager.py
    """

    @abstractmethod
    def get_template(self, template_name: str) -> str:
        pass


# Concrete Implementations
class TestGenerator:
    def __init__(self,
                 project_manager: ProjectManagerInterface,
                 template_manager: TemplateManagerInterface,
                 templates_dir: str = 'test_templates'):
        """
        Initializes the TestGenerator with necessary dependencies.

        :param project_manager: Instance of ProjectManagerInterface
        :param template_manager: Instance of TemplateManagerInterface
        :param templates_dir: Directory where test templates are stored
        """
        self.project_manager = project_manager
        self.template_manager = template_manager
        self.templates_dir = templates_dir
        os.makedirs(self.templates_dir, exist_ok=True)
        logging.info(f"TestGenerator initialized with templates directory at '{self.templates_dir}'.")

        # Initialize Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(self.templates_dir),
            autoescape=select_autoescape(['py'])
        )

    def generate_tests(self, project_id: str) -> None:
        """
        Initiates the test generation process for a specified project.

        :param project_id: Unique identifier for the project
        """
        logging.info(f"Starting test generation for project_id='{project_id}'.")
        try:
            project_path = self.project_manager.get_project_path(project_id)
            analysis_result = self.analyze_code(project_path)
            tests = self.create_test_files(project_id, analysis_result)
            self.save_tests(project_id, tests)
            logging.info(f"Test generation completed successfully for project_id='{project_id}'.")
        except Exception as e:
            logging.error(f"Failed to generate tests for project_id='{project_id}': {e}")
            raise e

    def analyze_code(self, project_path: str) -> Dict[str, Any]:
        """
        Analyzes the project's codebase to identify functions and classes that require testing.

        :param project_path: Path to the project's directory
        :return: Dictionary containing analysis results
        """
        logging.info(f"Analyzing code at '{project_path}' for test generation.")
        analysis = {
            "modules": []
        }
        try:
            src_path = os.path.join(project_path, 'src')
            for root, _, files in os.walk(src_path):
                for file in files:
                    if file.endswith('.py'):
                        file_path = os.path.join(root, file)
                        with open(file_path, 'r') as f:
                            node = ast.parse(f.read(), filename=file_path)
                        module_info = {
                            "module_name": os.path.splitext(file)[0],
                            "classes": []
                        }
                        for obj in node.body:
                            if isinstance(obj, ast.ClassDef):
                                class_info = {
                                    "class_name": obj.name,
                                    "methods": [method.name for method in obj.body if
                                                isinstance(method, ast.FunctionDef)]
                                }
                                module_info["classes"].append(class_info)
                        if module_info["classes"]:
                            analysis["modules"].append(module_info)
            logging.debug(f"Code analysis result: {analysis}")
            return analysis
        except Exception as e:
            logging.error(f"Error during code analysis at '{project_path}': {e}")
            raise e

    def create_test_files(self, project_id: str, analysis_result: Dict[str, Any]) -> Dict[str, str]:
        """
        Generates test files based on the analysis results and templates.

        :param project_id: Unique identifier for the project
        :param analysis_result: Dictionary containing analysis results
        :return: Dictionary with test file paths as keys and their content as values
        """
        logging.info(f"Creating test files for project_id='{project_id}'.")
        tests = {}
        try:
            for module in analysis_result.get("modules", []):
                module_name = module["module_name"]
                for cls in module["classes"]:
                    class_name = cls["class_name"]
                    methods = cls["methods"]
                    template_content = self.template_manager.get_template("test_template.py")
                    template = Template(template_content)
                    context = {
                        "module_name": module_name,
                        "class_name": class_name,
                        "methods": methods
                    }
                    test_content = template.render(context)
                    test_file_name = f"test_{module_name}.py"
                    tests[test_file_name] = test_content
                    logging.debug(f"Generated test file '{test_file_name}'.")
            return tests
        except Exception as e:
            logging.error(f"Error creating test files for project_id='{project_id}': {e}")
            raise e

    def save_tests(self, project_id: str, tests: Dict[str, str]) -> None:
        """
        Saves the generated test files to the project's testing directory.

        :param project_id: Unique identifier for the project
        :param tests: Dictionary with test file names and their content
        """
        logging.info(f"Saving test files for project_id='{project_id}'.")
        try:
            tests_dir = os.path.join(self.project_manager.get_project_path(project_id), 'tests')
            os.makedirs(tests_dir, exist_ok=True)
            for file_name, content in tests.items():
                file_path = os.path.join(tests_dir, file_name)
                with open(file_path, 'w') as f:
                    f.write(content)
                logging.debug(f"Saved test file '{file_path}'.")
            logging.info(f"All test files saved successfully for project_id='{project_id}'.")
        except Exception as e:
            logging.error(f"Error saving test files for project_id='{project_id}': {e}")
            raise e


# Example usage and test cases
if __name__ == "__main__":
    # Replace mock dependencies with actual implementations
    from src.modules.project_management.project_manager import ProjectManager
    from src.modules.template_management.template_manager import TemplateManager

    # Initialize real dependencies
    project_manager = ProjectManager()
    template_manager = TemplateManager(templates_dir='test_templates')

    # Initialize TestGenerator
    test_generator = TestGenerator(project_manager, template_manager)

    # Define project ID
    project_id = "proj_12345"

    # Generate tests for the project
    try:
        test_generator.generate_tests(project_id)
        print(f"Tests generated successfully for project '{project_id}'.")
    except Exception as e:
        print(f"Failed to generate tests: {e}")

    # List generated test files
    tests_path = os.path.join(project_manager.get_project_path(project_id), 'tests')
    if os.path.exists(tests_path):
        print("\nGenerated Test Files:")
        for test_file in os.listdir(tests_path):
            print(f"- {test_file}")
    else:
        print("No tests directory found.")
