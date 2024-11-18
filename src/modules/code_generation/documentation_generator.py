import ast
import logging
import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any

from jinja2 import Template

from staging import ProjectManager, TemplateManager

# Configure logging
logging.basicConfig(
    filename='hermod_documentation_generator.log',
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
    def get_project_structure(self, project_id: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_code_files(self, project_id: str) -> List[str]:
        pass


class TemplateManagerInterface(ABC):
    """
    Interface for the TemplateManager module.
    This should be replaced with the actual implementation from template_manager.py
    """

    @abstractmethod
    def get_template(self, template_name: str) -> str:
        pass


# Documentation Generator Class
class DocumentationGenerator:
    def __init__(self, project_manager: ProjectManagerInterface, template_manager: TemplateManagerInterface):
        """
        Initializes the DocumentationGenerator with necessary dependencies.

        :param project_manager: Instance of ProjectManagerInterface
        :param template_manager: Instance of TemplateManagerInterface
        """
        self.project_manager = project_manager
        self.template_manager = template_manager
        logging.info("DocumentationGenerator initialized.")

    def generate_readme(self, project_id: str) -> str:
        """
        Generates a README.md file for the project.

        :param project_id: Unique identifier for the project
        :return: README content as a string
        """
        logging.info(f"Generating README for project_id='{project_id}'.")
        try:
            metadata = self.project_manager.get_project_metadata(project_id)
            project_name = metadata.get("name", "Project")
            project_description = metadata.get("description", "No description provided.")
            features = "\n".join(
                [f"- {feature.replace('_', ' ').capitalize()}" for feature in metadata.get("features", [])])
            main_file = self.get_main_file(metadata)

            template_content = self.template_manager.get_template("README.md")
            template = Template(template_content)
            readme_content = template.render(
                project_name=project_name,
                project_description=project_description,
                features=features,
                main_file=main_file
            )
            logging.debug("README generated successfully.")
            return readme_content
        except Exception as e:
            logging.error(f"Error generating README for project_id='{project_id}': {e}")
            return ""

    def generate_api_docs(self, project_id: str) -> str:
        """
        Generates API documentation by parsing code files for docstrings.

        :param project_id: Unique identifier for the project
        :return: API documentation content as a string
        """
        logging.info(f"Generating API documentation for project_id='{project_id}'.")
        try:
            code_files = self.project_manager.get_code_files(project_id)
            api_details = ""
            for file_path in code_files:
                if file_path.endswith(".py"):
                    api_details += self.parse_python_file(file_path) + "\n"
                # Add support for other languages as needed

            template_content = self.template_manager.get_template("API_DOC.md")
            template = Template(template_content)
            metadata = self.project_manager.get_project_metadata(project_id)
            project_name = metadata.get("name", "Project")
            api_docs_content = template.render(
                project_name=project_name,
                endpoints=api_details
            )
            logging.debug("API documentation generated successfully.")
            return api_docs_content
        except Exception as e:
            logging.error(f"Error generating API documentation for project_id='{project_id}': {e}")
            return ""

    def generate_user_guide(self, project_id: str) -> str:
        """
        Generates a user guide for the project.

        :param project_id: Unique identifier for the project
        :return: User guide content as a string
        """
        logging.info(f"Generating user guide for project_id='{project_id}'.")
        try:
            metadata = self.project_manager.get_project_metadata(project_id)
            project_name = metadata.get("name", "Project")
            features = metadata.get("features", [])
            if len(features) < 2:
                features += ["additional_feature"] * (2 - len(features))  # Ensure at least two features

            feature_1 = features[0].replace('_', ' ').capitalize()
            feature_2 = features[1].replace('_', ' ').capitalize()

            template_content = self.template_manager.get_template("USER_GUIDE.md")
            template = Template(template_content)
            user_guide_content = template.render(
                project_name=project_name,
                feature_1=feature_1,
                feature_2=feature_2
            )
            logging.debug("User guide generated successfully.")
            return user_guide_content
        except Exception as e:
            logging.error(f"Error generating user guide for project_id='{project_id}': {e}")
            return ""

    def parse_python_file(self, file_path: str) -> str:
        """
        Parses a Python file to extract API endpoint information from docstrings.

        :param file_path: Path to the Python file
        :return: Formatted API endpoint information as a string
        """
        logging.debug(f"Parsing Python file '{file_path}' for API documentation.")
        api_info = ""
        try:
            with open(file_path, 'r') as file:
                node = ast.parse(file.read(), filename=file_path)

            for obj in node.body:
                if isinstance(obj, ast.FunctionDef):
                    docstring = ast.get_docstring(obj)
                    if docstring:
                        api_info += f"### {obj.name}\n\n{docstring}\n\n"
            return api_info
        except Exception as e:
            logging.error(f"Error parsing Python file '{file_path}': {e}")
            return ""


# Example Usage
if __name__ == "__main__":
    # Initialize real dependencies
    project_manager = ProjectManager()  # Use real ProjectManager
    template_manager = TemplateManager()  # Use real TemplateManager

    # Initialize DocumentationGenerator
    doc_generator = DocumentationGenerator(project_manager, template_manager)

    # Define project ID
    project_id = "proj_12345"

    # Generate all documentation
    documentation = {
        "README.md": doc_generator.generate_readme(project_id),
        "API_DOC.md": doc_generator.generate_api_docs(project_id),
        "USER_GUIDE.md": doc_generator.generate_user_guide(project_id)
    }

    # Save documentation to file system
    base_path = os.path.join('generated_projects')
