import os
import shutil
from typing import Dict, Any
from abc import ABC, abstractmethod

from src.modules.code_generation.template_manager import TemplateManagerInterface

import logging

# Configure logging
logging.basicConfig(
    filename='hermod_flask_template.log',
    level=logging.DEBUG,
    format='%(asctime)s:%(levelname)s:%(message)s'
)


class FlaskTemplateGeneratorInterface(ABC):
    """
    Interface for Flask App Template Generator.
    """
    @abstractmethod
    def generate_flask_app(self, project_id: str, project_info: Dict[str, Any]) -> None:
        pass


class FlaskTemplateGenerator(FlaskTemplateGeneratorInterface):
    """
    Generates Flask web applications based on templates.
    """
    def __init__(self, template_manager: TemplateManagerInterface, base_directory: str = 'generated_projects'):
        """
        Initializes the FlaskTemplateGenerator with necessary dependencies.

        :param template_manager: Instance of TemplateManagerInterface
        :param base_directory: Base directory where projects are stored
        """
        self.template_manager = template_manager
        self.base_directory = base_directory
        logging.info("FlaskTemplateGenerator initialized.")

    def generate_flask_app(self, project_id: str, project_info: Dict[str, Any]) -> None:
        """
        Generates a Flask application based on the provided project information.

        :param project_id: Unique identifier for the project
        :param project_info: Dictionary containing project details
        """
        logging.info(f"Generating Flask app for project_id='{project_id}'.")
        try:
            project_path = os.path.join(self.base_directory, project_id, 'FlaskApp')
            if os.path.exists(project_path):
                logging.warning(f"Project path '{project_path}' already exists. Overwriting.")
                shutil.rmtree(project_path)
            os.makedirs(project_path)
            logging.debug(f"Created project directory at '{project_path}'.")

            # Load Flask templates
            template_files = self.template_manager.list_templates()
            flask_templates = [tpl for tpl in template_files if tpl.endswith('.py') or tpl.endswith('.html') or tpl.endswith('.css')]

            for template_name in flask_templates:
                template_content = self.template_manager.customize_template(template_name, project_info)
                # Determine the destination file path
                relative_path = os.path.relpath(os.path.join(self.template_manager.templates_dir, template_name), self.template_manager.templates_dir)
                dest_path = os.path.join(project_path, relative_path)
                dest_dir = os.path.dirname(dest_path)
                os.makedirs(dest_dir, exist_ok=True)
                with open(dest_path, 'w') as f:
                    f.write(template_content)
                logging.debug(f"Created file '{dest_path}' from template '{template_name}'.")

            logging.info(f"Flask app generated successfully at '{project_path}'.")
        except Exception as e:
            logging.error(f"Failed to generate Flask app for project_id='{project_id}': {e}")
            raise e


# Example usage and test case
if __name__ == "__main__":

    from src.modules.code_generation.template_manager import MockTemplateManager

    # Initialize mock TemplateManager
    mock_template_manager = MockTemplateManager()

    # Example Flask templates
    flask_app_py_template = """# {{ project_name }} Flask App

    from flask import Flask

    app = Flask(__name__)

    @app.route('/')
    def home():
        return "{{ welcome_message }}"

    if __name__ == '__main__':
        app.run(debug=True)
    """
    flask_index_html_template = """<!-- Flask Index HTML Template -->
    <!DOCTYPE html>
    <html>
    <head>
        <title>{{ project_name }}</title>
    </head>
    <body>
        <h1>{{ welcome_message }}</h1>
    </body>
    </html>
    """
    flask_style_css_template = """/* Flask Style CSS Template */
    body {
        font-family: Arial, sans-serif;
        background-color: #f8f9fa;
        text-align: center;
        margin-top: 50px;
    }
    h1 {
        color: #343a40;
    }
    """

    # Save example templates to MockTemplateManager
    mock_template_manager.save_template("app.py", flask_app_py_template)
    mock_template_manager.save_template("templates/index.html", flask_index_html_template)
    mock_template_manager.save_template("static/css/style.css", flask_style_css_template)

    # Initialize FlaskTemplateGenerator
    flask_generator = FlaskTemplateGenerator(mock_template_manager)

    # Define project information
    project_info = {
        "project_name": "MyFlaskApp",
        "project_description": "A Flask web application developed using Hermod templates.",
        "welcome_message": "Welcome to MyFlaskApp!"
    }

    # Generate Flask app
    project_id = "proj_flask_001"
    try:
        flask_generator.generate_flask_app(project_id, project_info)
        print(f"Flask app '{project_id}' generated successfully.")
    except Exception as e:
        print(f"Failed to generate Flask app: {e}")
