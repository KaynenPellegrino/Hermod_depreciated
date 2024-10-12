import os
import shutil
from typing import Dict, Any
from abc import ABC, abstractmethod

from src.modules.code_generation.template_manager import TemplateManagerInterface

import logging

# Configure logging
logging.basicConfig(
    filename='hermod_ios_template.log',
    level=logging.DEBUG,
    format='%(asctime)s:%(levelname)s:%(message)s'
)


class IOSTemplateGeneratorInterface(ABC):
    """
    Interface for iOS App Template Generator.
    """
    @abstractmethod
    def generate_ios_app(self, project_id: str, project_info: Dict[str, Any]) -> None:
        pass


class IOSTemplateGenerator(IOSTemplateGeneratorInterface):
    """
    Generates iOS applications based on templates.
    """
    def __init__(self, template_manager: TemplateManagerInterface, base_directory: str = 'generated_projects'):
        """
        Initializes the IOSTemplateGenerator with necessary dependencies.

        :param template_manager: Instance of TemplateManagerInterface
        :param base_directory: Base directory where projects are stored
        """
        self.template_manager = template_manager
        self.base_directory = base_directory
        logging.info("IOSTemplateGenerator initialized.")

    def generate_ios_app(self, project_id: str, project_info: Dict[str, Any]) -> None:
        """
        Generates an iOS application based on the provided project information.

        :param project_id: Unique identifier for the project
        :param project_info: Dictionary containing project details
        """
        logging.info(f"Generating iOS app for project_id='{project_id}'.")
        try:
            project_path = os.path.join(self.base_directory, project_id, 'iOSApp')
            if os.path.exists(project_path):
                logging.warning(f"Project path '{project_path}' already exists. Overwriting.")
                shutil.rmtree(project_path)
            os.makedirs(project_path)
            logging.debug(f"Created project directory at '{project_path}'.")

            # Load iOS templates
            template_files = self.template_manager.list_templates()
            ios_templates = [tpl for tpl in template_files if tpl.endswith('.xcodeproj') or tpl.endswith('.swift') or tpl.endswith('.storyboard')]

            for template_name in ios_templates:
                template_content = self.template_manager.customize_template(template_name, project_info)
                # Determine the destination file path
                relative_path = os.path.relpath(os.path.join(self.template_manager.templates_dir, template_name), self.template_manager.templates_dir)
                dest_path = os.path.join(project_path, relative_path)
                dest_dir = os.path.dirname(dest_path)
                os.makedirs(dest_dir, exist_ok=True)
                with open(dest_path, 'w') as f:
                    f.write(template_content)
                logging.debug(f"Created file '{dest_path}' from template '{template_name}'.")

            logging.info(f"iOS app generated successfully at '{project_path}'.")
        except Exception as e:
            logging.error(f"Failed to generate iOS app for project_id='{project_id}': {e}")
            raise e


# Example usage and test case
if __name__ == "__main__":
    from src.modules.code_generation.template_manager import MockTemplateManager

    # Initialize mock TemplateManager
    mock_template_manager = MockTemplateManager()

    # Example iOS templates
    ios_project_template = """// iOS Project Template
    // Project Name: {{ project_name }}
    // Description: {{ project_description }}

    // Add your project setup here
    """
    ios_view_controller_template = """// {{ class_name }}.swift

    import UIKit

    class {{ class_name }}: UIViewController {

        override func viewDidLoad() {
            super.viewDidLoad()
            // Do any additional setup after loading the view.
        }

    }
    """

    # Save example templates to MockTemplateManager
    mock_template_manager.save_template("MyApp.xcodeproj", ios_project_template)
    mock_template_manager.save_template("MainViewController.swift", ios_view_controller_template)

    # Initialize IOSTemplateGenerator
    ios_generator = IOSTemplateGenerator(mock_template_manager)

    # Define project information
    project_info = {
        "project_name": "MyiOSApp",
        "project_description": "An iOS application developed using Hermod templates.",
        "class_name": "MainViewController"
    }

    # Generate iOS app
    project_id = "proj_ios_001"
    try:
        ios_generator.generate_ios_app(project_id, project_info)
        print(f"iOS app '{project_id}' generated successfully.")
    except Exception as e:
        print(f"Failed to generate iOS app: {e}")
