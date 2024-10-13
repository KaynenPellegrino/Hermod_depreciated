import os
import shutil
from typing import Dict, Any
from abc import ABC, abstractmethod

from src.modules.code_generation.template_manager import TemplateManagerInterface

import logging

# Configure logging
logging.basicConfig(
    filename='hermod_unity_template.log',
    level=logging.DEBUG,
    format='%(asctime)s:%(levelname)s:%(message)s'
)


class UnityTemplateGeneratorInterface(ABC):
    """
    Interface for Unity Project Template Generator.
    """
    @abstractmethod
    def generate_unity_project(self, project_id: str, project_info: Dict[str, Any]) -> None:
        pass


class UnityTemplateGenerator(UnityTemplateGeneratorInterface):
    """
    Generates Unity game projects based on templates.
    """
    def __init__(self, template_manager: TemplateManagerInterface, base_directory: str = 'generated_projects'):
        """
        Initializes the UnityTemplateGenerator with necessary dependencies.

        :param template_manager: Instance of TemplateManagerInterface
        :param base_directory: Base directory where projects are stored
        """
        self.template_manager = template_manager
        self.base_directory = base_directory
        logging.info("UnityTemplateGenerator initialized.")

    def generate_unity_project(self, project_id: str, project_info: Dict[str, Any]) -> None:
        """
        Generates a Unity project based on the provided project information.

        :param project_id: Unique identifier for the project
        :param project_info: Dictionary containing project details
        """
        logging.info(f"Generating Unity project for project_id='{project_id}'.")
        try:
            project_path = os.path.join(self.base_directory, project_id, 'UnityProject')
            if os.path.exists(project_path):
                logging.warning(f"Project path '{project_path}' already exists. Overwriting.")
                shutil.rmtree(project_path)
            os.makedirs(project_path)
            logging.debug(f"Created project directory at '{project_path}'.")

            # Load Unity template files
            template_files = self.template_manager.list_templates()
            unity_templates = [tpl for tpl in template_files if tpl.endswith('.unity') or tpl.endswith('.cs')]

            for template_name in unity_templates:
                template_content = self.template_manager.customize_template(template_name, project_info)
                # Determine the destination file path
                relative_path = os.path.relpath(os.path.join(self.template_manager.templates_dir, template_name), self.template_manager.templates_dir)
                dest_path = os.path.join(project_path, relative_path)
                dest_dir = os.path.dirname(dest_path)
                os.makedirs(dest_dir, exist_ok=True)
                with open(dest_path, 'w') as f:
                    f.write(template_content)
                logging.debug(f"Created file '{dest_path}' from template '{template_name}'.")

            logging.info(f"Unity project generated successfully at '{project_path}'.")
        except Exception as e:
            logging.error(f"Failed to generate Unity project for project_id='{project_id}': {e}")
            raise e

# Example usage and test case
if __name__ == "__main__":
    from src.modules.code_generation.template_manager import MockTemplateManager

    # Initialize mock TemplateManager
    mock_template_manager = MockTemplateManager()

    # Example Unity templates
    unity_scene_template = """# Unity Scene Template
    // Project Name: {{ project_name }}
    // Description: {{ project_description }}

    // Add your scene setup here
    """
    unity_script_template = """// Unity Script Template
    using UnityEngine;

    public class {{ class_name }} : MonoBehaviour
    {
        // Start is called before the first frame update
        void Start()
        {
            // Initialization code
        }

        // Update is called once per frame
        void Update()
        {
            // Frame update code
        }
    }
    """

    # Save example templates to MockTemplateManager
    mock_template_manager.save_template("MainScene.unity", unity_scene_template)
    mock_template_manager.save_template("PlayerController.cs", unity_script_template)

    # Initialize UnityTemplateGenerator
    unity_generator = UnityTemplateGenerator(mock_template_manager)

    # Define project information
    project_info = {
        "project_name": "AI Unity Game",
        "project_description": "A Unity game developed using Hermod templates.",
        "class_name": "PlayerController"
    }

    # Generate Unity project
    project_id = "proj_unity_001"
    try:
        unity_generator.generate_unity_project(project_id, project_info)
        print(f"Unity project '{project_id}' generated successfully.")
    except Exception as e:
        print(f"Failed to generate Unity project: {e}")
