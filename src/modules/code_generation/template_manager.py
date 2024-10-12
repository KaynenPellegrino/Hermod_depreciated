import logging
import os
from typing import List, Dict, Any
from abc import ABC, abstractmethod
from jinja2 import Environment, FileSystemLoader, select_autoescape

# Configure logging
logging.basicConfig(
    filename='hermod_template_manager.log',
    level=logging.DEBUG,
    format='%(asctime)s:%(levelname)s:%(message)s'
)


# Interface for Template Management
class TemplateManagerInterface(ABC):
    """
    Interface for the TemplateManager module.
    Defines methods for loading, saving, updating, deleting, and listing templates.
    """

    @abstractmethod
    def load_template(self, template_name: str) -> str:
        pass

    @abstractmethod
    def save_template(self, template_name: str, content: str) -> None:
        pass

    @abstractmethod
    def update_template(self, template_name: str, content: str) -> None:
        pass

    @abstractmethod
    def delete_template(self, template_name: str) -> None:
        pass

    @abstractmethod
    def list_templates(self) -> List[str]:
        pass

    @abstractmethod
    def customize_template(self, template_name: str, context: Dict[str, Any]) -> str:
        pass


# Concrete Implementation of TemplateManagerInterface
class TemplateManager(TemplateManagerInterface):
    """
    Manages templates stored in a specified directory using Jinja2 for customization.
    """

    def __init__(self, templates_dir: str = 'templates'):
        """
        Initializes the TemplateManager with the specified templates directory.
        Ensures the directory exists and sets up Jinja2 environment for templating.
        """
        self.templates_dir = templates_dir
        os.makedirs(self.templates_dir, exist_ok=True)
        logging.info(f"TemplateManager initialized with templates directory at '{self.templates_dir}'.")

        # Setup Jinja2 environment for template rendering
        self.env = Environment(
            loader=FileSystemLoader(self.templates_dir),
            autoescape=select_autoescape(['html', 'xml', 'md'])
        )

    def load_template(self, template_name: str) -> str:
        """
        Loads the content of the specified template file.
        """
        try:
            template_path = os.path.join(self.templates_dir, template_name)
            with open(template_path, 'r') as file:
                content = file.read()
            logging.debug(f"Loaded template '{template_name}'.")
            return content
        except FileNotFoundError:
            logging.error(f"Template '{template_name}' not found in '{self.templates_dir}'.")
            raise
        except Exception as e:
            logging.error(f"Error loading template '{template_name}': {e}")
            raise

    def save_template(self, template_name: str, content: str) -> None:
        """
        Saves a new template with the provided content.
        """
        try:
            template_path = os.path.join(self.templates_dir, template_name)
            with open(template_path, 'w') as file:
                file.write(content)
            logging.debug(f"Saved new template '{template_name}'.")
        except Exception as e:
            logging.error(f"Error saving template '{template_name}': {e}")
            raise

    def update_template(self, template_name: str, content: str) -> None:
        """
        Updates an existing template with new content.
        """
        try:
            template_path = os.path.join(self.templates_dir, template_name)
            if not os.path.exists(template_path):
                logging.error(f"Template '{template_name}' does not exist and cannot be updated.")
                raise FileNotFoundError(f"Template '{template_name}' does not exist.")
            with open(template_path, 'w') as file:
                file.write(content)
            logging.debug(f"Updated template '{template_name}'.")
        except Exception as e:
            logging.error(f"Error updating template '{template_name}': {e}")
            raise

    def delete_template(self, template_name: str) -> None:
        """
        Deletes the specified template.
        """
        try:
            template_path = os.path.join(self.templates_dir, template_name)
            if os.path.exists(template_path):
                os.remove(template_path)
                logging.debug(f"Deleted template '{template_name}'.")
            else:
                logging.warning(f"Attempted to delete non-existent template '{template_name}'.")
                raise FileNotFoundError(f"Template '{template_name}' does not exist.")
        except Exception as e:
            logging.error(f"Error deleting template '{template_name}': {e}")
            raise

    def list_templates(self) -> List[str]:
        """
        Lists all available templates in the templates directory.
        """
        try:
            templates = [f for f in os.listdir(self.templates_dir) if
                         os.path.isfile(os.path.join(self.templates_dir, f))]
            logging.debug(f"Listed templates: {templates}")
            return templates
        except Exception as e:
            logging.error(f"Error listing templates in '{self.templates_dir}': {e}")
            raise

    def customize_template(self, template_name: str, context: Dict[str, Any]) -> str:
        """
        Customizes a template by replacing placeholders with actual values based on the provided context.
        Uses Jinja2 to render the template with the given context.
        """
        try:
            template = self.env.get_template(template_name)
            customized_content = template.render(context)
            logging.debug(f"Customized template '{template_name}' with context: {context}")
            return customized_content
        except Exception as e:
            logging.error(f"Error customizing template '{template_name}': {e}")
            raise


# Example usage of TemplateManager
if __name__ == "__main__":
    # Initialize TemplateManager
    template_manager = TemplateManager(templates_dir='templates')

    # Example template content
    readme_template = """# {{ project_name }}

    {{ project_description }}

    ## Features
    {% for feature in features %}
    - {{ feature }}
    {% endfor %}

    ## Installation
    ```bash
    pip install -r requirements.txt
    ```

    ## Usage
    ```bash
    python {{ main_file }}
    ```

    ## Contributing
    Contributions are welcome! Please open an issue or submit a pull request.

    ## License
    MIT License
    """

    # Save the README template
    template_manager.save_template("README.md", readme_template)

    # List available templates
    print("Available Templates:")
    templates = template_manager.list_templates()
    for tmpl in templates:
        print(f"- {tmpl}")

    # Customize README template
    readme_context = {
        "project_name": "AI Chatbot",
        "project_description": "A chatbot using NLP techniques.",
        "features": ["Authentication", "Database Integration"],
        "main_file": "app.py"
    }
    customized_readme = template_manager.customize_template("README.md", readme_context)
    print("\n=== Customized README.md ===\n")
    print(customized_readme)
