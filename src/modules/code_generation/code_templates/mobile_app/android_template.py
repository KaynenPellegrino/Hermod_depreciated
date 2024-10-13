import os
import shutil
from typing import Dict, Any
from abc import ABC, abstractmethod

from src.modules.code_generation.template_manager import TemplateManagerInterface

import logging

# Configure logging
logging.basicConfig(
    filename='hermod_android_template.log',
    level=logging.DEBUG,
    format='%(asctime)s:%(levelname)s:%(message)s'
)


class AndroidTemplateGeneratorInterface(ABC):
    """
    Interface for Android App Template Generator.
    """

    @abstractmethod
    def generate_android_app(self, project_id: str, project_info: Dict[str, Any]) -> None:
        pass


class AndroidTemplateGenerator(AndroidTemplateGeneratorInterface):
    """
    Generates Android applications based on templates.
    """

    def __init__(self, template_manager: TemplateManagerInterface, base_directory: str = 'generated_projects'):
        """
        Initializes the AndroidTemplateGenerator with necessary dependencies.

        :param template_manager: Instance of TemplateManagerInterface
        :param base_directory: Base directory where projects are stored
        """
        self.template_manager = template_manager
        self.base_directory = base_directory
        logging.info("AndroidTemplateGenerator initialized.")

    def generate_android_app(self, project_id: str, project_info: Dict[str, Any]) -> None:
        """
        Generates an Android application based on the provided project information.

        :param project_id: Unique identifier for the project
        :param project_info: Dictionary containing project details
        """
        logging.info(f"Generating Android app for project_id='{project_id}'.")
        try:
            project_path = os.path.join(self.base_directory, project_id, 'AndroidApp')
            if os.path.exists(project_path):
                logging.warning(f"Project path '{project_path}' already exists. Overwriting.")
                shutil.rmtree(project_path)
            os.makedirs(project_path)
            logging.debug(f"Created project directory at '{project_path}'.")

            # Load Android templates
            template_files = self.template_manager.list_templates()
            android_templates = [tpl for tpl in template_files if
                                 tpl.endswith('.xml') or tpl.endswith('.java') or tpl.endswith('.kt')]

            for template_name in android_templates:
                template_content = self.template_manager.customize_template(template_name, project_info)
                # Determine the destination file path
                relative_path = os.path.relpath(os.path.join(self.template_manager.templates_dir, template_name),
                                                self.template_manager.templates_dir)
                dest_path = os.path.join(project_path, relative_path)
                dest_dir = os.path.dirname(dest_path)
                os.makedirs(dest_dir, exist_ok=True)
                with open(dest_path, 'w') as f:
                    f.write(template_content)
                logging.debug(f"Created file '{dest_path}' from template '{template_name}'.")

            logging.info(f"Android app generated successfully at '{project_path}'.")
        except Exception as e:
            logging.error(f"Failed to generate Android app for project_id='{project_id}': {e}")
            raise e


# Example usage and test case
if __name__ == "__main__":
    from src.modules.code_generation.template_manager import MockTemplateManager

    # Initialize mock TemplateManager
    mock_template_manager = MockTemplateManager()

    # Example Android templates
    android_layout_template = """<!-- Android Layout Template -->
    <LinearLayout xmlns:android="http://schemas.android.com/apk/res/android"
        android:orientation="vertical"
        android:layout_width="match_parent"
        android:layout_height="match_parent">

        <TextView
            android:id="@+id/textView"
            android:text="{{ welcome_text }}"
            android:layout_width="wrap_content"
            android:layout_height="wrap_content" />

    </LinearLayout>
    """
    android_main_activity_template = """// MainActivity.java

    package com.example.{{ package_name }};

    import android.os.Bundle;
    import androidx.appcompat.app.AppCompatActivity;

    public class MainActivity extends AppCompatActivity {

        @Override
        protected void onCreate(Bundle savedInstanceState) {
            super.onCreate(savedInstanceState);
            setContentView(R.layout.activity_main);
        }
    }
    """

    # Save example templates to MockTemplateManager
    mock_template_manager.save_template("activity_main.xml", android_layout_template)
    mock_template_manager.save_template("MainActivity.java", android_main_activity_template)

    # Initialize AndroidTemplateGenerator
    android_generator = AndroidTemplateGenerator(mock_template_manager)

    # Define project information
    project_info = {
        "project_name": "MyAndroidApp",
        "project_description": "An Android application developed using Hermod templates.",
        "welcome_text": "Welcome to MyAndroidApp!",
        "package_name": "myandroidapp"
    }

    # Generate Android app
    project_id = "proj_android_001"
    try:
        android_generator.generate_android_app(project_id, project_info)
        print(f"Android app '{project_id}' generated successfully.")
    except Exception as e:
        print(f"Failed to generate Android app: {e}")
