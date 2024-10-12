import os
import shutil
from typing import Dict, Any
from abc import ABC, abstractmethod

from src.modules.code_generation.template_manager import TemplateManagerInterface

import logging

# Configure logging
logging.basicConfig(
    filename='hermod_unreal_template.log',
    level=logging.DEBUG,
    format='%(asctime)s:%(levelname)s:%(message)s'
)


class UnrealTemplateGeneratorInterface(ABC):
    """
    Interface for Unreal Engine Project Template Generator.
    """

    @abstractmethod
    def generate_unreal_project(self, project_id: str, project_info: Dict[str, Any]) -> None:
        pass


class UnrealTemplateGenerator(UnrealTemplateGeneratorInterface):
    """
    Generates Unreal Engine game projects based on templates.
    """

    def __init__(self, template_manager: TemplateManagerInterface, base_directory: str = 'generated_projects'):
        """
        Initializes the UnrealTemplateGenerator with necessary dependencies.

        :param template_manager: Instance of TemplateManagerInterface
        :param base_directory: Base directory where projects are stored
        """
        self.template_manager = template_manager
        self.base_directory = base_directory
        logging.info("UnrealTemplateGenerator initialized.")

    def generate_unreal_project(self, project_id: str, project_info: Dict[str, Any]) -> None:
        """
        Generates an Unreal Engine project based on the provided project information.

        :param project_id: Unique identifier for the project
        :param project_info: Dictionary containing project details
        """
        logging.info(f"Generating Unreal Engine project for project_id='{project_id}'.")
        try:
            project_path = os.path.join(self.base_directory, project_id, 'UnrealProject')
            if os.path.exists(project_path):
                logging.warning(f"Project path '{project_path}' already exists. Overwriting.")
                shutil.rmtree(project_path)
            os.makedirs(project_path)
            logging.debug(f"Created project directory at '{project_path}'.")

            # Load Unreal templates
            template_files = self.template_manager.list_templates()
            unreal_templates = [tpl for tpl in template_files if
                                tpl.endswith('.uproject') or tpl.endswith('.cpp') or tpl.endswith('.h')]

            for template_name in unreal_templates:
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

            logging.info(f"Unreal Engine project generated successfully at '{project_path}'.")
        except Exception as e:
            logging.error(f"Failed to generate Unreal Engine project for project_id='{project_id}': {e}")
            raise e


# Example usage and test case
if __name__ == "__main__":
    from src.modules.code_generation.template_manager import MockTemplateManager

    # Initialize mock TemplateManager
    mock_template_manager = MockTemplateManager()

    # Example Unreal templates
    unreal_project_template = """{
        "FileVersion": 3,
        "EngineAssociation": "4.26",
        "Category": "",
        "Description": "{{ project_description }}",
        "Modules": [
            {
                "Name": "{{ project_name }}",
                "Type": "Runtime",
                "LoadingPhase": "Default"
            }
        ]
    }
    """
    unreal_cpp_template = """// {{ class_name }}.cpp

    #include "{{ class_name }}.h"

    {{ class_name }}::{{ class_name }}()
    {
        // Constructor implementation
    }

    void {{ class_name }}::BeginPlay()
    {
        Super::BeginPlay();
        // BeginPlay implementation
    }

    void {{ class_name }}::Tick(float DeltaTime)
    {
        Super::Tick(DeltaTime);
        // Tick implementation
    }
    """

    unreal_header_template = """// {{ class_name }}.h

    #pragma once

    #include "CoreMinimal.h"
    #include "GameFramework/Actor.h"
    #include "{{ project_name }}.generated.h"

    UCLASS()
    class {{ project_name }}_API A{{ class_name }} : public AActor
    {
        GENERATED_BODY()

    public:    
        // Sets default values for this actor's properties
        A{{ class_name }}();

    protected:
        // Called when the game starts or when spawned
        virtual void BeginPlay() override;

    public:    
        // Called every frame
        virtual void Tick(float DeltaTime) override;
    };
    """

    # Save example templates to MockTemplateManager
    mock_template_manager.save_template("MyGame.uproject", unreal_project_template)
    mock_template_manager.save_template("MyClass.cpp", unreal_cpp_template)
    mock_template_manager.save_template("MyClass.h", unreal_header_template)

    # Initialize UnrealTemplateGenerator
    unreal_generator = UnrealTemplateGenerator(mock_template_manager)

    # Define project information
    project_info = {
        "project_name": "MyGame",
        "project_description": "An Unreal Engine game developed using Hermod templates.",
        "class_name": "PlayerCharacter"
    }

    # Generate Unreal project
    project_id = "proj_unreal_001"
    try:
        unreal_generator.generate_unreal_project(project_id, project_info)
        print(f"Unreal Engine project '{project_id}' generated successfully.")
    except Exception as e:
        print(f"Failed to generate Unreal Engine project: {e}")
