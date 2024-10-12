import logging
import os
import shutil
from typing import List, Dict, Any
from abc import ABC, abstractmethod
from template_manager import TemplateManager  # Import real TemplateManager
from project_manager import ProjectManager  # Import real ProjectManager
from language_models.code_gen_model import OpenAIModel, MockCodeGenModel  # Import CodeGen models
from language_models.syntax_checker import PythonSyntaxChecker, JavaScriptSyntaxChecker, JavaSyntaxChecker  # Import Syntax Checkers
from code_templates.web_app.django_template import DjangoTemplateGenerator
from code_templates.web_app.flask_template import FlaskTemplateGenerator
from code_templates.mobile_app.android_template import AndroidTemplateGenerator
from code_templates.mobile_app.ios_template import IOSTemplateGenerator
from code_templates.game_dev.unity_template import UnityTemplateGenerator
from code_templates.game_dev.unreal_template import UnrealTemplateGenerator

# Configure logging
logging.basicConfig(
    filename='hermod_code_generator.log',
    level=logging.DEBUG,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# Interfaces for dependencies
class AIModelInterface(ABC):
    """
    Interface for integrating AI models.
    This should be replaced with the actual implementation that interacts with an AI service.
    """

    @abstractmethod
    def generate_code(self, prompt: str, context: Dict[str, Any]) -> str:
        pass


class CodeGenerator:
    def __init__(self,
                 template_manager: TemplateManager,
                 ai_model: AIModelInterface,
                 project_manager: ProjectManager,
                 django_generator: DjangoTemplateGenerator,
                 flask_generator: FlaskTemplateGenerator,
                 android_generator: AndroidTemplateGenerator,
                 ios_generator: IOSTemplateGenerator,
                 unity_generator: UnityTemplateGenerator,
                 unreal_generator: UnrealTemplateGenerator):
        """
        Initializes the CodeGenerator with necessary dependencies.

        :param template_manager: Instance of TemplateManager
        :param ai_model: Instance of AIModelInterface
        :param project_manager: Instance of ProjectManager
        :param django_generator: DjangoTemplateGenerator
        :param flask_generator: FlaskTemplateGenerator
        :param android_generator: AndroidTemplateGenerator
        :param ios_generator: IOSTemplateGenerator
        :param unity_generator: UnityTemplateGenerator
        :param unreal_generator: UnrealTemplateGenerator
        """
        self.template_manager = template_manager
        self.ai_model = ai_model
        self.project_manager = project_manager
        self.django_generator = django_generator
        self.flask_generator = flask_generator
        self.android_generator = android_generator
        self.ios_generator = ios_generator
        self.unity_generator = unity_generator
        self.unreal_generator = unreal_generator
        self.python_syntax_checker = PythonSyntaxChecker()
        self.js_syntax_checker = JavaScriptSyntaxChecker()
        self.java_syntax_checker = JavaSyntaxChecker()
        logging.info("CodeGenerator initialized.")

    def generate_codebase(self, project_id: str, user_requirements: Dict[str, Any]) -> None:
        """
        Generates the entire codebase based on user requirements by invoking the appropriate template generator.

        :param project_id: Unique identifier for the project
        :param user_requirements: Structured user requirements
        """
        logging.info(f"Starting codebase generation for project_id='{project_id}' with requirements: {user_requirements}")
        project_type = user_requirements.get("project_type")

        if project_type == 'django':
            self.django_generator.generate_django_app(project_id, user_requirements)
        elif project_type == 'flask':
            self.flask_generator.generate_flask_app(project_id, user_requirements)
        elif project_type == 'android':
            self.android_generator.generate_android_app(project_id, user_requirements)
        elif project_type == 'ios':
            self.ios_generator.generate_ios_app(project_id, user_requirements)
        elif project_type == 'unity':
            self.unity_generator.generate_unity_project(project_id, user_requirements)
        elif project_type == 'unreal':
            self.unreal_generator.generate_unreal_project(project_id, user_requirements)
        else:
            logging.error(f"Unsupported project type: {project_type}")
            raise ValueError(f"Unsupported project type: {project_type}")

        logging.info(f"Codebase generated for project '{project_id}'.")

    def generate_file(self, feature: str, language: str, project_type: str) -> Dict[str, Any]:
        """
        Generates an individual file based on a feature.

        :param feature: The feature for which to generate the file
        :param language: Programming language
        :param project_type: Type of the project
        :return: Dictionary with filename and content
        """
        try:
            prompt = f"Generate a {feature} module for a {project_type} written in {language}."
            context = {
                "feature": feature,
                "project_type": project_type,
                "language": language
            }
            ai_generated_code = self.integrate_ai_assistance(prompt, context)
            filename = self.get_filename_for_feature(feature, language)
            logging.debug(f"Generated file '{filename}' for feature='{feature}'.")
            return {
                "filename": filename,
                "content": ai_generated_code
            }
        except Exception as e:
            logging.error(f"Error generating file for feature='{feature}': {e}")
            return {}

    def integrate_ai_assistance(self, prompt: str, context: Dict[str, Any]) -> str:
        """
        Enhances code snippets using AI models and performs syntax checking.

        :param prompt: The prompt to send to the AI model
        :param context: Additional context for the AI model
        :return: Generated code as a string
        """
        try:
            ai_code = self.ai_model.generate_code(prompt, context)
            language = context.get("language", "").lower()

            # Syntax checking based on the language
            if language == "python":
                syntax_result = self.python_syntax_checker.check_syntax(ai_code, language)
            elif language == "javascript":
                syntax_result = self.js_syntax_checker.check_syntax(ai_code, language)
            elif language == "java":
                syntax_result = self.java_syntax_checker.check_syntax(ai_code, language)
            else:
                logging.warning(f"No syntax checker available for language '{language}'.")
                syntax_result = {"success": True}

            if not syntax_result["success"]:
                logging.error(f"Syntax check failed for language '{language}': {syntax_result['error']}")
                return f"// Syntax error: {syntax_result['error']}\n{ai_code}"

            logging.debug(f"AI assistance generated and passed syntax check for language '{language}'.")
            return ai_code
        except Exception as e:
            logging.error(f"AI assistance failed for prompt='{prompt}': {e}")
            return "// AI assistance failed to generate code."

    def save_codebase(self, project_id: str, codebase: Dict[str, Any]) -> None:
        """
        Saves the generated codebase to the file system.

        :param project_id: Unique identifier for the project
        :param codebase: Organized codebase dictionary
        """
        try:
            base_path = os.path.join('generated_projects', project_id)
            if os.path.exists(base_path):
                shutil.rmtree(base_path)
            os.makedirs(base_path)
            for dir_path, files in codebase.items():
                full_dir_path = os.path.join(base_path, dir_path)
                os.makedirs(full_dir_path, exist_ok=True)
                for filename, content in files.items():
                    file_path = os.path.join(full_dir_path, filename)
                    with open(file_path, 'w') as file:
                        file.write(content)
                    logging.debug(f"Saved file '{file_path}'.")
            logging.info(f"Codebase saved successfully for project_id='{project_id}'.")
        except Exception as e:
            logging.error(f"Error saving codebase for project_id='{project_id}': {e}")

    def get_filename_for_feature(self, feature: str, language: str) -> str:
        """
        Determines the filename for a given feature based on language.

        :param feature: The feature name
        :param language: Programming language
        :return: Filename as a string
        """
        filename = f"{feature.lower()}.{self.get_language_extension(language)}"
        logging.debug(f"Determined filename '{filename}' for feature='{feature}', language='{language}'.")
        return filename

    def get_language_extension(self, language: str) -> str:
        """
        Returns the file extension for a given programming language.

        :param language: Programming language
        :return: File extension as a string
        """
        extensions = {
            'Python': 'py',
            'JavaScript': 'js',
            'C++': 'cpp',
            # Add more languages and extensions as needed
        }
        return extensions.get(language, 'txt')


# Example usage and test cases
if __name__ == "__main__":
    # Initialize real dependencies
    template_manager = TemplateManager()  # Use real TemplateManager
    ai_model = MockCodeGenModel()  # AI model will remain mock unless real one is implemented
    project_manager = ProjectManager()  # Use real ProjectManager

    # Initialize template generators
    django_generator = DjangoTemplateGenerator(template_manager)
    flask_generator = FlaskTemplateGenerator(template_manager)
    android_generator = AndroidTemplateGenerator(template_manager)
    ios_generator = IOSTemplateGenerator(template_manager)
    unity_generator = UnityTemplateGenerator(template_manager)
    unreal_generator = UnrealTemplateGenerator(template_manager)

    # Initialize CodeGenerator with all template generators
    code_generator = CodeGenerator(
        template_manager=template_manager,
        ai_model=ai_model,
        project_manager=project_manager,
        django_generator=django_generator,
        flask_generator=flask_generator,
        android_generator=android_generator,
        ios_generator=ios_generator,
        unity_generator=unity_generator,
        unreal_generator=unreal_generator
    )

    # Define user requirements
    user_requirements = {
        "project_type": "django",  # Could be 'flask', 'android', 'ios', 'unity', 'unreal'
        "project_name": "MyDjangoApp",
        "app_name": "mainapp"
    }

    # Generate codebase for the project
    project_id = "proj_12345"
    code_generator.generate_codebase(project_id, user_requirements)
