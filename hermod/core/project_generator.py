# hermod/core/project_generator.py

"""
Module: project_generator.py

Generates project structures and files based on specifications.
"""

import os
import json
from hermod.core.code_generation import generate_code, save_code
from hermod.utils.logger import setup_logger

logger = setup_logger()

def parse_specification(specification, language):
    """
    Parses the project specification and returns a list of components.

    Args:
        specification (str): The project requirements and specifications.
        language (str): The programming language to use.

    Returns:
        list: A list of components with descriptions and filenames.
    """
    prompt = (
        f"Based on the following project specification, list the components/modules needed "
        f"with their descriptions and filenames in {language}:\n\n{specification}\n\n"
        "Format the output as a JSON array of objects with 'description' and 'filename' keys."
    )
    components_json = generate_code(prompt)

    # Handle case where no components were generated
    if not components_json:
        logger.error("Code generation returned no response.")
        return []

    # Convert the JSON string to a Python list
    try:
        components = json.loads(components_json)
        return components
    except json.JSONDecodeError as json_error:
        logger.error(f"Failed to parse components JSON: {json_error}")
        return []


def generate_project(specification, project_name="GeneratedProject", language="Python"):
    logger.info("Project specification: %s", specification)

    if not os.path.exists(project_name):
        os.makedirs(project_name)

    main_py_path = os.path.join(project_name, 'main.py')
    if not os.path.exists(main_py_path):
        logger.info(f"Generating main.py for {project_name}.")
        main_py_code = generate_code(f"Generate a main entry point for the project in {language}.")
        if main_py_code:
            save_code(main_py_code, 'main.py', directory=project_name)
        else:
            logger.error(f"Failed to generate main.py for {project_name}. Please check the code generation logic.")
            return

    # Parse specification to get components
    components = parse_specification(specification, language)

    if not components:
        logger.error("No components generated from the specification.")
        return

    extension_mapping = {
        "Python": ".py",
        "JavaScript": ".js",
        "Java": ".java",
        "C++": ".cpp",
        "C#": ".cs",
        "Go": ".go",
        "Ruby": ".rb",
        "PHP": ".php",
        "Markdown": ".md",
    }

    for component in components:
        if component["filename"].lower() == "readme":
            ext = extension_mapping.get("Markdown", ".txt")
        else:
            ext = extension_mapping.get(language, ".txt")

        filename = f"{component['filename']}{ext}"
        prompt = f"{component['description']} Write the code in {language}."
        code = generate_code(prompt)

        if not code:
            logger.error(f"Code generation failed for component: {filename}")
            continue  # Continue to the next component if code generation fails.

        try:
            save_code(code, filename, directory=project_name)
            logger.info("Generated and saved %s in %s/", filename, project_name)
        except OSError as os_error:
            logger.error("Error saving code to %s: %s", filename, os_error)
            raise


