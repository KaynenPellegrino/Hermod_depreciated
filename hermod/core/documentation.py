# hermod/core/documentation.py

"""
Module: documentation.py

Generates documentation for the project.
"""

import os
import subprocess
from hermod.core.code_generation import generate_code, save_code
from hermod.utils.logger import setup_logger

# Initialize logger
logger = setup_logger()


def generate_documentation(project_path):
    logger.info("Generating documentation for project: %s", project_path)

    docs_path = os.path.join(project_path, 'docs')
    if not os.path.exists(docs_path):
        os.makedirs(docs_path)

    # Check if docs directory is empty
    if os.listdir(docs_path):
        logger.error("Docs directory is not empty. Sphinx cannot initialize in a non-empty directory.")
        return

    try:
        subprocess.run(['sphinx-quickstart', '--quiet', '--project', 'HermodProject', '--author', 'Your Name', '--sep',
                        '--makefile', '--no-batchfile'], cwd=docs_path, check=True)
        logger.info("Sphinx documentation initialized successfully.")
    except subprocess.CalledProcessError as e:
        logger.error("Failed to initialize Sphinx: %s", e);       return

    # Generate documentation from docstrings
    if os.path.exists('hermod/core'):
        try:
            subprocess.run(['sphinx-apidoc', '-o', 'docs/source', 'hermod/core'], cwd=project_path, check=True)
            subprocess.run(['make', 'html'], cwd=docs_path, check=True)
            logger.info("Documentation generated successfully at %s/build/html/index.html", docs_path)
        except subprocess.CalledProcessError as e:
            logger.error("Failed to generate documentation: %s", e)
    else:
        logger.error("hermod/core directory does not exist.")



def add_documentation_to_workflow(project_path):
    """
    Adds documentation generation to the development workflow.

    Args:
        project_path (str): The path to the project directory.
    """
    generate_documentation(project_path)
