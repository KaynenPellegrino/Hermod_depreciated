import os
import shutil
from typing import Dict, Any
from abc import ABC, abstractmethod

from src.modules.code_generation.template_manager import TemplateManagerInterface

import logging

# Configure logging
logging.basicConfig(
    filename='hermod_django_template.log',
    level=logging.DEBUG,
    format='%(asctime)s:%(levelname)s:%(message)s'
)


class DjangoTemplateGeneratorInterface(ABC):
    """
    Interface for Django App Template Generator.
    """
    @abstractmethod
    def generate_django_app(self, project_id: str, project_info: Dict[str, Any]) -> None:
        pass


class DjangoTemplateGenerator(DjangoTemplateGeneratorInterface):
    """
    Generates Django web applications based on templates.
    """
    def __init__(self, template_manager: TemplateManagerInterface, base_directory: str = 'generated_projects'):
        """
        Initializes the DjangoTemplateGenerator with necessary dependencies.

        :param template_manager: Instance of TemplateManagerInterface
        :param base_directory: Base directory where projects are stored
        """
        self.template_manager = template_manager
        self.base_directory = base_directory
        logging.info("DjangoTemplateGenerator initialized.")

    def generate_django_app(self, project_id: str, project_info: Dict[str, Any]) -> None:
        """
        Generates a Django application based on the provided project information.

        :param project_id: Unique identifier for the project
        :param project_info: Dictionary containing project details
        """
        logging.info(f"Generating Django app for project_id='{project_id}'.")
        try:
            project_path = os.path.join(self.base_directory, project_id, 'DjangoApp')
            if os.path.exists(project_path):
                logging.warning(f"Project path '{project_path}' already exists. Overwriting.")
                shutil.rmtree(project_path)
            os.makedirs(project_path)
            logging.debug(f"Created project directory at '{project_path}'.")

            # Load Django templates
            template_files = self.template_manager.list_templates()
            django_templates = [tpl for tpl in template_files if tpl.endswith('.py') or tpl.endswith('.html') or tpl.endswith('.txt')]

            for template_name in django_templates:
                template_content = self.template_manager.customize_template(template_name, project_info)
                # Determine the destination file path
                relative_path = os.path.relpath(os.path.join(self.template_manager.templates_dir, template_name), self.template_manager.templates_dir)
                dest_path = os.path.join(project_path, relative_path)
                dest_dir = os.path.dirname(dest_path)
                os.makedirs(dest_dir, exist_ok=True)
                with open(dest_path, 'w') as f:
                    f.write(template_content)
                logging.debug(f"Created file '{dest_path}' from template '{template_name}'.")

            logging.info(f"Django app generated successfully at '{project_path}'.")
        except Exception as e:
            logging.error(f"Failed to generate Django app for project_id='{project_id}': {e}")
            raise e


# Example usage and test case
if __name__ == "__main__":
    from src.modules.code_generation.template_manager import MockTemplateManager

    # Initialize mock TemplateManager
    mock_template_manager = MockTemplateManager()

    # Example Django templates
    django_manage_py_template = """#!/usr/bin/env python
    """ + """\"\"\"Django's command-line utility for administrative tasks.\"\"\" 
    import os
    import sys

    def main():
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', '{{ project_name }}.settings')
        try:
            from django.core.management import execute_from_command_line
        except ImportError as exc:
            raise ImportError(
                "Couldn't import Django. Are you sure it's installed and "
                "available on your PYTHONPATH environment variable? Did you "
                "forget to activate a virtual environment?"
            ) from exc
        execute_from_command_line(sys.argv)

    if __name__ == '__main__':
        main()
    """
    django_settings_py_template = """# Django Settings Template

    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parent.parent

    SECRET_KEY = 'your-secret-key'
    DEBUG = True
    ALLOWED_HOSTS = []

    INSTALLED_APPS = [
        'django.contrib.admin',
        'django.contrib.auth',
        'django.contrib.contenttypes',
        'django.contrib.sessions',
        'django.contrib.messages',
        'django.contrib.staticfiles',
        '{{ app_name }}',
    ]

    MIDDLEWARE = [
        'django.middleware.security.SecurityMiddleware',
        'django.contrib.sessions.middleware.SessionMiddleware',
        'django.middleware.common.CommonMiddleware',
        'django.middleware.csrf.CsrfViewMiddleware',
        'django.contrib.auth.middleware.AuthenticationMiddleware',
        'django.contrib.messages.middleware.MessageMiddleware',
        'django.middleware.clickjacking.XFrameOptionsMiddleware',
    ]

    ROOT_URLCONF = '{{ project_name }}.urls'

    TEMPLATES = [
        {
            'BACKEND': 'django.template.backends.django.DjangoTemplates',
            'DIRS': [],
            'APP_DIRS': True,
            'OPTIONS': {
                'context_processors': [
                    'django.template.context_processors.debug',
                    'django.template.context_processors.request',
                    'django.contrib.auth.context_processors.auth',
                    'django.contrib.messages.context_processors.messages',
                ],
            },
        },
    ]

    WSGI_APPLICATION = '{{ project_name }}.wsgi.application'
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.sqlite3',
            'NAME': BASE_DIR / 'db.sqlite3',
        }
    }

    AUTH_PASSWORD_VALIDATORS = [
        {
            'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
        },
        {
            'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
        },
        {
            'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
        },
        {
            'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
        },
    ]
    LANGUAGE_CODE = 'en-us'
    TIME_ZONE = 'UTC'
    USE_I18N = True
    USE_TZ = True
    STATIC_URL = '/static/'
    DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
    """
    django_urls_py_template = """# Django URLs Template

    from django.contrib import admin
    from django.urls import path

    urlpatterns = [
        path('admin/', admin.site.urls),
    ]
    """
    django_app_views_py_template = """# {{ app_name }} Views Template
    from django.shortcuts import render

    # Create your views here.
    """
    django_app_models_py_template = """# {{ app_name }} Models Template
    from django.db import models

    # Create your models here.
    """

    # Save example templates to MockTemplateManager
    mock_template_manager.save_template("manage.py", django_manage_py_template)
    mock_template_manager.save_template("settings.py", django_settings_py_template)
    mock_template_manager.save_template("urls.py", django_urls_py_template)
    mock_template_manager.save_template("views.py", django_app_views_py_template)
    mock_template_manager.save_template("models.py", django_app_models_py_template)

    # Initialize DjangoTemplateGenerator
    django_generator = DjangoTemplateGenerator(mock_template_manager)

    # Define project information
    project_info = {
        "project_name": "MyDjangoProject",
        "project_description": "A Django web application developed using Hermod templates.",
        "app_name": "mainapp"
    }

    # Generate Django app
    project_id = "proj_django_001"
    try:
        django_generator.generate_django_app(project_id, project_info)
        print(f"Django app '{project_id}' generated successfully.")
    except Exception as e:
        print(f"Failed to generate Django app: {e}")
