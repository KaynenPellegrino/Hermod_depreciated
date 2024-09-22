import sys
import os

# Add the project root to the Python path
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_path)


import argparse
from hermod.core.code_generation import generate_code, save_code
from hermod.core.project_generator import generate_project
from hermod.core.workflow import develop_application
from hermod.utils.config import load_config
from hermod.utils.logger import setup_logger

# Initialize logger
logger = setup_logger()

def main():
    """
    Main function to execute Hermod's functionalities.
    """
    try:
        parser = argparse.ArgumentParser(
            description='Hermod: Autonomous AI-powered Development Assistant'
        )
        subparsers = parser.add_subparsers(dest='command', help='Available commands')

        # Subparser for generating code
        code_parser = subparsers.add_parser('generate_code', help='Generate code from a prompt')
        code_parser.add_argument('prompt', type=str, help='Prompt for code generation')
        code_parser.add_argument('--language', type=str, default='Python', help='Programming language')

        # Subparser for generating projects
        project_parser = subparsers.add_parser('generate_project', help='Generate a project from a specification')
        project_parser.add_argument('specification', type=str, help='Project specification')
        project_parser.add_argument('--name', type=str, default='GeneratedProject', help='Project name')
        project_parser.add_argument('--language', type=str, default='Python', help='Programming language')

        # Subparser for profiling
        profile_parser = subparsers.add_parser('profile', help='Profile the Hermod application')

        # Subparser for static code analysis
        analyze_parser = subparsers.add_parser('analyze', help='Run static code analysis on the codebase')

        # Subparser for generating tests
        test_gen_parser = subparsers.add_parser('generate_tests', help='Generate unit tests for a module')
        test_gen_parser.add_argument('module_path', type=str, help='Path to the module to generate tests for')

        # Subparser for auto debugging
        auto_debug_parser = subparsers.add_parser('auto_debug', help='Analyze logs and suggest fixes')
        auto_debug_parser.add_argument('--log_file', type=str, default='logs/hermod.log', help='Path to the log file')

        # Subparser for self refactoring
        refactor_parser = subparsers.add_parser('refactor', help='Refactor a module for improved readability and performance')
        refactor_parser.add_argument('module_path', type=str, help='Path to the module to refactor')

        # Subparser for developing applications
        develop_parser = subparsers.add_parser('develop_app', help='Develop an application end-to-end based on a specification')
        develop_parser.add_argument('specification', type=str, help='Project specification')
        develop_parser.add_argument('--name', type=str, default='GeneratedProject', help='Project name')
        develop_parser.add_argument('--language', type=str, default='Python', help='Programming language')

        args = parser.parse_args()

        # Load configuration
        config = load_config()

        if config is None:
            logger.warning("No configuration file found, proceeding with default settings.")
        else:
            logger.info("Configuration loaded: %s", config)

        if args.command == 'generate_code':
            logger.info("Generating code for prompt: %s in %s", args.prompt, args.language)
            code = generate_code(args.prompt, language=args.language)
            if code:
                filename = f'generated_code/{args.language.lower()}_code.{args.language.lower()}'
                save_code(code, filename)
            else:
                logger.error("Failed to generate code.")

        elif args.command == 'generate_project':
            logger.info("Generating project for specification: %s in %s", args.specification, args.language)
            generate_project(args.specification, project_name=args.name, language=args.language)

        elif args.command == 'profile':
            from hermod.profile_code import profile_application
            profile_application()

        elif args.command == 'analyze':
            from hermod.self_analysis import run_static_code_analysis
            run_static_code_analysis()

        elif args.command == 'generate_tests':
            from hermod.core.test_generator import generate_tests_for_module
            generate_tests_for_module(args.module_path)

        elif args.command == 'auto_debug':
            from hermod.core.auto_debugger import analyze_logs_and_suggest_fixes
            analyze_logs_and_suggest_fixes(args.log_file)

        elif args.command == 'refactor':
            from hermod.core.self_refactor import refactor_module
            refactor_module(args.module_path)

        elif args.command == 'develop_app':
            from hermod.core.workflow import develop_application
            develop_application(args.specification, project_name=args.name, language=args.language)

        else:
            parser.print_help()

    except Exception as e:
        if logger:
            logger.exception("An unexpected error occurred: %s", e)
        else:
            print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
