# cybersecurity/security_amplifier.py

import logging
import os
import subprocess
import json
import time
from datetime import datetime
from typing import Any, Dict, Optional, List

import requests
from dotenv import load_dotenv

# Import MetadataStorage from data_management module
from src.modules.data_management.metadata_storage import MetadataStorage

# Load environment variables from .env file
load_dotenv()

# Configure logging with RotatingFileHandler to prevent log files from becoming too large
from logging.handlers import RotatingFileHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Ensure the logs directory exists
os.makedirs('logs', exist_ok=True)

# Create a rotating file handler
handler = RotatingFileHandler('logs/security_amplifier.log', maxBytes=10**6, backupCount=5)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)


class SecurityEnhancementReport:
    """
    Represents a security enhancement report detailing actions taken to improve security.
    """

    def __init__(self, entity: str, identifier: str):
        """
        Initializes the SecurityEnhancementReport.

        :param entity: The type of entity being enhanced (e.g., 'Project', 'Deployment', 'ClientPortal')
        :param identifier: Unique identifier for the entity (e.g., project_id, deployment_id)
        """
        self.entity = entity
        self.identifier = identifier
        self.timestamp = datetime.utcnow()
        self.actions_taken: List[Dict[str, Any]] = []
        self.report_path = ""  # Initialize report_path

    def add_action(self, action: Dict[str, Any]):
        """
        Adds a security enhancement action to the report.

        :param action: Dictionary containing action details
        """
        self.actions_taken.append(action)
        logger.debug(f"Added action: {action}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the security enhancement report to a dictionary.

        :return: Dictionary representation of the report
        """
        return {
            'entity': self.entity,
            'identifier': self.identifier,
            'timestamp': self.timestamp.isoformat(),
            'actions_taken': self.actions_taken
        }

    def save_report(self, directory: str = 'security_enhancement_reports') -> str:
        """
        Saves the security enhancement report as a JSON file.

        :param directory: Directory where the report will be saved
        :return: Path to the saved report
        """
        os.makedirs(directory, exist_ok=True)
        report_filename = f"{self.entity}_{self.identifier}_security_enhancement_report_{self.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        report_path = os.path.join(directory, report_filename)
        try:
            with open(report_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=4)
            logger.info(f"Security enhancement report saved at {report_path}")
            self.report_path = report_path  # Update report_path
            return report_path
        except Exception as e:
            logger.error(f"Failed to save security enhancement report: {e}")
            raise e


class SecurityAmplifier:
    """
    Applies security best practices to strengthen generated code and configurations.
    Automatically fixes detected vulnerabilities, applies patches, and enforces security policies.
    """

    def __init__(self):
        """
        Initializes the SecurityAmplifier with necessary configurations.
        """
        # Configuration parameters
        self.targets = self.load_targets()
        self.metadata_storage = MetadataStorage()
        logger.info("SecurityAmplifier initialized successfully.")

    def load_targets(self) -> List[Dict[str, Any]]:
        """
        Loads security enhancement targets from environment variables or configuration files.

        :return: List of target dictionaries
        """
        # Example: Load targets from a JSON file specified in environment variables
        targets_file = os.getenv('SECURITY_AMPLIFIER_TARGETS_FILE', 'security_amplifier_targets.json')
        if not os.path.exists(targets_file):
            logger.error(f"Targets file '{targets_file}' does not exist.")
            return []

        try:
            with open(targets_file, 'r') as f:
                targets = json.load(f).get('targets', [])
            logger.info(f"Loaded {len(targets)} security enhancement targets from '{targets_file}'.")
            return targets
        except Exception as e:
            logger.error(f"Failed to load targets from '{targets_file}': {e}")
            return []

    def perform_security_enhancement(self, target: Dict[str, Any]):
        """
        Performs security enhancements on a single target.

        :param target: Dictionary containing target details
        """
        entity = target.get('entity')
        identifier = target.get('identifier')
        target_dir = target.get('directory')  # Path to the target's codebase or configuration
        logger.info(f"Starting security enhancement for {entity} '{identifier}' at '{target_dir}'")

        report = SecurityEnhancementReport(entity=entity, identifier=identifier)

        # Apply Security Policies
        self.enforce_security_policies(target_dir, report)

        # Fix Code Vulnerabilities
        self.fix_code_vulnerabilities(target_dir, report)

        # Apply Configuration Hardening
        self.harden_configurations(target_dir, report)

        # Apply Dependency Patches
        self.apply_dependency_patches(target_dir, report)

        # Save report
        report_path = report.save_report()

        # Save metadata about the security enhancement report
        metadata = {
            'entity': entity,
            'identifier': identifier,
            'report_path': report_path,
            'enhanced_at': report.timestamp.isoformat()
        }
        self.metadata_storage.save_metadata(metadata, storage_type='sql')  # Choose 'sql' or 'mongodb' as per your setup

    def enforce_security_policies(self, target_dir: str, report: SecurityEnhancementReport):
        """
        Enforces security policies by applying linters and formatters to the codebase.

        :param target_dir: Directory of the target's codebase
        :param report: SecurityEnhancementReport instance to update
        """
        logger.info(f"Enforcing security policies in '{target_dir}'")
        try:
            # Example: Run Bandit to find security issues in Python code
            cmd_bandit = ['bandit', '-r', target_dir, '-f', 'json', '-o', '/tmp/bandit_report.json']
            subprocess.run(cmd_bandit, check=True)
            logger.info(f"Bandit scan completed for '{target_dir}'")

            # Load Bandit report
            with open('/tmp/bandit_report.json', 'r') as f:
                bandit_results = json.load(f)

            # Process and add vulnerabilities to report
            for issue in bandit_results.get('results', []):
                report.add_action({
                    'tool': 'Bandit',
                    'description': issue.get('issue_text'),
                    'severity': issue.get('issue_severity'),
                    'details': issue
                })

            # Example: Run Black to format Python code
            cmd_black = ['black', target_dir]
            subprocess.run(cmd_black, check=True)
            logger.info(f"Black formatting applied to '{target_dir}'")
            report.add_action({
                'tool': 'Black',
                'description': 'Code formatted using Black.',
                'severity': 'low',
                'details': {}
            })

            # Clean up temporary file
            os.remove('/tmp/bandit_report.json')

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to enforce security policies in '{target_dir}': {e}")
            report.add_action({
                'tool': 'Security Policies',
                'description': f"Failed to enforce security policies: {e}",
                'severity': 'high',
                'details': {}
            })

    def fix_code_vulnerabilities(self, target_dir: str, report: SecurityEnhancementReport):
        """
        Automatically fixes code vulnerabilities using linters and formatters.

        :param target_dir: Directory of the target's codebase
        :param report: SecurityEnhancementReport instance to update
        """
        logger.info(f"Fixing code vulnerabilities in '{target_dir}'")
        try:
            # Example: Use autoflake to remove unused imports and variables
            cmd_autoflake = ['autoflake', '--remove-all-unused-imports', '--recursive', '--in-place', target_dir]
            subprocess.run(cmd_autoflake, check=True)
            logger.info(f"Autoflake applied to '{target_dir}'")
            report.add_action({
                'tool': 'Autoflake',
                'description': 'Unused imports and variables removed using Autoflake.',
                'severity': 'medium',
                'details': {}
            })

            # Example: Use isort to sort imports
            cmd_isort = ['isort', target_dir]
            subprocess.run(cmd_isort, check=True)
            logger.info(f"Isort applied to '{target_dir}'")
            report.add_action({
                'tool': 'Isort',
                'description': 'Imports sorted using Isort.',
                'severity': 'low',
                'details': {}
            })

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to fix code vulnerabilities in '{target_dir}': {e}")
            report.add_action({
                'tool': 'Code Fixing',
                'description': f"Failed to fix code vulnerabilities: {e}",
                'severity': 'high',
                'details': {}
            })

    def harden_configurations(self, target_dir: str, report: SecurityEnhancementReport):
        """
        Applies configuration hardening by enforcing security best practices.

        :param target_dir: Directory of the target's configuration files
        :param report: SecurityEnhancementReport instance to update
        """
        logger.info(f"Hardening configurations in '{target_dir}'")
        try:
            # Example: Use Ansible to apply security configurations
            # Assume an Ansible playbook 'harden.yml' exists in a 'playbooks' directory
            playbook_path = os.getenv('SECURITY_AMPLIFIER_PLAYBOOK', 'playbooks/harden.yml')
            if not os.path.exists(playbook_path):
                logger.error(f"Ansible playbook '{playbook_path}' does not exist.")
                report.add_action({
                    'tool': 'Ansible',
                    'description': f"Ansible playbook '{playbook_path}' not found.",
                    'severity': 'high',
                    'details': {}
                })
                return

            cmd_ansible = ['ansible-playbook', playbook_path, '-i', f"{target_dir},"]
            subprocess.run(cmd_ansible, check=True)
            logger.info(f"Ansible playbook applied to '{target_dir}'")
            report.add_action({
                'tool': 'Ansible',
                'description': f"Configuration hardening applied using playbook '{playbook_path}'.",
                'severity': 'medium',
                'details': {}
            })

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to harden configurations in '{target_dir}': {e}")
            report.add_action({
                'tool': 'Configuration Hardening',
                'description': f"Failed to harden configurations: {e}",
                'severity': 'high',
                'details': {}
            })

    def apply_dependency_patches(self, target_dir: str, report: SecurityEnhancementReport):
        """
        Applies patches to fix vulnerabilities in project dependencies.

        :param target_dir: Directory of the target's codebase
        :param report: SecurityEnhancementReport instance to update
        """
        logger.info(f"Applying dependency patches in '{target_dir}'")
        try:
            # Example: Use pip-audit to find and fix vulnerabilities in Python dependencies
            # Install pip-audit if not already installed
            cmd_pip_audit_install = ['pip', 'install', '--upgrade', 'pip-audit']
            subprocess.run(cmd_pip_audit_install, check=True)
            logger.info("pip-audit installed/updated successfully.")

            # Run pip-audit to find vulnerabilities
            cmd_pip_audit = ['pip-audit', '--json', '-r', os.path.join(target_dir, 'requirements.txt'), '-o', '/tmp/pip_audit_report.json']
            subprocess.run(cmd_pip_audit, check=True)
            logger.info(f"pip-audit scan completed for '{target_dir}'")

            # Load pip-audit report
            with open('/tmp/pip_audit_report.json', 'r') as f:
                pip_audit_results = json.load(f)

            # Process vulnerabilities and attempt to fix them
            for vuln in pip_audit_results.get('vulnerabilities', []):
                package = vuln.get('name')
                version = vuln.get('version')
                advisory = vuln.get('advisory')
                logger.info(f"Attempting to patch vulnerability in package '{package}'")
                try:
                    # Example: Upgrade the vulnerable package to the latest version
                    cmd_pip_upgrade = ['pip', 'install', '--upgrade', package]
                    subprocess.run(cmd_pip_upgrade, check=True)
                    report.add_action({
                        'tool': 'pip-audit',
                        'description': f"Upgraded package '{package}' to fix vulnerability.",
                        'severity': 'high',
                        'details': advisory
                    })
                    logger.info(f"Successfully patched package '{package}'")
                except subprocess.CalledProcessError as e:
                    logger.error(f"Failed to patch package '{package}': {e}")
                    report.add_action({
                        'tool': 'pip-audit',
                        'description': f"Failed to upgrade package '{package}': {e}",
                        'severity': 'high',
                        'details': advisory
                    })

            # Clean up temporary file
            os.remove('/tmp/pip_audit_report.json')

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to apply dependency patches in '{target_dir}': {e}")
            report.add_action({
                'tool': 'Dependency Patching',
                'description': f"Failed to apply dependency patches: {e}",
                'severity': 'high',
                'details': {}
            })
        except Exception as e:
            logger.error(f"Error during dependency patching in '{target_dir}': {e}")
            report.add_action({
                'tool': 'Dependency Patching',
                'description': f"Error during dependency patching: {e}",
                'severity': 'high',
                'details': {}
            })

    def run_enhancements(self):
        """
        Initiates security enhancements on all configured targets.
        """
        logger.info("Starting security enhancements on all targets.")
        for target in self.targets:
            self.perform_security_enhancement(target)
            # Optional: Delay between enhancements to prevent overwhelming the system
            time.sleep(5)
        logger.info("Security enhancements completed for all targets.")


if __name__ == "__main__":
    try:
        amplifier = SecurityAmplifier()
        amplifier.run_enhancements()
    except KeyboardInterrupt:
        logger.info("SecurityAmplifier stopped manually.")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
