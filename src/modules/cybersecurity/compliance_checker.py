# hermod/src/modules/data_management/compliance_checker.py

import logging
import os
import re
from typing import List, Dict, Any, Optional
import json
from datetime import datetime

# Import MetadataStorage from metadata_storage.py
from src.modules.data_management.staging import MetadataStorage

# Existing logging configuration
from logging.handlers import RotatingFileHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a rotating file handler
handler = RotatingFileHandler('logs/hermod_compliance_checker.log', maxBytes=10**6, backupCount=5)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)


class ComplianceReport:
    """
    Represents a compliance report detailing the status of various regulatory checks.
    """

    def __init__(self, entity: str, identifier: str):
        """
        Initializes the ComplianceReport.

        :param entity: The type of entity being checked (e.g., 'Project', 'Deployment', 'ClientPortal')
        :param identifier: Unique identifier for the entity (e.g., project_id, deployment_id)
        """
        self.entity = entity
        self.identifier = identifier
        self.timestamp = datetime.utcnow()
        self.compliance_status = {
            'GDPR': {'status': 'Not Checked', 'issues': []},
            'HIPAA': {'status': 'Not Checked', 'issues': []},
            'PCI_DSS': {'status': 'Not Checked', 'issues': []}
        }
        self.report_path = ""  # Initialize report_path

    def update_status(self, regulation: str, status: str, issues: Optional[List[str]] = None):
        if regulation in self.compliance_status:
            self.compliance_status[regulation]['status'] = status
            self.compliance_status[regulation]['issues'] = issues if issues else []
        else:
            logger.warning(f"Attempted to update unsupported regulation: {regulation}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            'entity': self.entity,
            'identifier': self.identifier,
            'timestamp': self.timestamp.isoformat(),
            'compliance_status': self.compliance_status
        }

    def save_report(self, directory: str = 'compliance_reports') -> str:
        """
        Saves the compliance report as a JSON file.

        :param directory: Directory where the report will be saved
        :return: Path to the saved report
        """
        os.makedirs(directory, exist_ok=True)
        report_filename = f"{self.entity}_{self.identifier}_compliance_report_{self.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        report_path = os.path.join(directory, report_filename)
        try:
            with open(report_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=4)
            logger.info(f"Compliance report saved at {report_path}")
            self.report_path = report_path  # Update report_path
            return report_path
        except Exception as e:
            logger.error(f"Failed to save compliance report: {e}")
            raise e


class ComplianceChecker:
    """
    Performs regulatory compliance checks on generated code and configuration files.
    """

    def __init__(self, base_directory: str = 'Hermod/generated_projects'):
        """
        Initializes the ComplianceChecker.

        :param base_directory: Path to the Hermod generated_projects directory
        """
        self.base_directory = os.path.abspath(base_directory)
        if not os.path.exists(self.base_directory):
            logger.error(f"Base directory '{self.base_directory}' does not exist.")
            raise FileNotFoundError(f"Base directory '{self.base_directory}' does not exist.")
        logger.info(f"ComplianceChecker initialized with base directory '{self.base_directory}'.")

        # Initialize MetadataStorage
        self.metadata_storage = MetadataStorage()
        logger.info("MetadataStorage initialized within ComplianceChecker.")

    def perform_compliance_checks(self):
        """
        Performs compliance checks on all projects, deployments, and client portal configurations.
        """
        logger.info("Starting compliance checks for all entities.")

        # Check all projects
        projects_path = os.path.join(self.base_directory, 'projects')
        if os.path.exists(projects_path):
            for project_id in os.listdir(projects_path):
                project_dir = os.path.join(projects_path, project_id)
                if os.path.isdir(project_dir):
                    logger.info(f"Performing compliance checks for Project '{project_id}'.")
                    compliance_report = ComplianceReport(entity='Project', identifier=project_id)
                    self._check_compliance(project_dir, compliance_report)
                    report_path = compliance_report.save_report()

                    # Save metadata about the compliance report
                    metadata = {
                        'project_id': project_id,
                        'entity': 'Project',
                        'report_path': report_path,
                        'checked_at': compliance_report.timestamp.isoformat()
                    }
                    self.metadata_storage.save_metadata(metadata, storage_type='sql')  # Choose 'sql' or 'mongodb' as per your setup
        else:
            logger.warning(f"Projects directory '{projects_path}' does not exist.")

        # Check deployments
        deployments_path = os.path.join(self.base_directory, 'deployments')
        if os.path.exists(deployments_path):
            for deployment_id in os.listdir(deployments_path):
                deployment_dir = os.path.join(deployments_path, deployment_id)
                if os.path.isdir(deployment_dir):
                    logger.info(f"Performing compliance checks for Deployment '{deployment_id}'.")
                    compliance_report = ComplianceReport(entity='Deployment', identifier=deployment_id)
                    self._check_compliance(deployment_dir, compliance_report, is_deployment=True)
                    report_path = compliance_report.save_report()

                    # Save metadata about the compliance report
                    metadata = {
                        'deployment_id': deployment_id,
                        'entity': 'Deployment',
                        'report_path': report_path,
                        'checked_at': compliance_report.timestamp.isoformat()
                    }
                    self.metadata_storage.save_metadata(metadata, storage_type='sql')  # Choose 'sql' or 'mongodb' as per your setup
        else:
            logger.warning(f"Deployments directory '{deployments_path}' does not exist.")

        # Check client portal
        client_portal_path = os.path.join(self.base_directory, 'client_portal')
        if os.path.exists(client_portal_path):
            for client_id in os.listdir(client_portal_path):
                client_dir = os.path.join(client_portal_path, client_id)
                if os.path.isdir(client_dir):
                    logger.info(f"Performing compliance checks for ClientPortal '{client_id}'.")
                    compliance_report = ComplianceReport(entity='ClientPortal', identifier=client_id)
                    self._check_compliance(client_dir, compliance_report, is_client_portal=True)
                    report_path = compliance_report.save_report()

                    # Save metadata about the compliance report
                    metadata = {
                        'client_id': client_id,
                        'entity': 'ClientPortal',
                        'report_path': report_path,
                        'checked_at': compliance_report.timestamp.isoformat()
                    }
                    self.metadata_storage.save_metadata(metadata, storage_type='sql')  # Choose 'sql' or 'mongodb' as per your setup
        else:
            logger.warning(f"Client Portal directory '{client_portal_path}' does not exist.")

        logger.info("All compliance checks completed.")

    def _check_compliance(self, target_dir: str, report: ComplianceReport, is_deployment: bool = False, is_client_portal: bool = False):
        """
        Performs individual compliance checks on a given directory.

        :param target_dir: Directory to perform compliance checks on
        :param report: ComplianceReport instance to update
        :param is_deployment: Flag indicating if the target is a deployment
        :param is_client_portal: Flag indicating if the target is a client portal
        """
        try:
            # Perform GDPR Compliance Check
            self.check_gdpr_compliance(target_dir, report, is_deployment, is_client_portal)

            # Perform HIPAA Compliance Check
            self.check_hipaa_compliance(target_dir, report, is_deployment, is_client_portal)

            # Perform PCI DSS Compliance Check
            self.check_pci_dss_compliance(target_dir, report, is_deployment, is_client_portal)
        except Exception as e:
            logger.error(f"Error during compliance checks in '{target_dir}': {e}")

    def check_gdpr_compliance(self, target_dir: str, report: ComplianceReport, is_deployment: bool, is_client_portal: bool):
        """
        Checks GDPR compliance by ensuring that personal data handling meets GDPR standards.

        :param target_dir: Directory to check
        :param report: ComplianceReport instance to update
        :param is_deployment: Flag indicating if the target is a deployment
        :param is_client_portal: Flag indicating if the target is a client portal
        """
        logger.info(f"Starting GDPR compliance check in '{target_dir}'.")
        issues = []

        # Example Checks:
        # 1. Ensure data encryption at rest
        # 2. Ensure data minimization principles
        # 3. Check for presence of data processing agreements

        # 1. Check for encryption usage in code/configurations (simplified)
        encryption_patterns = [r'encrypt\(', r'AES\.', r'cryptography\.Fernet']
        encryption_found = False
        for root, dirs, files in os.walk(target_dir):
            for file in files:
                if file.endswith(('.py', '.java', '.js', '.yaml', '.json', '.sh')):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read()
                            for pattern in encryption_patterns:
                                if re.search(pattern, content):
                                    encryption_found = True
                                    logger.debug(f"Encryption pattern '{pattern}' found in '{file_path}'.")
                                    break
                            if encryption_found:
                                break
                    except Exception as e:
                        logger.error(f"Error reading file '{file_path}': {e}")
            if encryption_found:
                break
        if not encryption_found:
            issues.append("Data encryption at rest not found in code or configurations.")

        # 2. Check for data minimization (e.g., limiting data collection)
        # This is highly context-specific; here, we check if sensitive fields are limited
        sensitive_fields = ['password', 'ssn', 'credit_card', 'email']
        for field in sensitive_fields:
            pattern = rf'[\'\"]{field}[\'\"]'
            found = False
            for root, dirs, files in os.walk(target_dir):
                for file in files:
                    if file.endswith(('.py', '.java', '.js', '.json', '.yaml', '.yml', '.sh')):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r') as f:
                                content = f.read().lower()
                                if re.search(pattern, content):
                                    found = True
                                    logger.debug(f"Sensitive field '{field}' found in '{file_path}'.")
                                    break
                        except Exception as e:
                            logger.error(f"Error reading file '{file_path}': {e}")
                if not found:
                    issues.append(f"Sensitive field '{field}' might not be handled appropriately.")
                    break

        # 3. Check for data processing agreements (simplified: look for specific files)
        agreements = ['DPA.md', 'data_processing_agreement.pdf', 'privacy_policy.md']
        agreement_found = False
        for agreement in agreements:
            agreement_path = os.path.join(target_dir, agreement)
            if os.path.exists(agreement_path):
                agreement_found = True
                logger.debug(f"Data Processing Agreement '{agreement}' found in '{target_dir}'.")
                break
        if not agreement_found:
            issues.append("Data Processing Agreement (DPA) not found.")

        # Update report
        if not issues:
            report.update_status('GDPR', 'Compliant')
            logger.info(f"GDPR compliance check passed in '{target_dir}'.")
        else:
            report.update_status('GDPR', 'Non-Compliant', issues)
            logger.warning(f"GDPR compliance check failed in '{target_dir}' with issues:")
            for issue in issues:
                logger.warning(f"GDPR Issue: {issue}")

    def check_hipaa_compliance(self, target_dir: str, report: ComplianceReport, is_deployment: bool, is_client_portal: bool):
        """
        Checks HIPAA compliance by ensuring that protected health information (PHI) is handled correctly.

        :param target_dir: Directory to check
        :param report: ComplianceReport instance to update
        :param is_deployment: Flag indicating if the target is a deployment
        :param is_client_portal: Flag indicating if the target is a client portal
        """
        logger.info(f"Starting HIPAA compliance check in '{target_dir}'.")
        issues = []

        # Example Checks:
        # 1. Ensure PHI is encrypted in transit and at rest
        # 2. Ensure access controls are in place
        # 3. Check for audit logs

        # 1. Check for encryption (similar to GDPR)
        encryption_patterns = [r'encrypt\(', r'AES\.', r'cryptography\.Fernet']
        encryption_found = False
        for root, dirs, files in os.walk(target_dir):
            for file in files:
                if file.endswith(('.py', '.java', '.js', '.yaml', '.json', '.sh')):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read()
                            for pattern in encryption_patterns:
                                if re.search(pattern, content):
                                    encryption_found = True
                                    logger.debug(f"Encryption pattern '{pattern}' found in '{file_path}'.")
                                    break
                            if encryption_found:
                                break
                    except Exception as e:
                        logger.error(f"Error reading file '{file_path}': {e}")
            if encryption_found:
                break
        if not encryption_found:
            issues.append("PHI encryption at rest not found in code or configurations.")

        # 2. Check for access controls (e.g., authentication mechanisms)
        access_control_patterns = [r'authenticate', r'auth\.login', r'authmiddleware', r'accesscontrol']
        access_control_found = False
        for root, dirs, files in os.walk(target_dir):
            for file in files:
                if file.endswith(('.py', '.java', '.js', '.sh')):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read().lower()
                            for pattern in access_control_patterns:
                                if re.search(pattern, content):
                                    access_control_found = True
                                    logger.debug(f"Access control pattern '{pattern}' found in '{file_path}'.")
                                    break
                            if access_control_found:
                                break
                    except Exception as e:
                        logger.error(f"Error reading file '{file_path}': {e}")
            if access_control_found:
                break
        if not access_control_found:
            issues.append("Access control mechanisms not found or insufficient.")

        # 3. Check for audit logs
        audit_log_patterns = [r'logger\.info', r'logger\.debug', r'logger\.warning', r'logging\.']
        audit_logs_found = False
        for root, dirs, files in os.walk(target_dir):
            for file in files:
                if file.endswith(('.py', '.java', '.js', '.sh')):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read()
                            if any(re.search(pattern, content) for pattern in audit_log_patterns):
                                audit_logs_found = True
                                logger.debug(f"Audit log pattern found in '{file_path}'.")
                                break
                    except Exception as e:
                        logger.error(f"Error reading file '{file_path}': {e}")
            if audit_logs_found:
                break
        if not audit_logs_found:
            issues.append("Audit logging not implemented.")

        # Update report
        if not issues:
            report.update_status('HIPAA', 'Compliant')
            logger.info(f"HIPAA compliance check passed in '{target_dir}'.")
        else:
            report.update_status('HIPAA', 'Non-Compliant', issues)
            logger.warning(f"HIPAA compliance check failed in '{target_dir}' with issues:")
            for issue in issues:
                logger.warning(f"HIPAA Issue: {issue}")

    def check_pci_dss_compliance(self, target_dir: str, report: ComplianceReport, is_deployment: bool, is_client_portal: bool):
        """
        Checks PCI DSS compliance by ensuring that payment card information is handled securely.

        :param target_dir: Directory to check
        :param report: ComplianceReport instance to update
        :param is_deployment: Flag indicating if the target is a deployment
        :param is_client_portal: Flag indicating if the target is a client portal
        """
        logger.info(f"Starting PCI DSS compliance check in '{target_dir}'.")
        issues = []

        # Example Checks:
        # 1. Ensure storage of cardholder data is minimized and encrypted
        # 2. Ensure secure transmission of cardholder data
        # 3. Implement strong access control measures

        # 1. Check for storage of cardholder data (simplified: look for specific patterns)
        card_data_patterns = [r'card_number', r'cvv', r'expiration_date', r'credit_card']
        card_data_found = False
        for root, dirs, files in os.walk(target_dir):
            for file in files:
                if file.endswith(('.py', '.java', '.js', '.json', '.yaml', '.yml', '.sh')):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read().lower()
                            if any(re.search(pattern, content) for pattern in card_data_patterns):
                                card_data_found = True
                                logger.debug(f"Cardholder data pattern found in '{file_path}'.")
                                break
                    except Exception as e:
                        logger.error(f"Error reading file '{file_path}': {e}")
            if card_data_found:
                break
        if card_data_found:
            issues.append("Potential storage of cardholder data detected. Ensure encryption and necessity.")

        # 2. Check for secure transmission (e.g., use of HTTPS)
        insecure_transmission_patterns = [r'http://', r'request\.url', r'fetch\(', r'axios\.get', r'requests\.get']
        insecure_transmission_found = False
        for root, dirs, files in os.walk(target_dir):
            for file in files:
                if file.endswith(('.py', '.java', '.js', '.sh')):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read().lower()
                            if any(re.search(pattern, content) for pattern in insecure_transmission_patterns):
                                insecure_transmission_found = True
                                issues.append("Insecure transmission of data detected (use HTTPS).")
                                logger.debug(f"Insecure transmission pattern found in '{file_path}'.")
                                break
                    except Exception as e:
                        logger.error(f"Error reading file '{file_path}': {e}")
            if insecure_transmission_found:
                break

        # 3. Check for access controls (similar to HIPAA)
        access_control_patterns = [r'authenticate', r'auth\.login', r'authmiddleware', r'accesscontrol']
        access_control_found = False
        for root, dirs, files in os.walk(target_dir):
            for file in files:
                if file.endswith(('.py', '.java', '.js', '.sh')):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read().lower()
                            for pattern in access_control_patterns:
                                if re.search(pattern, content):
                                    access_control_found = True
                                    logger.debug(f"Access control pattern '{pattern}' found in '{file_path}'.")
                                    break
                            if access_control_found:
                                break
                    except Exception as e:
                        logger.error(f"Error reading file '{file_path}': {e}")
            if access_control_found:
                break
        if not access_control_found:
            issues.append("Access control mechanisms not found or insufficient.")

        # Update report
        if not issues:
            report.update_status('PCI_DSS', 'Compliant')
            logger.info(f"PCI DSS compliance check passed in '{target_dir}'.")
        else:
            report.update_status('PCI_DSS', 'Non-Compliant', issues)
            logger.warning(f"PCI DSS compliance check failed in '{target_dir}' with issues:")
            for issue in issues:
                logger.warning(f"PCI DSS Issue: {issue}")

# Example usage and test cases
if __name__ == "__main__":
    # Define the base directory where Hermod's projects are generated
    base_dir = os.path.join(os.getcwd(), 'Hermod', 'generated_projects')

    # Initialize ComplianceChecker
    try:
        compliance_checker = ComplianceChecker(base_directory=base_dir)
    except Exception as e:
        logger.error(f"Failed to initialize ComplianceChecker: {e}")
        exit(1)

    # Perform compliance checks on all entities
    compliance_checker.perform_compliance_checks()

    print("Compliance checks completed. Check the 'compliance_reports/' directory and metadata storage for detailed reports.")

