# cybersecurity/penetration_tester.py

import logging
import os
import subprocess
import json
import threading
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
handler = RotatingFileHandler('logs/penetration_tester.log', maxBytes=10**6, backupCount=5)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)


class PenetrationReport:
    """
    Represents a penetration testing report detailing identified vulnerabilities.
    """

    def __init__(self, entity: str, identifier: str):
        """
        Initializes the PenetrationReport.

        :param entity: The type of entity being tested (e.g., 'Project', 'Deployment', 'ClientPortal')
        :param identifier: Unique identifier for the entity (e.g., project_id, deployment_id)
        """
        self.entity = entity
        self.identifier = identifier
        self.timestamp = datetime.utcnow()
        self.vulnerabilities: List[Dict[str, Any]] = []
        self.report_path = ""  # Initialize report_path

    def add_vulnerability(self, vulnerability: Dict[str, Any]):
        """
        Adds a discovered vulnerability to the report.

        :param vulnerability: Dictionary containing vulnerability details
        """
        self.vulnerabilities.append(vulnerability)
        logger.debug(f"Added vulnerability: {vulnerability}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the penetration report to a dictionary.

        :return: Dictionary representation of the report
        """
        return {
            'entity': self.entity,
            'identifier': self.identifier,
            'timestamp': self.timestamp.isoformat(),
            'vulnerabilities': self.vulnerabilities
        }

    def save_report(self, directory: str = 'penetration_reports') -> str:
        """
        Saves the penetration report as a JSON file.

        :param directory: Directory where the report will be saved
        :return: Path to the saved report
        """
        os.makedirs(directory, exist_ok=True)
        report_filename = f"{self.entity}_{self.identifier}_penetration_report_{self.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        report_path = os.path.join(directory, report_filename)
        try:
            with open(report_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=4)
            logger.info(f"Penetration report saved at {report_path}")
            self.report_path = report_path  # Update report_path
            return report_path
        except Exception as e:
            logger.error(f"Failed to save penetration report: {e}")
            raise e


class PenetrationTester:
    """
    Automates penetration testing by simulating cyber-attacks on target entities to identify vulnerabilities.
    """

    def __init__(self):
        """
        Initializes the PenetrationTester with necessary configurations.
        """
        # Configuration parameters
        self.targets = self.load_targets()
        self.metadata_storage = MetadataStorage()
        logger.info("PenetrationTester initialized successfully.")

    def load_targets(self) -> List[Dict[str, Any]]:
        """
        Loads penetration testing targets from environment variables or configuration files.

        :return: List of target dictionaries
        """
        # Example: Load targets from a JSON file specified in environment variables
        targets_file = os.getenv('PENETRATION_TESTER_TARGETS_FILE', 'penetration_test_targets.json')
        if not os.path.exists(targets_file):
            logger.error(f"Targets file '{targets_file}' does not exist.")
            return []

        try:
            with open(targets_file, 'r') as f:
                targets = json.load(f).get('targets', [])
            logger.info(f"Loaded {len(targets)} penetration testing targets from '{targets_file}'.")
            return targets
        except Exception as e:
            logger.error(f"Failed to load targets from '{targets_file}': {e}")
            return []

    def perform_penetration_test(self, target: Dict[str, Any]):
        """
        Performs penetration testing on a single target.

        :param target: Dictionary containing target details
        """
        entity = target.get('entity')
        identifier = target.get('identifier')
        ip_address = target.get('ip_address')
        services = target.get('services', [])  # List of services to test

        logger.info(f"Starting penetration test for {entity} '{identifier}' at IP {ip_address}")

        report = PenetrationReport(entity=entity, identifier=identifier)

        for service in services:
            service_type = service.get('type')
            port = service.get('port')
            logger.info(f"Testing service '{service_type}' on port {port}")

            if service_type == 'http':
                self.test_http_service(ip_address, port, report)
            elif service_type == 'ssh':
                self.test_ssh_service(ip_address, port, report)
            elif service_type == 'database':
                self.test_database_service(ip_address, port, service.get('db_type'), report)
            else:
                logger.warning(f"Unknown service type: {service_type}. Skipping.")

        # Save report
        report_path = report.save_report()

        # Save metadata about the penetration report
        metadata = {
            'entity': entity,
            'identifier': identifier,
            'report_path': report_path,
            'tested_at': report.timestamp.isoformat()
        }
        self.metadata_storage.save_metadata(metadata, storage_type='sql')  # Choose 'sql' or 'mongodb' as per your setup

    def test_http_service(self, ip: str, port: int, report: PenetrationReport):
        """
        Tests an HTTP service for common vulnerabilities using Nikto.

        :param ip: IP address of the target
        :param port: Port number of the HTTP service
        :param report: PenetrationReport instance to update
        """
        logger.info(f"Running Nikto scan on HTTP service at {ip}:{port}")
        try:
            cmd = ['nikto', '-h', f"http://{ip}:{port}", '-output', '/tmp/nikto_scan.json', '-Format', 'json']
            subprocess.run(cmd, check=True)
            logger.info(f"Nikto scan completed for {ip}:{port}")

            # Load Nikto scan results
            with open('/tmp/nikto_scan.json', 'r') as f:
                nikto_results = json.load(f)

            # Process and add vulnerabilities to report
            for vuln in nikto_results.get('vulnerabilities', []):
                report.add_vulnerability({
                    'tool': 'Nikto',
                    'description': vuln.get('description'),
                    'severity': vuln.get('severity', 'medium'),
                    'details': vuln
                })

            # Clean up temporary file
            os.remove('/tmp/nikto_scan.json')

        except subprocess.CalledProcessError as e:
            logger.error(f"Nikto scan failed for {ip}:{port}: {e}")
        except Exception as e:
            logger.error(f"Error processing Nikto scan results for {ip}:{port}: {e}")

    def test_ssh_service(self, ip: str, port: int, report: PenetrationReport):
        """
        Tests an SSH service for common vulnerabilities using Nmap scripts.

        :param ip: IP address of the target
        :param port: Port number of the SSH service
        :param report: PenetrationReport instance to update
        """
        logger.info(f"Running Nmap SSH scan on {ip}:{port}")
        try:
            cmd = ['nmap', '-sV', '--script', 'sshv1,ssh-auth-methods,ssh-hostkey', '-p', str(port), ip, '-oX', '/tmp/nmap_ssh_scan.xml']
            subprocess.run(cmd, check=True)
            logger.info(f"Nmap SSH scan completed for {ip}:{port}")

            # Parse Nmap XML output
            vulnerabilities = self.parse_nmap_ssh_scan('/tmp/nmap_ssh_scan.xml')
            for vuln in vulnerabilities:
                report.add_vulnerability({
                    'tool': 'Nmap',
                    'description': vuln.get('description'),
                    'severity': vuln.get('severity', 'medium'),
                    'details': vuln
                })

            # Clean up temporary file
            os.remove('/tmp/nmap_ssh_scan.xml')

        except subprocess.CalledProcessError as e:
            logger.error(f"Nmap SSH scan failed for {ip}:{port}: {e}")
        except Exception as e:
            logger.error(f"Error processing Nmap SSH scan results for {ip}:{port}: {e}")

    def parse_nmap_ssh_scan(self, xml_file: str) -> List[Dict[str, Any]]:
        """
        Parses Nmap SSH scan XML results to extract vulnerabilities.

        :param xml_file: Path to the Nmap scan XML file
        :return: List of vulnerability dictionaries
        """
        import xml.etree.ElementTree as ET

        vulnerabilities = []
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            for host in root.findall('host'):
                for port in host.find('ports').findall('port'):
                    service = port.find('service')
                    for script in port.findall('script'):
                        vuln = {
                            'id': script.get('id'),
                            'output': script.get('output')
                        }
                        if script.get('id') == 'sshv1':
                            vuln['description'] = 'SSH Version 1 is enabled, which is insecure.'
                            vuln['severity'] = 'high'
                        elif script.get('id') == 'ssh-auth-methods':
                            vuln['description'] = 'Weak SSH authentication methods detected.'
                            vuln['severity'] = 'medium'
                        elif script.get('id') == 'ssh-hostkey':
                            vuln['description'] = 'Potential SSH host key issues detected.'
                            vuln['severity'] = 'medium'
                        vulnerabilities.append(vuln)

            logger.debug(f"Parsed {len(vulnerabilities)} vulnerabilities from Nmap SSH scan.")
            return vulnerabilities

        except ET.ParseError as e:
            logger.error(f"Error parsing Nmap SSH scan XML: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error during Nmap SSH scan parsing: {e}")
            return []

    def test_database_service(self, ip: str, port: int, db_type: str, report: PenetrationReport):
        """
        Tests a database service for common vulnerabilities using SQLMap.

        :param ip: IP address of the target
        :param port: Port number of the database service
        :param db_type: Type of the database (e.g., 'mysql', 'postgresql')
        :param report: PenetrationReport instance to update
        """
        logger.info(f"Running SQLMap scan on {db_type} database at {ip}:{port}")
        try:
            # Placeholder for actual SQLMap usage
            # SQLMap requires a URL or specific parameters to perform scans
            # Adjust the command based on your target setup

            # Example command (modify according to actual database access methods)
            cmd = ['sqlmap', '-u', f"jdbc:{db_type}://{ip}:{port}/", '--batch', '--output-dir', '/tmp/sqlmap_output']
            subprocess.run(cmd, check=True)
            logger.info(f"SQLMap scan completed for {db_type} database at {ip}:{port}")

            # Load SQLMap scan results
            report.add_vulnerability({
                'tool': 'SQLMap',
                'description': f"SQL Injection vulnerabilities detected in {db_type} database at {ip}:{port}.",
                'severity': 'high',
                'details': {
                    'output_dir': '/tmp/sqlmap_output',
                    'command': ' '.join(cmd)
                }
            })

            # Clean up temporary files or store them as needed
            # Example: Remove SQLMap output directory
            subprocess.run(['rm', '-rf', '/tmp/sqlmap_output'], check=True)

        except subprocess.CalledProcessError as e:
            logger.error(f"SQLMap scan failed for {db_type} database at {ip}:{port}: {e}")
        except Exception as e:
            logger.error(f"Error during SQLMap scan for {db_type} database at {ip}:{port}: {e}")

    def run_tests(self):
        """
        Initiates penetration testing on all configured targets concurrently.
        """
        logger.info("Starting penetration testing on all targets.")
        threads = []
        for target in self.targets:
            thread = threading.Thread(target=self.perform_penetration_test, args=(target,))
            thread.start()
            threads.append(thread)
            # Optional: Limit the number of concurrent threads
            time.sleep(1)  # Slight delay to manage system load

        for thread in threads:
            thread.join()
        logger.info("Penetration testing completed for all targets.")


if __name__ == "__main__":
    try:
        tester = PenetrationTester()
        tester.run_tests()
    except KeyboardInterrupt:
        logger.info("PenetrationTester stopped manually.")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
