# security/dynamic_security_hardener.py

import logging
import os
import subprocess
import sys
import time
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging with RotatingFileHandler to prevent log files from becoming too large
from logging.handlers import RotatingFileHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Ensure the logs directory exists
os.makedirs('logs', exist_ok=True)

# Create a rotating file handler
handler = RotatingFileHandler('logs/dynamic_security_hardener.log', maxBytes=10**6, backupCount=5)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)


class DynamicSecurityHardener:
    """
    Implements a dynamic security engine that adapts to new threats in real-time.
    Automatically applies hardening techniques such as patching vulnerabilities,
    adjusting firewall settings, or tightening access controls based on detected threats.
    """

    def __init__(self):
        """
        Initializes the DynamicSecurityHardener with necessary configurations.
        """
        # Configuration parameters
        self.threat_monitor_api = os.getenv('THREAT_MONITOR_API')  # e.g., 'https://threatmonitor.example.com/api/threats'
        self.polling_interval = int(os.getenv('POLLING_INTERVAL', '60'))  # in seconds
        self.api_auth_token = os.getenv('THREAT_MONITOR_API_TOKEN')  # Authorization token if required

        if not self.threat_monitor_api:
            logger.error("THREAT_MONITOR_API is not set in environment variables.")
            sys.exit(1)

        # Headers for API requests
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f"Bearer {self.api_auth_token}" if self.api_auth_token else ''
        }

        logger.info("DynamicSecurityHardener initialized successfully.")

    def fetch_new_threats(self) -> Optional[List[Dict[str, Any]]]:
        """
        Fetches new threats from the Threat Monitor API.

        :return: List of threat dictionaries or None if an error occurs.
        """
        try:
            response = requests.get(self.threat_monitor_api, headers=self.headers, timeout=10)
            response.raise_for_status()
            threats = response.json().get('threats', [])
            logger.info(f"Fetched {len(threats)} new threats from Threat Monitor.")
            return threats
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching threats from Threat Monitor: {e}")
            return None

    def apply_security_measures(self, threat: Dict[str, Any]):
        """
        Determines and applies appropriate security measures based on the threat type.

        :param threat: A dictionary containing threat details.
        """
        threat_type = threat.get('type')
        severity = threat.get('severity', 'low').lower()

        logger.info(f"Applying security measures for threat: {threat_type} with severity: {severity}")

        if threat_type == 'vulnerability':
            self.patch_vulnerability(threat)
        elif threat_type == 'intrusion_attempt':
            self.adjust_firewall(threat)
        elif threat_type == 'unauthorized_access':
            self.tighten_access_controls(threat)
        else:
            logger.warning(f"Unknown threat type: {threat_type}. No action taken.")

    def patch_vulnerability(self, threat: Dict[str, Any]):
        """
        Applies patches to fix identified vulnerabilities.

        :param threat: A dictionary containing threat details.
        """
        package = threat.get('package')
        version = threat.get('version')
        logger.info(f"Patching vulnerability in package: {package}, version: {version}")

        try:
            # Example for Debian-based systems using apt
            # Update the package list
            subprocess.run(['sudo', 'apt-get', 'update'], check=True)
            # Upgrade the specific package
            subprocess.run(['sudo', 'apt-get', 'install', '--only-upgrade', f"{package}"], check=True)
            logger.info(f"Successfully patched package: {package}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to patch package {package}: {e}")

    def adjust_firewall(self, threat: Dict[str, Any]):
        """
        Adjusts firewall settings based on intrusion attempts.

        :param threat: A dictionary containing threat details.
        """
        source_ip = threat.get('source_ip')
        logger.info(f"Adjusting firewall to block source IP: {source_ip}")

        try:
            # Example using ufw (Uncomplicated Firewall)
            subprocess.run(['sudo', 'ufw', 'deny', 'from', source_ip], check=True)
            logger.info(f"Successfully blocked IP: {source_ip} using ufw.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to block IP {source_ip} using ufw: {e}")

    def tighten_access_controls(self, threat: Dict[str, Any]):
        """
        Tightens access controls based on unauthorized access attempts.

        :param threat: A dictionary containing threat details.
        """
        user = threat.get('user')
        resource = threat.get('resource')
        logger.info(f"Tightening access controls for user: {user} on resource: {resource}")

        try:
            # Example: Remove user access to the resource
            # This is highly dependent on how access controls are managed in your system
            # Below is a placeholder for actual access control adjustments

            # Placeholder command: Remove user's access from a file or system
            subprocess.run(['sudo', 'usermod', '-L', user], check=True)
            logger.info(f"Successfully tightened access controls for user: {user}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to tighten access controls for user {user}: {e}")

    def run(self):
        """
        Starts the dynamic security hardening process, continuously monitoring for new threats
        and applying security measures as necessary.
        """
        logger.info("Dynamic Security Hardener started.")
        while True:
            threats = self.fetch_new_threats()
            if threats:
                for threat in threats:
                    self.apply_security_measures(threat)
            else:
                logger.info("No new threats detected.")

            logger.debug(f"Sleeping for {self.polling_interval} seconds before next check.")
            time.sleep(self.polling_interval)


if __name__ == "__main__":
    try:
        hardener = DynamicSecurityHardener()
        hardener.run()
    except KeyboardInterrupt:
        logger.info("Dynamic Security Hardener stopped manually.")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
