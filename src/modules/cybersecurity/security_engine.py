# cybersecurity/security_engine.py

import json
import logging
import os
import time
from datetime import datetime

import schedule
from dotenv import load_dotenv

# Import other cybersecurity modules
from staging import ComplianceChecker, DynamicSecurityHardener, PenetrationTester, SecurityAmplifier, ThreatMonitor, VulnerabilityScanner
# Import MetadataStorage from data_management module
from src.modules.data_management.staging import MetadataStorage

# Load environment variables from .env file
load_dotenv()

# Configure logging with RotatingFileHandler to prevent log files from becoming too large
from logging.handlers import RotatingFileHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Ensure the logs directory exists
os.makedirs('logs', exist_ok=True)

# Create a rotating file handler
handler = RotatingFileHandler('logs/security_engine.log', maxBytes=10 ** 6, backupCount=5)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)


class SecurityEngine:
    """
    Core Security Engine that orchestrates various cybersecurity functionalities within Hermod.
    Integrates with other modules to provide comprehensive security features.
    """

    def __init__(self):
        """
        Initializes the SecurityEngine with necessary configurations and module instances.
        """
        # Initialize Metadata Storage
        self.metadata_storage = MetadataStorage()
        logger.info("MetadataStorage initialized within SecurityEngine.")

        # Initialize Cybersecurity Modules
        self.compliance_checker = ComplianceChecker()
        self.dynamic_security_hardener = DynamicSecurityHardener()
        self.penetration_tester = PenetrationTester()
        self.security_amplifier = SecurityAmplifier()
        self.vulnerability_scanner = VulnerabilityScanner()
        self.threat_monitor = ThreatMonitor()

        # Configuration parameters
        self.polling_interval = int(os.getenv('SECURITY_ENGINE_POLLING_INTERVAL', '60'))  # in seconds
        self.report_directory = os.getenv('SECURITY_ENGINE_REPORT_DIRECTORY', 'security_engine_reports')

        # Ensure the report directory exists
        os.makedirs(self.report_directory, exist_ok=True)

        logger.info("SecurityEngine initialized successfully.")

    def schedule_tasks(self):
        """
        Schedules periodic tasks for the Security Engine.
        """
        # Schedule Vulnerability Scanning every day at 2 AM
        schedule.every().day.at("02:00").do(self.run_vulnerability_scanning)

        # Schedule Compliance Checking every Sunday at 3 AM
        schedule.every().sunday.at("03:00").do(self.run_compliance_checking)

        # Schedule Penetration Testing every month on the first day at 4 AM
        schedule.every(4).weeks.do(self.run_penetration_testing)

        # Schedule Threat Monitoring every minute
        schedule.every(1).minutes.do(self.run_threat_monitoring)

        # Schedule Security Enhancements based on detected threats
        # This could be event-driven rather than scheduled

        logger.info("Scheduled security engine tasks.")

    def run_vulnerability_scanning(self):
        """
        Initiates vulnerability scanning using the VulnerabilityScanner module.
        """
        logger.info("Initiating Vulnerability Scanning.")
        try:
            self.vulnerability_scanner.run_scans()
            logger.info("Vulnerability Scanning completed successfully.")
        except Exception as e:
            logger.error(f"Error during Vulnerability Scanning: {e}")
            # Optionally, send an alert
            self.send_alert(
                subject="Security Engine Alert: Vulnerability Scanning Failure",
                message=f"An error occurred during vulnerability scanning: {e}"
            )

    def run_compliance_checking(self):
        """
        Initiates compliance checking using the ComplianceChecker module.
        """
        logger.info("Initiating Compliance Checking.")
        try:
            self.compliance_checker.perform_compliance_checks()
            logger.info("Compliance Checking completed successfully.")
        except Exception as e:
            logger.error(f"Error during Compliance Checking: {e}")
            # Optionally, send an alert
            self.send_alert(
                subject="Security Engine Alert: Compliance Checking Failure",
                message=f"An error occurred during compliance checking: {e}"
            )

    def run_penetration_testing(self):
        """
        Initiates penetration testing using the PenetrationTester module.
        """
        logger.info("Initiating Penetration Testing.")
        try:
            self.penetration_tester.run_tests()
            logger.info("Penetration Testing completed successfully.")
        except Exception as e:
            logger.error(f"Error during Penetration Testing: {e}")
            # Optionally, send an alert
            self.send_alert(
                subject="Security Engine Alert: Penetration Testing Failure",
                message=f"An error occurred during penetration testing: {e}"
            )

    def run_threat_monitoring(self):
        """
        Monitors for threats using the ThreatMonitor module and triggers responses.
        """
        logger.info("Running Threat Monitoring.")
        try:
            threats = self.threat_monitor.fetch_new_threats()
            if threats:
                logger.info(f"Detected {len(threats)} new threats.")
                for threat in threats:
                    # Log the threat
                    logger.info(f"Processing threat: {threat}")

                    # Apply dynamic security hardening based on threat
                    self.dynamic_security_hardener.apply_security_measures(threat)

                    # Optionally, trigger other modules or analytics
            else:
                logger.info("No new threats detected.")
        except Exception as e:
            logger.error(f"Error during Threat Monitoring: {e}")
            # Optionally, send an alert
            self.send_alert(
                subject="Security Engine Alert: Threat Monitoring Failure",
                message=f"An error occurred during threat monitoring: {e}"
            )

    def run_security_enhancements(self):
        """
        Initiates security enhancements using the SecurityAmplifier module.
        """
        logger.info("Initiating Security Enhancements.")
        try:
            self.security_amplifier.run_enhancements()
            logger.info("Security Enhancements completed successfully.")
        except Exception as e:
            logger.error(f"Error during Security Enhancements: {e}")
            # Optionally, send an alert
            self.send_alert(
                subject="Security Engine Alert: Security Enhancements Failure",
                message=f"An error occurred during security enhancements: {e}"
            )

    def send_alert(self, subject: str, message: str):
        """
        Sends an alert to administrators via email.

        :param subject: Subject of the alert email
        :param message: Body of the alert email
        """
        import smtplib
        from email.mime.text import MIMEText

        smtp_server = os.getenv('ALERT_SMTP_SERVER')
        smtp_port = os.getenv('ALERT_SMTP_PORT', '587')
        smtp_user = os.getenv('ALERT_SMTP_USER')
        smtp_password = os.getenv('ALERT_SMTP_PASSWORD')
        alert_recipient = os.getenv('ALERT_RECIPIENT')  # e.g., 'admin@example.com'

        if not all([smtp_server, smtp_port, smtp_user, smtp_password, alert_recipient]):
            logger.error("SMTP configuration incomplete. Cannot send alert.")
            return

        try:
            msg = MIMEText(message)
            msg['Subject'] = subject
            msg['From'] = smtp_user
            msg['To'] = alert_recipient

            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_user, [alert_recipient], msg.as_string())
            server.quit()
            logger.info(f"Alert sent to {alert_recipient} with subject: {subject}")
        except Exception as e:
            logger.error(f"Failed to send alert email: {e}")

    def generate_security_report(self):
        """
        Generates a comprehensive security report by aggregating data from all modules.
        """
        logger.info("Generating comprehensive Security Report.")
        try:
            # Fetch reports from Metadata Storage
            compliance_reports = self.metadata_storage.get_all_reports(entity='Compliance')
            penetration_reports = self.metadata_storage.get_all_reports(entity='Penetration')
            vulnerability_reports = self.metadata_storage.get_all_reports(entity='Vulnerability')
            enhancement_reports = self.metadata_storage.get_all_reports(entity='SecurityEnhancement')

            # Aggregate data
            aggregated_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'compliance_reports': compliance_reports,
                'penetration_reports': penetration_reports,
                'vulnerability_reports': vulnerability_reports,
                'security_enhancement_reports': enhancement_reports
            }

            # Save aggregated report
            report_filename = f"security_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            report_path = os.path.join(self.report_directory, report_filename)
            with open(report_path, 'w') as f:
                json.dump(aggregated_data, f, indent=4)

            logger.info(f"Comprehensive Security Report generated at {report_path}")

            # Optionally, send the report via email or integrate with a dashboard
        except Exception as e:
            logger.error(f"Failed to generate Security Report: {e}")
            self.send_alert(
                subject="Security Engine Alert: Security Report Generation Failure",
                message=f"An error occurred while generating the security report: {e}"
            )

    def run(self):
        """
        Starts the Security Engine, scheduling tasks and running the scheduler loop.
        """
        logger.info("Starting Security Engine.")
        self.schedule_tasks()

        # Schedule Security Enhancements after vulnerability scanning and compliance checking
        schedule.every().day.at("03:00").do(self.run_security_enhancements)
        schedule.every().day.at("05:00").do(self.generate_security_report)

        logger.info("Security Engine is running. Entering scheduler loop.")

        while True:
            schedule.run_pending()
            time.sleep(1)


if __name__ == "__main__":
    try:
        security_engine = SecurityEngine()
        security_engine.run()
    except KeyboardInterrupt:
        logger.info("Security Engine stopped manually.")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
