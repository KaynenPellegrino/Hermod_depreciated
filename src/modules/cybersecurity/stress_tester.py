# cybersecurity/stress_tester.py

import json
import logging
import os
import subprocess
import threading
import time
from datetime import datetime
from typing import Any, Dict, List

from dotenv import load_dotenv

# Import MetadataStorage from data_management module
from src.modules.data_management.staging import MetadataStorage

# Import other necessary cybersecurity modules if needed

# Load environment variables from .env file
load_dotenv()

# Configure logging with RotatingFileHandler to prevent log files from becoming too large
from logging.handlers import RotatingFileHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Ensure the logs directory exists
os.makedirs('logs', exist_ok=True)

# Create a rotating file handler
handler = RotatingFileHandler('logs/stress_tester.log', maxBytes=10**6, backupCount=5)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)


class StressTestReport:
    """
    Represents a stress testing report detailing simulated loads and their outcomes.
    """

    def __init__(self, target: str):
        """
        Initializes the StressTestReport.

        :param target: The target of the stress test (e.g., URL, IP address)
        """
        self.target = target
        self.timestamp = datetime.utcnow()
        self.tests_conducted: List[Dict[str, Any]] = []
        self.report_path = ""  # Initialize report_path

    def add_test(self, test: Dict[str, Any]):
        """
        Adds a conducted stress test to the report.

        :param test: Dictionary containing test details and outcomes
        """
        self.tests_conducted.append(test)
        logger.debug(f"Added test: {test}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the stress test report to a dictionary.

        :return: Dictionary representation of the report
        """
        return {
            'target': self.target,
            'timestamp': self.timestamp.isoformat(),
            'tests_conducted': self.tests_conducted
        }

    def save_report(self, directory: str = 'stress_test_reports') -> str:
        """
        Saves the stress test report as a JSON file.

        :param directory: Directory where the report will be saved
        :return: Path to the saved report
        """
        os.makedirs(directory, exist_ok=True)
        report_filename = f"{self.target}_stress_test_report_{self.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        report_path = os.path.join(directory, report_filename)
        try:
            with open(report_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=4)
            logger.info(f"Stress test report saved at {report_path}")
            self.report_path = report_path  # Update report_path
            return report_path
        except Exception as e:
            logger.error(f"Failed to save stress test report: {e}")
            raise e


class StressTester:
    """
    Simulates high-traffic and resource-intensive scenarios to test Hermod’s performance under extreme conditions.
    """

    def __init__(self):
        """
        Initializes the StressTester with necessary configurations.
        """
        # Configuration parameters
        self.targets = self.load_targets()
        self.metadata_storage = MetadataStorage()
        logger.info("StressTester initialized successfully.")

    def load_targets(self) -> List[Dict[str, Any]]:
        """
        Loads stress testing targets from environment variables or configuration files.

        :return: List of target dictionaries
        """
        # Example: Load targets from a JSON file specified in environment variables
        targets_file = os.getenv('STRESS_TESTER_TARGETS_FILE', 'stress_test_targets.json')
        if not os.path.exists(targets_file):
            logger.error(f"Targets file '{targets_file}' does not exist.")
            return []

        try:
            with open(targets_file, 'r') as f:
                targets = json.load(f).get('targets', [])
            logger.info(f"Loaded {len(targets)} stress testing targets from '{targets_file}'.")
            return targets
        except Exception as e:
            logger.error(f"Failed to load targets from '{targets_file}': {e}")
            return []

    def perform_stress_test(self, target: Dict[str, Any]):
        """
        Performs stress testing on a single target.

        :param target: Dictionary containing target details
        """
        target_url = target.get('url')
        test_types = target.get('test_types', [])
        logger.info(f"Starting stress test for target '{target_url}' with tests: {test_types}")

        report = StressTestReport(target=target_url)

        for test in test_types:
            test_type = test.get('type')
            if test_type == 'load_testing':
                self.run_load_test(target_url, report)
            elif test_type == 'resource_intensive_operations':
                self.run_resource_intensive_test(target_url, report)
            elif test_type == 'concurrent_users':
                self.run_concurrent_user_test(target_url, report)
            else:
                logger.warning(f"Unknown test type '{test_type}' for target '{target_url}'. Skipping.")

        # Save report
        report_path = report.save_report()

        # Save metadata about the stress test report
        metadata = {
            'target': target_url,
            'report_path': report_path,
            'tested_at': report.timestamp.isoformat()
        }
        self.metadata_storage.save_metadata(metadata, storage_type='sql')  # Choose 'sql' or 'mongodb' as per your setup

    def run_load_test(self, target_url: str, report: StressTestReport):
        """
        Runs a load test using Locust to simulate high-traffic scenarios.

        :param target_url: URL of the target to test
        :param report: StressTestReport instance to update
        """
        logger.info(f"Running load test on '{target_url}' using Locust")
        try:
            # Define Locust options
            users = 100  # Number of concurrent users
            spawn_rate = 10  # Users spawned per second
            run_time = '5m'  # Duration of the test

            # Run Locust in headless mode
            cmd = [
                'locust',
                '--headless',
                '--users', str(users),
                '--spawn-rate', str(spawn_rate),
                '--host', target_url,
                '--run-time', run_time,
                '--csv', f'locust_load_test_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}'
            ]
            subprocess.run(cmd, check=True)
            logger.info(f"Load test on '{target_url}' completed successfully.")
            report.add_test({
                'test_type': 'load_testing',
                'tool': 'Locust',
                'description': f'Simulated {users} concurrent users at a spawn rate of {spawn_rate} users/sec for {run_time}.',
                'outcome': 'Completed successfully.'
            })
        except subprocess.CalledProcessError as e:
            logger.error(f"Load test on '{target_url}' failed: {e}")
            report.add_test({
                'test_type': 'load_testing',
                'tool': 'Locust',
                'description': f'Simulated load test failed with error: {e}',
                'outcome': 'Failed.'
            })

    def run_resource_intensive_test(self, target_url: str, report: StressTestReport):
        """
        Executes resource-intensive operations to test system resilience.

        :param target_url: URL of the target to test
        :param report: StressTestReport instance to update
        """
        logger.info(f"Running resource-intensive test on '{target_url}' using Apache JMeter")
        try:
            # Define JMeter options
            test_plan = os.getenv('JMETR_TEST_PLAN', 'jmeter_test_plans/resource_intensive_test.jmx')
            if not os.path.exists(test_plan):
                logger.error(f"JMeter test plan '{test_plan}' does not exist.")
                report.add_test({
                    'test_type': 'resource_intensive_operations',
                    'tool': 'Apache JMeter',
                    'description': f'Resource-intensive test failed. Test plan {test_plan} not found.',
                    'outcome': 'Failed.'
                })
                return

            # Run JMeter in non-GUI mode
            cmd = [
                'jmeter',
                '-n',  # Non-GUI mode
                '-t', test_plan,
                '-Jtarget_url=' + target_url,
                '-l', '/tmp/resource_intensive_test_result.jtl'
            ]
            subprocess.run(cmd, check=True)
            logger.info(f"Resource-intensive test on '{target_url}' completed successfully.")
            report.add_test({
                'test_type': 'resource_intensive_operations',
                'tool': 'Apache JMeter',
                'description': 'Executed resource-intensive operations using Apache JMeter.',
                'outcome': 'Completed successfully.'
            })
        except subprocess.CalledProcessError as e:
            logger.error(f"Resource-intensive test on '{target_url}' failed: {e}")
            report.add_test({
                'test_type': 'resource_intensive_operations',
                'tool': 'Apache JMeter',
                'description': f'Resource-intensive test failed with error: {e}',
                'outcome': 'Failed.'
            })

    def run_concurrent_user_test(self, target_url: str, report: StressTestReport):
        """
        Simulates a large number of concurrent users to test system performance.

        :param target_url: URL of the target to test
        :param report: StressTestReport instance to update
        """
        logger.info(f"Running concurrent user test on '{target_url}' using Artillery")
        try:
            # Define Artillery options
            config_file = os.getenv('ARTILLERY_CONFIG', 'artillery_tests/concurrent_users_test.yml')
            if not os.path.exists(config_file):
                logger.error(f"Artillery config file '{config_file}' does not exist.")
                report.add_test({
                    'test_type': 'concurrent_users',
                    'tool': 'Artillery',
                    'description': f'Concurrent user test failed. Config file {config_file} not found.',
                    'outcome': 'Failed.'
                })
                return

            # Run Artillery test
            cmd = [
                'artillery',
                'run',
                config_file
            ]
            subprocess.run(cmd, check=True)
            logger.info(f"Concurrent user test on '{target_url}' completed successfully.")
            report.add_test({
                'test_type': 'concurrent_users',
                'tool': 'Artillery',
                'description': 'Simulated concurrent users using Artillery.',
                'outcome': 'Completed successfully.'
            })
        except subprocess.CalledProcessError as e:
            logger.error(f"Concurrent user test on '{target_url}' failed: {e}")
            report.add_test({
                'test_type': 'concurrent_users',
                'tool': 'Artillery',
                'description': f'Concurrent user test failed with error: {e}',
                'outcome': 'Failed.'
            })

    def run_stress_tests(self):
        """
        Initiates stress testing on all configured targets concurrently.
        """
        logger.info("Starting stress testing on all targets.")
        threads = []
        for target in self.targets:
            thread = threading.Thread(target=self.perform_stress_test, args=(target,))
            thread.start()
            threads.append(thread)
            # Optional: Limit the number of concurrent threads
            time.sleep(1)  # Slight delay to manage system load

        for thread in threads:
            thread.join()
        logger.info("Stress testing completed for all targets.")

    def generate_stress_test_report(self):
        """
        Generates a comprehensive stress test report by aggregating data from all tests.
        """
        logger.info("Generating comprehensive Stress Test Report.")
        try:
            # Fetch reports from Metadata Storage
            stress_reports = self.metadata_storage.get_all_reports(entity='StressTest')

            # Aggregate data
            aggregated_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'stress_test_reports': stress_reports
            }

            # Save aggregated report
            report_filename = f"stress_test_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            report_path = os.path.join('stress_test_reports', report_filename)
            with open(report_path, 'w') as f:
                json.dump(aggregated_data, f, indent=4)

            logger.info(f"Comprehensive Stress Test Report generated at {report_path}")

            # Optionally, send the report via email or integrate with a dashboard
            # Example:
            # self.notification_manager.send_email(
            #     subject="Stress Test Report Generated",
            #     message=f"A comprehensive stress test report has been generated and saved at {report_path}."
            # )
        except Exception as e:
            logger.error(f"Failed to generate Stress Test Report: {e}")
            # Optionally, send an alert
            # Example:
            # self.notification_manager.send_email(
            #     subject="Stress Tester Alert: Report Generation Failure",
            #     message=f"An error occurred while generating the stress test report: {e}"
            # )

    def run(self):
        """
        Starts the Stress Tester, executing stress tests and generating reports.
        """
        logger.info("Starting Stress Tester.")
        self.run_stress_tests()
        self.generate_stress_test_report()
        logger.info("Stress Tester completed successfully.")


if __name__ == "__main__":
    try:
        stress_tester = StressTester()
        stress_tester.run()
    except KeyboardInterrupt:
        logger.info("Stress Tester stopped manually.")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
