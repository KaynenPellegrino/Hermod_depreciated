# src/modules/performance_monitoring/scalability_tester.py

import asyncio
import os

import aiohttp
import logging
import time
import json
from typing import List, Dict, Any
from datetime import datetime
from src.utils.configuration_manager import ConfigurationManager
from src.modules.notifications.notification_manager import NotificationManager

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('logs/scalability_tester.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class ScalabilityTester:
    """
    Scalability Testing Tools
    Tests the application's ability to scale under increased load,
    simulating high-traffic scenarios to ensure the system can handle growth.
    """

    def __init__(self):
        """
        Initializes the ScalabilityTester with necessary configurations.
        """
        try:
            self.config_manager = ConfigurationManager()
            self.notification_manager = NotificationManager()
            self.load_tester_config()
            self.test_results: List[Dict[str, Any]] = []
            logger.info("ScalabilityTester initialized successfully.")
        except Exception as e:
            logger.exception(f"Failed to initialize ScalabilityTester: {e}")
            raise e

    def load_tester_config(self):
        """
        Loads tester configurations from the configuration manager or environment variables.
        """
        logger.info("Loading tester configurations.")
        try:
            self.tester_config = {
                'target_url': self.config_manager.get('TARGET_URL', 'http://localhost:8000'),
                'concurrency_levels': json.loads(self.config_manager.get('CONCURRENCY_LEVELS', '[10, 50, 100]')),
                'test_duration': int(self.config_manager.get('TEST_DURATION', 60)),
                'ramp_up_time': int(self.config_manager.get('RAMP_UP_TIME', 10)),
                'results_file': self.config_manager.get('RESULTS_FILE', 'reports/scalability_test_results.json'),
                'notification_recipients': self.config_manager.get('NOTIFICATION_RECIPIENTS', '').split(','),
                'request_timeout': int(self.config_manager.get('REQUEST_TIMEOUT', 30)),
            }
            logger.info(f"Tester configurations loaded: {self.tester_config}")
        except Exception as e:
            logger.error(f"Failed to load tester configurations: {e}")
            raise e

    async def perform_scalability_test(self):
        """
        Performs scalability testing by simulating concurrent requests to the target URL.
        """
        logger.info("Starting scalability testing.")
        try:
            for concurrency in self.tester_config['concurrency_levels']:
                logger.info(f"Testing with concurrency level: {concurrency}")
                test_result = await self.run_test(concurrency)
                self.test_results.append(test_result)
                logger.info(f"Test completed for concurrency level: {concurrency}")

            self.save_test_results()
            self.send_notification(
                subject="Scalability Testing Completed",
                message="The scalability testing has been completed successfully. Please review the test results."
            )
            logger.info("Scalability testing completed successfully.")
        except Exception as e:
            logger.error(f"Scalability testing failed: {e}")
            self.send_notification(
                subject="Scalability Testing Failed",
                message=f"Scalability testing failed with the following error:\n\n{e}"
            )
            raise e

    async def run_test(self, concurrency: int) -> Dict[str, Any]:
        """
        Runs a single test with the specified concurrency level.

        :param concurrency: Number of concurrent requests.
        :return: Test result data.
        """
        try:
            start_time = time.time()
            end_time = start_time + self.tester_config['test_duration']
            total_requests = 0
            successful_requests = 0
            failed_requests = 0
            response_times = []

            sem = asyncio.Semaphore(concurrency)

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.tester_config['request_timeout'])) as session:
                tasks = []

                while time.time() < end_time:
                    await sem.acquire()
                    task = asyncio.create_task(self.send_request(session, sem, response_times))
                    tasks.append(task)
                    total_requests += 1
                    # Control the rate of requests if needed
                    await asyncio.sleep(self.tester_config['ramp_up_time'] / (concurrency * self.tester_config['test_duration']))

                # Wait for all tasks to complete
                await asyncio.gather(*tasks)

            successful_requests = len(response_times)
            failed_requests = total_requests - successful_requests
            average_response_time = sum(response_times) / len(response_times) if response_times else 0

            test_result = {
                'concurrency': concurrency,
                'total_requests': total_requests,
                'successful_requests': successful_requests,
                'failed_requests': failed_requests,
                'average_response_time_ms': average_response_time * 1000,
                'timestamp': datetime.utcnow().isoformat()
            }

            return test_result
        except Exception as e:
            logger.error(f"Failed to run test at concurrency {concurrency}: {e}")
            raise e

    async def send_request(self, session: aiohttp.ClientSession, sem: asyncio.Semaphore, response_times: List[float]):
        """
        Sends a single HTTP request to the target URL.

        :param session: The aiohttp client session.
        :param sem: The semaphore controlling concurrency.
        :param response_times: List to record response times.
        """
        try:
            start_time = time.time()
            async with session.get(self.tester_config['target_url']) as response:
                await response.text()
                if response.status == 200:
                    elapsed_time = time.time() - start_time
                    response_times.append(elapsed_time)
                else:
                    logger.warning(f"Received non-200 response: {response.status}")
        except Exception as e:
            logger.warning(f"Request failed: {e}")
        finally:
            sem.release()

    def save_test_results(self):
        """
        Saves the scalability test results to a file.
        """
        logger.info("Saving test results.")
        try:
            results_file = self.tester_config['results_file']
            os.makedirs(os.path.dirname(results_file), exist_ok=True)
            with open(results_file, 'w') as f:
                json.dump(self.test_results, f, indent=4)
            logger.info(f"Test results saved to '{results_file}'.")
        except Exception as e:
            logger.error(f"Failed to save test results: {e}")
            raise e

    def send_notification(self, subject: str, message: str):
        """
        Sends a notification to the configured recipients.

        :param subject: Subject of the notification.
        :param message: Body of the notification.
        """
        try:
            recipients = self.tester_config['notification_recipients']
            if recipients:
                self.notification_manager.send_notification(
                    recipients=recipients,
                    subject=subject,
                    message=message
                )
                logger.info("Notification sent successfully.")
            else:
                logger.warning("No notification recipients configured.")
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")

# --------------------- Example Usage --------------------- #

def example_usage():
    """
    Demonstrates example usage of the ScalabilityTester class.
    """
    try:
        # Initialize ScalabilityTester
        tester = ScalabilityTester()

        # Perform the scalability test
        asyncio.run(tester.perform_scalability_test())

        # Access the test results
        print("Scalability Test Results:")
        print(json.dumps(tester.test_results, indent=4))

    except Exception as e:
        logger.exception(f"Error in example usage: {e}")

# --------------------- Main Execution --------------------- #

if __name__ == "__main__":
    # Run the scalability tester example
    example_usage()
