#!/usr/bin/env python3
"""
feedback_loop_manager.py

Function: Feedback Loop Orchestration
Purpose: Manages the entire feedback loop process, coordinating between data collection, analysis, and action execution.
         Ensures that insights lead to actionable improvements.
"""

import os
import sys
import yaml
import logging
from datetime import datetime
import subprocess
import time


# ----------------------------
# Configuration and Logging
# ----------------------------

def load_config(config_path='config.yaml'):
    """
    Load configuration from a YAML file.
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logging.error(f"Failed to load configuration file: {e}")
        sys.exit(1)


def setup_logging(log_dir='logs'):
    """
    Setup logging configuration.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'feedback_loop_manager_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


# ----------------------------
# Feedback Loop Manager Class
# ----------------------------

class FeedbackLoopManager:
    """
    Manages the feedback loop by coordinating data collection, analysis, and action execution.
    """

    def __init__(self, config):
        self.config = config
        self.collector_script = self.config.get('collector_script', 'feedback_collector.py')
        self.analyzer_script = self.config.get('analyzer_script', 'feedback_analyzer.py')
        self.action_scripts = self.config.get('action_scripts', [])
        self.virtual_env = self.config.get('virtual_env', 'venv')
        self.log_dir = self.config.get('log_dir', 'logs')
        self.data_dir = self.config.get('data_dir', 'data')
        self.report_dir = self.config.get('report_dir', 'reports')
        self.visualization_dir = self.config.get('visualization_dir', 'visualizations')
        self.model_dir = self.config.get('model_dir', 'models')
        self.retry_limit = self.config.get('retry_limit', 3)
        self.retry_delay = self.config.get('retry_delay', 5)  # in seconds

    def run_script(self, script_path):
        """
        Execute a Python script within the virtual environment.
        """
        try:
            logging.info(f"Executing script: {script_path}")
            result = subprocess.run([os.path.join(self.virtual_env, 'bin', 'python'), script_path],
                                    capture_output=True, text=True)
            if result.returncode != 0:
                logging.error(f"Script {script_path} failed with return code {result.returncode}")
                logging.error(f"Error Output: {result.stderr}")
                return False
            else:
                logging.info(f"Script {script_path} executed successfully.")
                logging.debug(f"Output: {result.stdout}")
                return True
        except Exception as e:
            logging.error(f"Exception occurred while executing {script_path}: {e}")
            return False

    def run_collector(self):
        """
        Run the feedback collector script.
        """
        script_path = os.path.join('src', 'modules', 'feedback_loop', self.collector_script)
        success = False
        for attempt in range(1, self.retry_limit + 1):
            logging.info(f"Attempt {attempt} to run collector script.")
            success = self.run_script(script_path)
            if success:
                break
            else:
                logging.warning(f"Retrying collector script in {self.retry_delay} seconds...")
                time.sleep(self.retry_delay)
        if not success:
            logging.error("Failed to execute collector script after multiple attempts.")
            return False
        return True

    def run_analyzer(self):
        """
        Run the feedback analyzer script.
        """
        script_path = os.path.join('src', 'modules', 'feedback_loop', self.analyzer_script)
        success = False
        for attempt in range(1, self.retry_limit + 1):
            logging.info(f"Attempt {attempt} to run analyzer script.")
            success = self.run_script(script_path)
            if success:
                break
            else:
                logging.warning(f"Retrying analyzer script in {self.retry_delay} seconds...")
                time.sleep(self.retry_delay)
        if not success:
            logging.error("Failed to execute analyzer script after multiple attempts.")
            return False
        return True

    def run_actions(self, insights):
        """
        Execute action scripts based on the insights.
        """
        if not self.action_scripts:
            logging.info("No action scripts configured to run.")
            return True

        for script in self.action_scripts:
            script_path = os.path.join('src', 'modules', 'feedback_loop', script)
            success = False
            for attempt in range(1, self.retry_limit + 1):
                logging.info(f"Attempt {attempt} to run action script: {script}")
                success = self.run_script(script_path)
                if success:
                    break
                else:
                    logging.warning(f"Retrying action script {script} in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
            if not success:
                logging.error(f"Failed to execute action script {script} after multiple attempts.")
                # Depending on requirements, decide whether to continue or halt
                continue
        return True

    def generate_insights_summary(self):
        """
        Generate a summary of insights from the analyzer reports.
        Placeholder function - customize based on actual analyzer outputs.
        """
        try:
            report_files = [f for f in os.listdir(self.report_dir) if f.startswith('feedback_analysis_report')]
            if not report_files:
                logging.warning("No analysis reports found to generate insights summary.")
                return {}

            latest_report = sorted(report_files)[-1]
            report_path = os.path.join(self.report_dir, latest_report)
            with open(report_path, 'r') as file:
                content = file.read()

            # Placeholder: Parse the report content to extract insights
            # This needs to be customized based on report format
            insights = {
                'total_anomalies': content.count('Total Anomalies Detected'),
                # Add more parsed insights as needed
            }
            logging.info(f"Generated insights summary: {insights}")
            return insights
        except Exception as e:
            logging.error(f"Error generating insights summary: {e}")
            return {}

    def run_feedback_loop(self):
        """
        Execute the entire feedback loop: collect, analyze, act.
        """
        logging.info("Starting the feedback loop process.")

        # Step 1: Data Collection
        if not self.run_collector():
            logging.error("Data collection step failed. Aborting feedback loop.")
            return False

        # Step 2: Data Analysis
        if not self.run_analyzer():
            logging.error("Data analysis step failed. Aborting feedback loop.")
            return False

        # Step 3: Generate Insights
        insights = self.generate_insights_summary()

        # Step 4: Execute Actions
        if not self.run_actions(insights):
            logging.error("Action execution step encountered errors.")
            # Depending on requirements, decide to continue or abort
            # Here, we continue
        logging.info("Feedback loop process completed successfully.")
        return True


# ----------------------------
# Main Function
# ----------------------------

def main():
    # Load configuration
    config = load_config()

    # Setup logging
    setup_logging(config.get('log_dir', 'logs'))

    logging.info("Initializing Feedback Loop Manager.")

    # Initialize FeedbackLoopManager
    manager = FeedbackLoopManager(config)

    # Run the feedback loop
    success = manager.run_feedback_loop()

    if success:
        logging.info("Feedback loop executed successfully.")
    else:
        logging.error("Feedback loop encountered critical errors.")


if __name__ == "__main__":
    main()
