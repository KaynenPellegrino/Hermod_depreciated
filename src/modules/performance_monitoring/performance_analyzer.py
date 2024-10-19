# src/modules/performance_monitoring/performance_analyzer.py

import os
import logging
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
from datetime import datetime
from src.utils.configuration_manager import ConfigurationManager
from src.modules.notifications.notification_manager import NotificationManager

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('logs/performance_analyzer.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class PerformanceAnalyzer:
    """
    Performance Analysis
    Analyzes collected performance metrics to identify bottlenecks, trends, or anomalies.
    Helps in optimizing system performance and resource allocation.
    """

    def __init__(self):
        """
        Initializes the PerformanceAnalyzer with necessary configurations.
        """
        try:
            self.config_manager = ConfigurationManager()
            self.notification_manager = NotificationManager()
            self.load_analyzer_config()
            logger.info("PerformanceAnalyzer initialized successfully.")
        except Exception as e:
            logger.exception(f"Failed to initialize PerformanceAnalyzer: {e}")
            raise e

    def load_analyzer_config(self):
        """
        Loads analyzer configurations from the configuration manager or environment variables.
        """
        logger.info("Loading analyzer configurations.")
        try:
            self.analyzer_config = {
                'metrics_data_file': self.config_manager.get('METRICS_DATA_FILE', 'data/metrics_data.json'),
                'analysis_report_path': self.config_manager.get('ANALYSIS_REPORT_PATH', 'reports/performance_analysis_report.json'),
                'visualizations_path': self.config_manager.get('VISUALIZATIONS_PATH', 'reports/visualizations'),
                'notification_recipients': self.config_manager.get('NOTIFICATION_RECIPIENTS', '').split(','),
                'anomaly_thresholds': {
                    'cpu_usage_percent': float(self.config_manager.get('CPU_USAGE_THRESHOLD', 90.0)),
                    'memory_usage_percent': float(self.config_manager.get('MEMORY_USAGE_THRESHOLD', 90.0)),
                    'latency_ms': float(self.config_manager.get('LATENCY_THRESHOLD', 200.0)),
                    'throughput_rps': float(self.config_manager.get('THROUGHPUT_THRESHOLD', 50.0)),
                }
            }
            logger.info(f"Analyzer configurations loaded: {self.analyzer_config}")
        except Exception as e:
            logger.error(f"Failed to load analyzer configurations: {e}")
            raise e

    def perform_analysis(self):
        """
        Performs performance analysis on the collected metrics data.
        """
        logger.info("Starting performance analysis.")
        try:
            metrics_data = self.load_metrics_data()
            df = self.prepare_data(metrics_data)
            analysis_results = self.analyze_metrics(df)
            self.generate_analysis_report(analysis_results)
            self.generate_visualizations(df)
            self.send_notification(
                subject="Performance Analysis Completed",
                message="The performance analysis has been completed successfully. Please review the analysis report and visualizations."
            )
            logger.info("Performance analysis completed successfully.")
        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            self.send_notification(
                subject="Performance Analysis Failed",
                message=f"Performance analysis failed with the following error:\n\n{e}"
            )
            raise e

    def load_metrics_data(self) -> List[Dict[str, Any]]:
        """
        Loads the collected metrics data from the metrics data file.

        :return: List of metrics data records.
        """
        logger.info("Loading metrics data.")
        try:
            metrics_data_file = self.analyzer_config['metrics_data_file']
            if not os.path.exists(metrics_data_file):
                raise FileNotFoundError(f"Metrics data file not found at '{metrics_data_file}'.")
            with open(metrics_data_file, 'r') as f:
                metrics_data = json.load(f)
            logger.info("Metrics data loaded successfully.")
            return metrics_data
        except Exception as e:
            logger.error(f"Failed to load metrics data: {e}")
            raise e

    def prepare_data(self, metrics_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Converts metrics data into a pandas DataFrame and preprocesses it.

        :param metrics_data: List of metrics data records.
        :return: Preprocessed pandas DataFrame.
        """
        logger.info("Preparing data for analysis.")
        try:
            df = pd.json_normalize(metrics_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.sort_values('timestamp', inplace=True)
            df.reset_index(drop=True, inplace=True)
            logger.info("Data prepared successfully.")
            return df
        except Exception as e:
            logger.error(f"Failed to prepare data: {e}")
            raise e

    def analyze_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyzes the metrics DataFrame to identify bottlenecks, trends, or anomalies.

        :param df: Metrics DataFrame.
        :return: Analysis results with insights and recommendations.
        """
        logger.info("Analyzing metrics data.")
        try:
            analysis_results = {
                'anomalies': [],
                'trends': [],
                'recommendations': []
            }

            thresholds = self.analyzer_config['anomaly_thresholds']

            # Detect anomalies based on thresholds
            for metric, threshold in thresholds.items():
                metric_column = self.get_metric_column(df.columns, metric)
                if metric_column:
                    anomalies = df[df[metric_column] > threshold]
                    if not anomalies.empty:
                        analysis_results['anomalies'].append({
                            'metric': metric,
                            'threshold': threshold,
                            'anomalies': anomalies[['timestamp', metric_column]].to_dict(orient='records'),
                            'recommendation': f"Investigate high {metric} values exceeding {threshold}."
                        })

            # Identify trends (e.g., increasing CPU usage over time)
            # Placeholder for trend analysis logic
            # Implement actual statistical analysis or machine learning models for trend detection

            # Recommendations based on analysis
            if analysis_results['anomalies']:
                analysis_results['recommendations'].append("Review the identified anomalies and address potential issues.")
            else:
                analysis_results['recommendations'].append("System performance is within normal parameters.")

            logger.info("Metrics analysis completed.")
            return analysis_results
        except Exception as e:
            logger.error(f"Failed to analyze metrics: {e}")
            raise e

    def get_metric_column(self, columns: pd.Index, metric: str) -> str:
        """
        Finds the full column name for a given metric.

        :param columns: DataFrame columns.
        :param metric: Metric short name.
        :return: Full column name if found, else None.
        """
        for column in columns:
            if column.endswith(metric):
                return column
        return None

    def generate_analysis_report(self, analysis_results: Dict[str, Any]):
        """
        Generates the performance analysis report and saves it to a file.

        :param analysis_results: Results of the performance analysis.
        """
        logger.info("Generating analysis report.")
        try:
            report_path = self.analyzer_config['analysis_report_path']
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            with open(report_path, 'w') as report_file:
                json.dump(analysis_results, report_file, indent=4)
            logger.info(f"Analysis report saved to '{report_path}'.")
        except Exception as e:
            logger.error(f"Failed to generate analysis report: {e}")
            raise e

    def generate_visualizations(self, df: pd.DataFrame):
        """
        Generates visualizations from the metrics data and saves them to files.

        :param df: Metrics DataFrame.
        """
        logger.info("Generating visualizations.")
        try:
            visualizations_path = self.analyzer_config['visualizations_path']
            os.makedirs(visualizations_path, exist_ok=True)

            # Plot CPU usage over time
            plt.figure(figsize=(12, 6))
            sns.lineplot(x='timestamp', y='system_metrics.cpu_usage_percent', data=df)
            plt.title('CPU Usage Over Time')
            plt.xlabel('Timestamp')
            plt.ylabel('CPU Usage (%)')
            plt.tight_layout()
            cpu_usage_plot_path = os.path.join(visualizations_path, 'cpu_usage.png')
            plt.savefig(cpu_usage_plot_path)
            plt.close()
            logger.info(f"CPU usage plot saved to '{cpu_usage_plot_path}'.")

            # Plot Memory usage over time
            plt.figure(figsize=(12, 6))
            sns.lineplot(x='timestamp', y='system_metrics.memory_usage_percent', data=df)
            plt.title('Memory Usage Over Time')
            plt.xlabel('Timestamp')
            plt.ylabel('Memory Usage (%)')
            plt.tight_layout()
            memory_usage_plot_path = os.path.join(visualizations_path, 'memory_usage.png')
            plt.savefig(memory_usage_plot_path)
            plt.close()
            logger.info(f"Memory usage plot saved to '{memory_usage_plot_path}'.")

            # Plot Latency over time
            plt.figure(figsize=(12, 6))
            sns.lineplot(x='timestamp', y='application_metrics.latency_ms', data=df)
            plt.title('Latency Over Time')
            plt.xlabel('Timestamp')
            plt.ylabel('Latency (ms)')
            plt.tight_layout()
            latency_plot_path = os.path.join(visualizations_path, 'latency.png')
            plt.savefig(latency_plot_path)
            plt.close()
            logger.info(f"Latency plot saved to '{latency_plot_path}'.")

            # Plot Throughput over time
            plt.figure(figsize=(12, 6))
            sns.lineplot(x='timestamp', y='application_metrics.throughput_rps', data=df)
            plt.title('Throughput Over Time')
            plt.xlabel('Timestamp')
            plt.ylabel('Throughput (requests per second)')
            plt.tight_layout()
            throughput_plot_path = os.path.join(visualizations_path, 'throughput.png')
            plt.savefig(throughput_plot_path)
            plt.close()
            logger.info(f"Throughput plot saved to '{throughput_plot_path}'.")

            logger.info("Visualizations generated successfully.")
        except Exception as e:
            logger.error(f"Failed to generate visualizations: {e}")
            raise e

    def send_notification(self, subject: str, message: str):
        """
        Sends a notification to the configured recipients.

        :param subject: Subject of the notification.
        :param message: Body of the notification.
        """
        try:
            recipients = self.analyzer_config['notification_recipients']
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
    Demonstrates example usage of the PerformanceAnalyzer class.
    """
    try:
        # Initialize PerformanceAnalyzer
        analyzer = PerformanceAnalyzer()

        # Perform the performance analysis
        analyzer.perform_analysis()

    except Exception as e:
        logger.exception(f"Error in example usage: {e}")

# --------------------- Main Execution --------------------- #

if __name__ == "__main__":
    # Run the performance analyzer example
    example_usage()
