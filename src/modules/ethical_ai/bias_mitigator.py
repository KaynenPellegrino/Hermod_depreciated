# src/modules/ethical_ai/bias_mitigator.py

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from src.utils.configuration_manager import ConfigurationManager
from src.modules.notifications.notification_manager import NotificationManager

# Import scikit-learn and AIF360 libraries
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.inprocessing import PrejudiceRemover

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('logs/bias_mitigator.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class BiasMitigator:
    """
    Bias Detection and Mitigation
    Identifies and reduces biases in AI models and datasets, ensuring fairness and equity in AI outputs.
    Includes techniques for bias measurement, re-sampling, and algorithmic adjustments.
    """

    def __init__(self):
        """
        Initializes the BiasMitigator with necessary configurations.
        """
        try:
            self.config_manager = ConfigurationManager()
            self.notification_manager = NotificationManager()
            self.load_mitigator_config()
            self.dataset: Optional[StandardDataset] = None
            self.model = None
            logger.info("BiasMitigator initialized successfully.")
        except Exception as e:
            logger.exception(f"Failed to initialize BiasMitigator: {e}")
            raise e

    def load_mitigator_config(self):
        """
        Loads bias mitigator configurations from the configuration manager or environment variables.
        """
        logger.info("Loading bias mitigator configurations.")
        try:
            self.mitigator_config = {
                'data_file': self.config_manager.get('DATA_FILE', 'data/dataset.csv'),
                'protected_attribute': self.config_manager.get('PROTECTED_ATTRIBUTE', 'gender'),
                'privileged_groups': [{self.config_manager.get('PROTECTED_ATTRIBUTE', 'gender'): 1}],
                'unprivileged_groups': [{self.config_manager.get('PROTECTED_ATTRIBUTE', 'gender'): 0}],
                'target_column': self.config_manager.get('TARGET_COLUMN', 'outcome'),
                'favorable_label': float(self.config_manager.get('FAVORABLE_LABEL', 1)),
                'unfavorable_label': float(self.config_manager.get('UNFAVORABLE_LABEL', 0)),
                'notification_recipients': self.config_manager.get('NOTIFICATION_RECIPIENTS', '').split(','),
            }
            logger.info(f"Bias mitigator configurations loaded: {self.mitigator_config}")
        except Exception as e:
            logger.error(f"Failed to load bias mitigator configurations: {e}")
            raise e

    def load_dataset(self):
        """
        Loads the dataset and converts it to an AIF360 StandardDataset.
        """
        logger.info("Loading dataset.")
        try:
            df = pd.read_csv(self.mitigator_config['data_file'])
            self.dataset = StandardDataset(
                df,
                label_name=self.mitigator_config['target_column'],
                favorable_classes=[self.mitigator_config['favorable_label']],
                protected_attribute_names=[self.mitigator_config['protected_attribute']],
                privileged_classes=[[1]],
                instance_weights_name=None
            )
            logger.info("Dataset loaded and converted to StandardDataset.")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise e

    def detect_bias(self) -> Dict[str, Any]:
        """
        Detects bias in the dataset using fairness metrics.

        :return: Dictionary of bias metrics.
        """
        logger.info("Detecting bias in the dataset.")
        try:
            metric = BinaryLabelDatasetMetric(
                self.dataset,
                unprivileged_groups=self.mitigator_config['unprivileged_groups'],
                privileged_groups=self.mitigator_config['privileged_groups']
            )
            bias_metrics = {
                'mean_difference': metric.mean_difference(),
                'disparate_impact': metric.disparate_impact(),
                'statistical_parity_difference': metric.statistical_parity_difference(),
                'equal_opportunity_difference': metric.equal_opportunity_difference(),
            }
            logger.info(f"Bias metrics calculated: {bias_metrics}")
            return bias_metrics
        except Exception as e:
            logger.error(f"Failed to detect bias: {e}")
            raise e

    def mitigate_bias(self):
        """
        Applies bias mitigation techniques to the dataset and retrains the model.
        """
        logger.info("Applying bias mitigation techniques.")
        try:
            # Split the dataset
            train, test = self.dataset.split([0.7], shuffle=True)
            # Apply reweighing
            rw = Reweighing(
                unprivileged_groups=self.mitigator_config['unprivileged_groups'],
                privileged_groups=self.mitigator_config['privileged_groups']
            )
            rw.fit(train)
            train_transformed = rw.transform(train)

            # Train a Prejudice Remover model
            pr = PrejudiceRemover(
                sensitive_attr=self.mitigator_config['protected_attribute'],
                eta=1.0
            )
            pr.fit(train_transformed)

            # Predict on test set
            predictions = pr.predict(test)

            # Evaluate the model
            classified_metric = ClassificationMetric(
                test,
                predictions,
                unprivileged_groups=self.mitigator_config['unprivileged_groups'],
                privileged_groups=self.mitigator_config['privileged_groups']
            )
            self.model = pr
            logger.info("Bias mitigation applied and model retrained.")
            logger.info(f"Model accuracy: {classified_metric.accuracy()}")
        except Exception as e:
            logger.error(f"Failed to mitigate bias: {e}")
            raise e

    def evaluate_fairness(self) -> Dict[str, Any]:
        """
        Evaluates fairness metrics after bias mitigation.

        :return: Dictionary of fairness metrics.
        """
        logger.info("Evaluating fairness of the mitigated model.")
        try:
            test = self.dataset.copy(deepcopy=True)
            predictions = self.model.predict(test)
            metric = ClassificationMetric(
                test,
                predictions,
                unprivileged_groups=self.mitigator_config['unprivileged_groups'],
                privileged_groups=self.mitigator_config['privileged_groups']
            )
            fairness_metrics = {
                'accuracy': metric.accuracy(),
                'mean_difference': metric.mean_difference(),
                'disparate_impact': metric.disparate_impact(),
                'statistical_parity_difference': metric.statistical_parity_difference(),
                'equal_opportunity_difference': metric.equal_opportunity_difference(),
            }
            logger.info(f"Fairness metrics after mitigation: {fairness_metrics}")
            return fairness_metrics
        except Exception as e:
            logger.error(f"Failed to evaluate fairness: {e}")
            raise e

    def send_notification(self, subject: str, message: str):
        """
        Sends a notification to the configured recipients.

        :param subject: Subject of the notification.
        :param message: Body of the notification.
        """
        try:
            recipients = self.mitigator_config['notification_recipients']
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
    Demonstrates example usage of the BiasMitigator class.
    """
    try:
        # Initialize BiasMitigator
        mitigator = BiasMitigator()

        # Load dataset
        mitigator.load_dataset()

        # Detect bias
        bias_metrics = mitigator.detect_bias()
        print("Bias Metrics Before Mitigation:")
        print(bias_metrics)

        # Mitigate bias
        mitigator.mitigate_bias()

        # Evaluate fairness after mitigation
        fairness_metrics = mitigator.evaluate_fairness()
        print("Fairness Metrics After Mitigation:")
        print(fairness_metrics)

    except Exception as e:
        logger.exception(f"Error in example usage: {e}")


# --------------------- Main Execution --------------------- #

if __name__ == "__main__":
    # Run the bias mitigator example
    example_usage()
