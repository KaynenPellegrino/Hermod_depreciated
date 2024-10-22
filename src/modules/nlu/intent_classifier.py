# src/modules/nlu/intent_classifier.py

import logging
from typing import Dict, Any

from src.utils.logger import get_logger
from src.utils.configuration_manager import ConfigurationManager
from src.modules.nlu.language_models.roberta_model import RoBERTAModel

class IntentClassifier:
    """
    Classifies user intents using a fine-tuned RoBERTa model.
    """

    def __init__(self, project_id: str):
        """
        Initializes the IntentClassifier with a fine-tuned RoBERTAModel.

        Args:
            project_id (str): Unique identifier for the project.
        """
        self.logger = get_logger(__name__)
        self.config_manager = ConfigurationManager()
        self.config = self.config_manager.get_configuration(project_id)
        self.project_id = project_id

        # Initialize RoBERTAModel
        self.roberta_model = RoBERTAModel(project_id)

    def classify_intent(self, text: str) -> Dict[str, Any]:
        """
        Classifies the intent of the given text.

        Args:
            text (str): Input text.

        Returns:
            Dict[str, Any]: Classification results.
        """
        self.logger.debug(f"Classifying intent for text: {text}")
        result = self.roberta_model.classify_text(text, return_all_scores=True)
        return result
