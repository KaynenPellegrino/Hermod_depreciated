import logging
from typing import Dict, Any, Optional, List

import joblib
import os

from transformers import AutoTokenizer
import shap
import torch

from src.utils.logger import get_logger
from src.utils.configuration_manager import ConfigurationManager


class ExplainableAI:
    """
    ExplainableAI provides methods for interpreting AI decisions,
    offering transparency and explanations to users.
    """

    def __init__(self, project_id: str):
        """
        Initializes the ExplainableAI with configurations specific to the project.

        Args:
            project_id (str): Unique identifier for the project.
        """
        self.logger = get_logger(__name__)
        self.config_manager = ConfigurationManager()
        self.config = self.config_manager.get_configuration(project_id)
        self.project_id = project_id

        # Initialize paths to the models needing explanations
        self.model_path_intent = self.config_manager.get_value(
            project_id, 'intent_classifier.model_path') or 'models/intent_classifier.joblib'
        self.model_path_entities = self.config_manager.get_value(
            project_id, 'entity_recognizer.model_path') or 'models/entity_recognizer.joblib'

        # Initialize Hugging Face tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        self.logger.debug(f"Loading models for explanations from {self.model_path_intent} and {self.model_path_entities}")

        try:
            self.model_intent = joblib.load(self.model_path_intent)
            self.logger.info(f"Intent model loaded successfully from {self.model_path_intent}")
        except Exception as e:
            self.logger.error(f"Failed to load intent model: {e}", exc_info=True)

        try:
            self.model_entities = joblib.load(self.model_path_entities)
            self.logger.info(f"Entity model loaded successfully from {self.model_path_entities}")
        except Exception as e:
            self.logger.error(f"Failed to load entity model: {e}", exc_info=True)

        self.explainer_intent = shap.Explainer(self.model_intent)
        self.explainer_entities = shap.Explainer(self.model_entities)

    def generate_explanation(self, text: str, component: str) -> Dict[str, Any]:
        """
        Generates an explanation for the model's prediction on the given text.

        Args:
            text (str): The input text for which to generate an explanation.
            component (str): The component to explain ('intent' or 'entities').

        Returns:
            Dict[str, Any]: A dictionary containing the status and the explanation details.
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        shap_values = None
        model = None
        if component == 'intent':
            model = self.model_intent
            shap_values = self.explainer_intent(inputs["input_ids"])
        elif component == 'entities':
            model = self.model_entities
            shap_values = self.explainer_entities(inputs["input_ids"])

        top_features = self.get_top_features(model, shap_values, component)
        prediction = model.predict([text])[0]

        return {
            "status": "success",
            "explanation": {
                "prediction": prediction,
                "top_features": top_features
            }
        }

    def get_top_features(self, model, shap_values, component: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Get top features from the SHAP values contributing to predictions.

        Args:
            model: The ML model used.
            shap_values: SHAP values explaining the model's predictions.
            component: The model component.
            top_n: The number of top features to return.

        Returns:
            List of top features.
        """
        feature_importance = shap_values.values[0]  # Assuming single input
        sorted_features = sorted(zip(self.tokenizer.get_vocab().keys(), feature_importance), key=lambda x: abs(x[1]), reverse=True)
        return [{"feature": name, "contribution": value} for name, value in sorted_features[:top_n]]


if __name__ == "__main__":
    # Example usage
    project_id = "proj_12345"
    explainable_ai = ExplainableAI(project_id)

    # Example intents
    intents = [
        "Book a flight to London",
        "Set an alarm for 6 AM",
        "Play some relaxing music"
    ]

    for intent_text in intents:
        explanation = explainable_ai.generate_explanation(intent_text)
        print(f"Intent: {intent_text}\nExplanation: {explanation}\n")
