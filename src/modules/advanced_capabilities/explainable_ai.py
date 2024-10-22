# src/modules/advanced_capabilities/explainable_ai.py

import logging
from typing import Dict, Any, Optional, List

import shap  # Ensure SHAP is installed
import joblib
import os

import spacy
from sklearn.pipeline import Pipeline
from win32comext.shell.demos.IActiveDesktop import component

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
        self.model_path_intent = self.config_manager.get_value(project_id, 'intent_classifier.model_path') or 'models/intent_classifier.joblib'
        self.model_path_entities = self.config_manager.get_value(project_id, 'entity_recognizer.model_path') or 'models/entity_recognizer.joblib'
        # Add paths for other models if necessary

        self.logger.debug(f"Loading models for explanations from {self.model_path_intent} and {self.model_path_entities}")

        try:
            self.model_intent = joblib.load(self.model_path_intent)
            self.logger.info(f"Intent model loaded successfully from {self.model_path_intent}")
        except Exception as e:
            self.logger.error(f"Failed to load intent model from '{self.model_path_intent}': {e}", exc_info=True)
            raise

        try:
            self.model_entities = joblib.load(self.model_path_entities)
            self.logger.info(f"Entity model loaded successfully from {self.model_path_entities}")
        except Exception as e:
            self.logger.error(f"Failed to load entity model from '{self.model_path_entities}': {e}", exc_info=True)
            raise

        # Initialize SHAP explainers for each model
        try:
            self.explainer_intent = shap.Explainer(self.model_intent.named_steps['clf'], self.model_intent.named_steps['tfidf'].transform)
            self.logger.info("SHAP Explainer for IntentClassifier initialized successfully.")
        except Exception as e:
            self.logger.error(f"Failed to initialize SHAP Explainer for IntentClassifier: {e}", exc_info=True)
            raise

        try:
            self.explainer_entities = shap.Explainer(self.model_entities.named_steps['clf'], self.model_entities.named_steps['tfidf'].transform)
            self.logger.info("SHAP Explainer for EntityRecognizer initialized successfully.")
        except Exception as e:
            self.logger.error(f"Failed to initialize SHAP Explainer for EntityRecognizer: {e}", exc_info=True)
            raise

    def generate_explanation(self, text: str, component: str) -> Dict[str, Any]:
        """
        Generates an explanation for the model's prediction on the given text.

        Args:
            text (str): The input text for which to generate an explanation.
            component (str): The component to explain ('intent' or 'entities').

        Returns:
            Dict[str, Any]: A dictionary containing the status and the explanation details.
        """
        self.logger.debug(f"Generating explanation for component: {component} with text: {text}")

        try:
            # Select the appropriate model and explainer
            if component == 'intent':
                model = self.model_intent
                explainer = self.explainer_intent
            elif component == 'entities':
                model = self.model_entities
                explainer = self.explainer_entities
            else:
                self.logger.error(f"Invalid component '{component}' for explanation.")
                return {
                    "status": "error",
                    "message": f"Invalid component '{component}' for explanation."
                }

            # Preprocess text using the same steps as the model
            preprocessed_text = self.preprocess_text(text, model)
            vectorized_text = model.named_steps['tfidf'].transform([preprocessed_text])

            # Generate SHAP values
            shap_values = explainer(vectorized_text)

            # Extract feature importance
            feature_importance = shap_values.values

            # Get the prediction
            prediction = model.predict([preprocessed_text])[0]

            # Get top features contributing to the prediction
            top_features = self.get_top_features(model, vectorized_text, shap_values, component)

            explanation = {
                "prediction": prediction,
                "top_features": top_features
            }

            self.logger.debug(f"Generated explanation for {component}: {explanation}")

            return {
                "status": "success",
                "explanation": explanation
            }

        except Exception as e:
            self.logger.error(f"Error in generate_explanation: {e}", exc_info=True)
            return {
                "status": "error",
                "message": "An error occurred while generating the explanation."
            }

    def get_top_features(self, model: Pipeline, vectorized_text, shap_values, component: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves the top N features contributing to the prediction.

        Args:
            model (Pipeline): The ML pipeline model.
            vectorized_text: The vectorized input text.
            shap_values: SHAP values for the input.
            component (str): The component ('intent' or 'entities').
            top_n (int, optional): Number of top features to retrieve. Defaults to 5.

        Returns:
            List[Dict[str, Any]]: List of top features with their contributions.
        """
        feature_names = model.named_steps['tfidf'].get_feature_names_out()
        shap_values = shap_values.values[0]  # Assuming single input
        feature_contributions = zip(feature_names, shap_values)
        sorted_features = sorted(feature_contributions, key=lambda x: abs(x[1]), reverse=True)
        top_features = [{"feature": name, "contribution": value} for name, value in sorted_features[:top_n]]
        return top_features

    def preprocess_text(self, text: str, model: Pipeline) -> str:
        """
        Preprocesses the input text using spaCy (tokenization, lemmatization, etc.).

        Args:
            text (str): The text to preprocess.
            model (Pipeline): The ML pipeline model to ensure consistent preprocessing.

        Returns:
            str: The preprocessed text.
        """
        self.logger.debug(f"Preprocessing text for {model}: {text}")

        try:
            # Assuming the model uses the same preprocessing as other modules
            # Extract the spaCy model used by the pipeline
            spacy_model_name = self.config_manager.get_value(self.project_id, f'{component}.spacy_model') or 'en_core_web_sm'
            nlp = spacy.load(spacy_model_name, disable=['parser', 'ner'])
            doc = nlp(text.lower())
            tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
            preprocessed_text = ' '.join(tokens)
            self.logger.debug(f"Preprocessed text: {preprocessed_text}")
            return preprocessed_text

        except Exception as e:
            self.logger.error(f"Error in preprocess_text: {e}", exc_info=True)
            return text

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
