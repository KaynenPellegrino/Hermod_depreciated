# src/nlu/nlu_engine.py

import logging
from typing import Dict, Any, List

from src.utils.logger import get_logger
from src.utils.configuration_manager import ConfigurationManager

from intent_classifier import IntentClassifier
from entity_recognizer import EntityRecognizer
from modules.advanced_capabilities.emotion_recognizer import EmotionRecognizer
from modules.advanced_capabilities.ethical_decision_making import EthicalDecisionMaker
from modules.advanced_capabilities.explainable_ai import ExplainableAI


class ContextManager:
    """
    Manages conversation context to maintain state across user interactions.
    """
    def __init__(self):
        self.context = {}
        self.logger = get_logger(__name__)
        self.logger.debug("ContextManager initialized with empty context.")

    def update_context(self, key: str, value: Any):
        self.context[key] = value
        self.logger.debug(f"Context updated: {key} = {value}")

    def get_context(self, key: str) -> Any:
        value = self.context.get(key, None)
        self.logger.debug(f"Retrieved context: {key} = {value}")
        return value

    def clear_context(self):
        self.context = {}
        self.logger.debug("Context cleared.")


class NLUEngine:
    """
    Natural Language Understanding Engine orchestrates NLU tasks,
    integrating intent classification, entity recognition, emotion recognition,
    ethical decision making, explainable AI, and context management
    to interpret user inputs comprehensively.
    """

    def __init__(self, project_id: str):
        """
        Initializes the NLUEngine with configurations specific to the project.

        Args:
            project_id (str): Unique identifier for the project.
        """
        self.logger = get_logger(__name__)
        self.config_manager = ConfigurationManager()
        self.config = self.config_manager.get_configuration(project_id)
        self.project_id = project_id

        self.logger.debug(f"Initializing NLUEngine for project_id='{project_id}'.")

        # Initialize ContextManager
        self.context_manager = ContextManager()

        # Initialize IntentClassifier
        self.intent_classifier = IntentClassifier(project_id)

        # Initialize EntityRecognizer
        self.entity_recognizer = EntityRecognizer(project_id)

        # Initialize EmotionRecognizer
        self.emotion_recognizer = EmotionRecognizer(project_id)

        # Initialize EthicalDecisionMaker
        self.ethical_decision_maker = EthicalDecisionMaker(project_id)

        # Initialize ExplainableAI
        self.explainable_ai = ExplainableAI(project_id)

    def process_input(self, text: str) -> Dict[str, Any]:
        """
        Processes user input by performing intent classification, entity recognition,
        emotion recognition, ethical assessment, and generating explanations.

        Args:
            text (str): User input text.

        Returns:
            Dict[str, Any]: A dictionary containing the intent, entities, emotions,
                            ethical assessment, explanations, and updated context.
        """
        self.logger.info(f"Processing input: {text}")

        result = {}

        # Intent Classification
        intent_result = self.intent_classifier.predict_intent(text)
        if intent_result['status'] == 'success':
            intent = intent_result['intent']
            result['intent'] = intent
            self.logger.debug(f"Identified intent: {intent}")
            # Update context based on intent
            self.context_manager.update_context('last_intent', intent)
        else:
            self.logger.error("Intent classification failed.")
            result['intent'] = None

        # Entity Recognition
        entities_result = self.entity_recognizer.recognize_entities(text)
        if entities_result['status'] == 'success':
            entities = entities_result['entities']
            result['entities'] = entities
            self.logger.debug(f"Extracted entities: {entities}")
            # Update context with entities
            self.context_manager.update_context('entities', entities)
        else:
            self.logger.error("Entity recognition failed.")
            result['entities'] = []

        # Emotion Recognition
        emotions_result = self.emotion_recognizer.recognize_emotions(text)
        if emotions_result['status'] == 'success':
            emotions = emotions_result['emotion']
            result['emotions'] = emotions
            self.logger.debug(f"Identified emotions: {emotions}")
            # Update context with emotions
            self.context_manager.update_context('emotions', emotions)
        else:
            self.logger.error("Emotion recognition failed.")
            result['emotions'] = None

        # Ethical Decision Making
        ethics_result = self.ethical_decision_maker.assess_ethics(text)
        if ethics_result['status'] == 'success':
            ethics_label = ethics_result['ethics_label']
            result['ethics'] = ethics_label
            self.logger.debug(f"Ethics assessment: {ethics_label}")
            # Update context with ethics
            self.context_manager.update_context('ethics', ethics_label)
        else:
            self.logger.error("Ethical decision making failed.")
            result['ethics'] = None

        # Explainable AI - Generate explanations for intent and entities
        explanations = {}
        if intent_result['status'] == 'success' and result['intent']:
            explanation_intent = self.explainable_ai.generate_explanation(text, component='intent')
            explanations['intent_explanation'] = explanation_intent
            self.logger.debug(f"Intent explanation: {explanation_intent}")

        if entities_result['status'] == 'success' and result['entities']:
            explanation_entities = self.explainable_ai.generate_explanation(text, component='entities')
            explanations['entities_explanation'] = explanation_entities
            self.logger.debug(f"Entities explanation: {explanation_entities}")

        result['explanations'] = explanations

        # Context-Based Logic (Optional)
        # Example: Handling follow-up questions based on previous intent
        # This can be expanded as per project requirements

        return result

    def get_context(self) -> Dict[str, Any]:
        """
        Retrieves the current conversation context.

        Returns:
            Dict[str, Any]: The current context.
        """
        context = self.context_manager.context
        self.logger.debug(f"Current context: {context}")
        return context

    def clear_context(self):
        """
        Clears the conversation context.
        """
        self.context_manager.clear_context()
        self.logger.info("Conversation context cleared.")


if __name__ == "__main__":
    # Example usage
    project_id = "proj_12345"
    nlu_engine = NLUEngine(project_id)

    user_inputs = [
        "Book a flight to Paris",
        "I need to travel next week",
        "What's the weather in Paris?",
        "Set an alarm for 6 AM",
        "Play some relaxing music",
        "I feel so happy today!",
        "This decision might not be ethical.",
        "Can you explain why you made that recommendation?",
        "Cancel my reservation",
        "Send an email to John about the meeting."
    ]

    for input_text in user_inputs:
        nlu_result = nlu_engine.process_input(input_text)
        print(f"Input: {input_text}\nNLU Result: {nlu_result}\n")
