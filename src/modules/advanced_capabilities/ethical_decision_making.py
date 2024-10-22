# src/modules/advanced_capabilities/ethical_decision_making.py

import logging
from typing import Dict, Any, List

import pandas as pd
import joblib
import os
import re

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from src.utils.logger import get_logger
from src.utils.configuration_manager import ConfigurationManager
from src.utils.helpers import format_code_snippet


class EthicalDecisionMaker:
    """
    EthicalDecisionMaker integrates ethical considerations into AI decision-making processes,
    ensuring outputs align with ethical guidelines.
    """

    def __init__(self, project_id: str):
        """
        Initializes the EthicalDecisionMaker with configurations specific to the project.

        Args:
            project_id (str): Unique identifier for the project.
        """
        # Initialize the logger
        self.logger = get_logger(__name__)

        # Initialize ConfigurationManager
        self.config_manager = ConfigurationManager()

        # Load configuration settings for the project
        self.config = self.config_manager.get_configuration(project_id)

        # Initialize spaCy model for text preprocessing
        self.model_name = self.config.get('ethical_decision_maker.spacy_model', 'en_core_web_sm')
        self.logger.debug(f"Loading spaCy model: {self.model_name}")

        try:
            self.nlp = spacy.load(self.model_name, disable=['parser', 'ner'])
            self.logger.info(f"spaCy model '{self.model_name}' loaded successfully.")
        except Exception as e:
            self.logger.error(f"Failed to load spaCy model '{self.model_name}': {e}", exc_info=True)
            raise

        # Initialize the machine learning pipeline
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', LogisticRegression(max_iter=1000)),
        ])

        # Path to save/load the trained model
        self.model_path = self.config.get('model_path', 'models/ethical_decision_maker.joblib')

    def train_model(self, training_data: pd.DataFrame, test_size: float = 0.2) -> None:
        """
        Trains the ethical decision-making model using labeled data.

        Args:
            training_data (pd.DataFrame): DataFrame containing 'decision_text' and 'ethics_label' columns.
            test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.2.
        """
        self.logger.debug("Starting ethical decision-making model training.")

        try:
            X = training_data['decision_text']
            y = training_data['ethics_label']

            # Split the data into training and testing sets
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            # Fit the pipeline on the training data
            self.pipeline.fit(X_train, y_train)
            self.logger.info("Ethical decision-making model training completed successfully.")

            # Evaluate the model
            y_pred = self.pipeline.predict(X_test)
            report = classification_report(y_test, y_pred)
            self.logger.info(f"Ethical Decision-Making Model Evaluation Report:\n{report}")

        except Exception as e:
            self.logger.error(f"Error during ethical decision-making model training: {e}", exc_info=True)

    def save_model(self) -> None:
        """
        Saves the trained ethical decision-making model to disk.
        """
        self.logger.debug(f"Saving ethical decision-making model to {self.model_path}")

        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

            # Save the pipeline
            joblib.dump(self.pipeline, self.model_path)
            self.logger.info(f"Ethical decision-making model saved successfully at {self.model_path}")

        except Exception as e:
            self.logger.error(f"Error saving the ethical decision-making model: {e}", exc_info=True)

    def load_model(self) -> None:
        """
        Loads a pre-trained ethical decision-making model from disk.
        """
        self.logger.debug(f"Loading ethical decision-making model from {self.model_path}")

        try:
            self.pipeline = joblib.load(self.model_path)
            self.logger.info(f"Ethical decision-making model loaded successfully from {self.model_path}")

        except Exception as e:
            self.logger.error(f"Error loading the ethical decision-making model: {e}", exc_info=True)
            raise

    def assess_ethics(self, decision_text: str) -> Dict[str, Any]:
        """
        Assesses the ethical compliance of the given decision text.

        Args:
            decision_text (str): The decision text to assess.

        Returns:
            Dict[str, Any]: A dictionary containing the status and the ethics assessment result.
        """
        self.logger.debug(f"Assessing ethics for decision: {decision_text}")

        try:
            preprocessed_text = self.preprocess_text(decision_text)
            ethics_label = self.pipeline.predict([preprocessed_text])[0]
            self.logger.info(f"Ethics Assessment Result: {ethics_label}")

            return {
                "status": "success",
                "ethics_label": ethics_label
            }

        except Exception as e:
            self.logger.error(f"Error in assess_ethics: {e}", exc_info=True)
            return {
                "status": "error",
                "message": "An error occurred while assessing ethics."
            }

    def preprocess_text(self, text: str) -> str:
        """
        Preprocesses the input text using spaCy (tokenization, lemmatization, etc.).

        Args:
            text (str): The text to preprocess.

        Returns:
            str: The preprocessed text.
        """
        self.logger.debug(f"Preprocessing text for ethical assessment: {text}")

        try:
            doc = self.nlp(text.lower())
            tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
            preprocessed_text = ' '.join(tokens)
            self.logger.debug(f"Preprocessed text: {preprocessed_text}")
            return preprocessed_text

        except Exception as e:
            self.logger.error(f"Error in preprocess_text: {e}", exc_info=True)
            return text


if __name__ == "__main__":
    # Example usage
    import pandas as pd

    # Initialize the EthicalDecisionMaker with a project_id
    project_id = "proj_12345"
    ethical_maker = EthicalDecisionMaker(project_id)

    # Example training data
    data = {
        'decision_text': [
            "We should collect user data without explicit consent.",
            "Implementing end-to-end encryption to protect user privacy.",
            "Discriminating against users based on their location.",
            "Ensuring transparency in our data usage policies.",
            "Limiting access to user data to authorized personnel only."
        ],
        'ethics_label': [
            "Unethical",
            "Ethical",
            "Unethical",
            "Ethical",
            "Ethical"
        ]
    }

    training_data = pd.DataFrame(data)

    # Train the model
    ethical_maker.train_model(training_data)

    # Save the trained model
    ethical_maker.save_model()

    # Load the model (for demonstration)
    ethical_maker.load_model()

    # Assess ethics in new decision texts
    decision_texts = [
        "We will store user passwords in plain text for easy access.",
        "Using data anonymization techniques to protect user identities.",
        "Restricting certain users from accessing premium features without valid reasons."
    ]

    for decision in decision_texts:
        assessment = ethical_maker.assess_ethics(decision)
        print(f"Decision: {decision}\nEthics Assessment: {assessment}\n")
