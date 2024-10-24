import logging
from typing import Dict, Any, List

import pandas as pd
import joblib
import os

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from src.utils.logger import get_logger
from src.utils.configuration_manager import ConfigurationManager
from src.modules.nlu.language_models.roberta_model import RoBERTAModel


class EntityRecognizer:
    """
    EntityRecognizer extracts relevant entities from user inputs.
    It leverages RoBERTa for improved entity recognition performance.
    """

    def __init__(self, project_id: str):
        """
        Initializes the EntityRecognizer with configurations specific to the project.

        Args:
            project_id (str): Unique identifier for the project.
        """
        self.logger = get_logger(__name__)
        self.config_manager = ConfigurationManager()
        self.config = self.config_manager.get_configuration(project_id)
        self.project_id = project_id

        # Initialize RoBERTAModel
        self.roberta_model = RoBERTAModel(project_id)

        # Path to save/load the entity recognition model
        self.model_path = self.config_manager.get_value(project_id, 'entity_recognizer.model_path') or 'data/models/nlu_models/entity_recognizer.joblib'

    def train_model(self, training_data: pd.DataFrame, test_size: float = 0.2) -> None:
        """
        Trains the entity recognition model using labeled data.

        Args:
            training_data (pd.DataFrame): DataFrame containing 'text' and 'entities' columns.
            test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.2.
        """
        self.logger.debug("Starting entity recognition model training.")

        try:
            X = training_data['text']
            y = training_data['entities']

            # Split the data into training and testing sets
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            # Generate embeddings for training data using RoBERTAModel
            self.logger.debug("Generating embeddings for training data.")
            X_train_embeddings = [self.roberta_model.generate_embeddings(text)['embeddings'] for text in X_train]
            X_test_embeddings = [self.roberta_model.generate_embeddings(text)['embeddings'] for text in X_test]

            # Train a multi-label classifier (One-vs-Rest with Logistic Regression)
            self.logger.debug("Training Logistic Regression classifier for entity recognition.")
            mlb = MultiLabelBinarizer()
            y_train_binarized = mlb.fit_transform(y_train)
            y_test_binarized = mlb.transform(y_test)

            classifier = OneVsRestClassifier(LogisticRegression(max_iter=1000))
            classifier.fit(X_train_embeddings, y_train_binarized)
            self.logger.info("Entity recognition model training completed successfully.")

            # Evaluate the model
            y_pred = classifier.predict(X_test_embeddings)
            report = classification_report(y_test_binarized, y_pred, target_names=mlb.classes_)
            self.logger.info(f"Entity Recognition Model Evaluation Report:\n{report}")

            # Save the trained classifier and the MultiLabelBinarizer
            self.classifier = classifier
            self.mlb = mlb
            self.save_model()

        except Exception as e:
            self.logger.error(f"Error during entity recognition model training: {e}", exc_info=True)

    def save_model(self) -> None:
        """
        Saves the trained entity recognition model to disk.
        """
        self.logger.debug(f"Saving entity recognition model to {self.model_path}")

        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump({
                'classifier': self.classifier,
                'mlb': self.mlb
            }, self.model_path)
            self.logger.info(f"Entity recognition model saved successfully at {self.model_path}")

        except Exception as e:
            self.logger.error(f"Error saving the entity recognition model: {e}", exc_info=True)

    def load_model(self) -> None:
        """
        Loads a pre-trained entity recognition model from disk.
        """
        self.logger.debug(f"Loading entity recognition model from {self.model_path}")

        try:
            model_data = joblib.load(self.model_path)
            self.classifier = model_data['classifier']
            self.mlb = model_data['mlb']
            self.logger.info(f"Entity recognition model loaded successfully from {self.model_path}")

        except Exception as e:
            self.logger.error(f"Error loading the entity recognition model: {e}", exc_info=True)
            raise

    def recognize_entities(self, text: str) -> Dict[str, Any]:
        """
        Extracts entities from the given user input text.

        Args:
            text (str): The user input text.

        Returns:
            Dict[str, Any]: A dictionary containing the status and the list of extracted entities.
        """
        self.logger.debug(f"Recognizing entities in text: {text}")

        try:
            embeddings = self.roberta_model.generate_embeddings(text)

            # Predict multi-label entities
            predicted = self.classifier.predict([embeddings['embeddings']])[0]
            entities = self.mlb.inverse_transform([predicted])[0]
            self.logger.info(f"Extracted entities: {entities}")

            return {
                "status": "success",
                "entities": list(entities)
            }

        except Exception as e:
            self.logger.error(f"Error in recognize_entities: {e}", exc_info=True)
            return {
                "status": "error",
                "message": "An error occurred while recognizing entities."
            }

if __name__ == "__main__":
    # Example usage
    project_id = "proj_12345"
    entity_recognizer = EntityRecognizer(project_id)

    # Example training data (ensure to provide actual data)
    training_data = pd.DataFrame({
        'text': [
            "Book a flight to New York",
            "What's the weather like today?",
            "Set an alarm for 7 AM",
            "Play some jazz music",
            "Show me the latest news",
            "Turn off the lights",
            "Send an email to John",
            "How do I make a cake?",
            "Schedule a meeting with the team",
            "Find Italian restaurants nearby"
        ],
        'entities': [
            ["New York"],
            ["today"],
            ["7 AM"],
            ["jazz music"],
            ["latest news"],
            ["lights"],
            ["John"],
            ["cake"],
            ["meeting", "team"],
            ["Italian restaurants", "nearby"]
        ]
    })

    # Train the model
    entity_recognizer.train_model(training_data)

    # Load the model
    entity_recognizer.load_model()

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
        recognition = entity_recognizer.recognize_entities(input_text)
        print(f"Input: {input_text}\nExtracted Entities: {recognition}\n")
