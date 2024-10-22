# src/nlu/emotion_recognizer.py

import logging
from typing import Dict, Any

import pandas as pd
import os

from src.utils.logger import get_logger
from src.utils.configuration_manager import ConfigurationManager
from src.modules.nlu.language_models.roberta_model import BERTModel


class EmotionRecognizer:
    """
    EmotionRecognizer identifies emotions conveyed in user inputs.
    It leverages BERT for improved emotion recognition performance.
    """

    def __init__(self, project_id: str):
        """
        Initializes the EmotionRecognizer with configurations specific to the project.

        Args:
            project_id (str): Unique identifier for the project.
        """
        self.logger = get_logger(__name__)
        self.config_manager = ConfigurationManager()
        self.config = self.config_manager.get_configuration(project_id)
        self.project_id = project_id

        # Initialize BERTModel
        self.bert_model = BERTModel(project_id)

        # Path to save/load the emotion recognition model (if using a separate classifier)
        self.model_path = self.config.get('emotion_recognizer.model_path', 'models/emotion_recognizer_proj_12345.joblib')

    def train_model(self, training_data: pd.DataFrame, test_size: float = 0.2) -> None:
        """
        Trains the emotion recognition model using labeled data.

        Args:
            training_data (pd.DataFrame): DataFrame containing 'text' and 'emotion' columns.
            test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.2.
        """
        self.logger.debug("Starting emotion recognition model training.")

        try:
            X = training_data['text']
            y = training_data['emotion']

            from sklearn.model_selection import train_test_split
            from sklearn.linear_model import LogisticRegression

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            # Generate embeddings for training data
            self.logger.debug("Generating embeddings for training data.")
            X_train_embeddings = [self.bert_model.generate_embeddings(text)['embeddings'] for text in X_train]
            X_test_embeddings = [self.bert_model.generate_embeddings(text)['embeddings'] for text in X_test]

            # Train a logistic regression classifier
            self.logger.debug("Training Logistic Regression classifier for emotion recognition.")
            classifier = LogisticRegression(max_iter=1000)
            classifier.fit(X_train_embeddings, y_train)
            self.logger.info("Emotion recognition model training completed successfully.")

            # Evaluate the model
            y_pred = classifier.predict(X_test_embeddings)
            from sklearn.metrics import classification_report
            report = classification_report(y_test, y_pred)
            self.logger.info(f"Emotion Recognition Model Evaluation Report:\n{report}")

            # Save the trained classifier
            self.classifier = classifier
            self.save_model()

        except Exception as e:
            self.logger.error(f"Error during emotion recognition model training: {e}", exc_info=True)

    def save_model(self) -> None:
        """
        Saves the trained emotion recognition model to disk.
        """
        self.logger.debug(f"Saving emotion recognition model to {self.model_path}")

        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            import joblib
            joblib.dump(self.classifier, self.model_path)
            self.logger.info(f"Emotion recognition model saved successfully at {self.model_path}")

        except Exception as e:
            self.logger.error(f"Error saving the emotion recognition model: {e}", exc_info=True)

    def load_model(self) -> None:
        """
        Loads a pre-trained emotion recognition model from disk.
        """
        self.logger.debug(f"Loading emotion recognition model from {self.model_path}")

        try:
            import joblib
            self.classifier = joblib.load(self.model_path)
            self.logger.info(f"Emotion recognition model loaded successfully from {self.model_path}")

        except Exception as e:
            self.logger.error(f"Error loading the emotion recognition model: {e}", exc_info=True)
            raise

    def recognize_emotions(self, text: str) -> Dict[str, Any]:
        """
        Identifies emotions present in the given user input text.

        Args:
            text (str): The user input text.

        Returns:
            Dict[str, Any]: A dictionary containing the status and the identified emotion.
        """
        self.logger.debug(f"Recognizing emotions in text: {text}")

        try:
            embeddings = self.bert_model.generate_embeddings(text)
            if embeddings['status'] != 'success':
                return {
                    "status": "error",
                    "message": embeddings.get("message", "Failed to generate embeddings.")
                }

            emotion = self.classifier.predict([embeddings['embeddings']])[0]
            self.logger.info(f"Identified emotion: {emotion}")

            return {
                "status": "success",
                "emotion": emotion
            }

        except Exception as e:
            self.logger.error(f"Error in recognize_emotions: {e}", exc_info=True)
            return {
                "status": "error",
                "message": "An error occurred while recognizing emotions."
            }


if __name__ == "__main__":
    # Example usage
    import pandas as pd

    # Initialize the EmotionRecognizer with a project_id
    project_id = "proj_12345"
    emotion_recognizer = EmotionRecognizer(project_id)

    # Example training data
    data = {
        'text': [
            "I am so happy today!",
            "This is the worst day ever.",
            "I'm feeling quite neutral about this.",
            "Absolutely fantastic performance!",
            "I hate when things go wrong.",
            "I'm excited for the upcoming trip.",
            "This makes me so sad.",
            "What a lovely surprise!",
            "I'm frustrated with the delays.",
            "Nothing much happening here."
        ],
        'emotion': [
            "Happy",
            "Angry",
            "Neutral",
            "Happy",
            "Angry",
            "Happy",
            "Sad",
            "Happy",
            "Angry",
            "Neutral"
        ]
    }

    training_data = pd.DataFrame(data)

    # Train the model
    emotion_recognizer.train_model(training_data)

    # Save the trained model
    emotion_recognizer.save_model()

    # Load the model (for demonstration)
    emotion_recognizer.load_model()

    # Recognize emotions in new user inputs
    user_inputs = [
        "I just got a promotion!",
        "I'm not sure how I feel about this.",
        "It's a gloomy day today.",
        "I love spending time with friends.",
        "This is so annoying."
    ]

    for input_text in user_inputs:
        recognition = emotion_recognizer.recognize_emotions(input_text)
        print(f"Input: {input_text}\nIdentified Emotion: {recognition}\n")
