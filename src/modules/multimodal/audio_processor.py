# src/modules/multimodal/audio_processor.py

import os
import logging
import librosa
import numpy as np
import speech_recognition as sr
from typing import Any, Dict, Optional, List
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from utils.logger import get_logger
from utils.configuration_manager import ConfigurationManager


class AudioProcessor:
    """
    Handles processing of audio data, including speech recognition, audio feature extraction,
    and audio classification. Enables the system to interpret and analyze audio inputs.
    """

    def __init__(self, project_id: str):
        """
        Initializes the AudioProcessor with necessary configurations and tools.

        Args:
            project_id (str): Unique identifier for the project.
        """
        self.logger = get_logger(__name__)
        self.project_id = project_id
        self.config_manager = ConfigurationManager()
        self.config = self.config_manager.get_configuration(project_id)

        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()

        # Initialize classification model (placeholder: SVM classifier)
        self.classifier = make_pipeline(StandardScaler(), SVC(probability=True))
        self.model_trained = False

        self.logger.info(f"AudioProcessor initialized for project '{project_id}'.")

    # ----------------------------
    # Speech Recognition
    # ----------------------------

    def recognize_speech_from_audio(self, audio_file_path: str) -> Optional[str]:
        """
        Converts speech in an audio file to text.

        Args:
            audio_file_path (str): Path to the audio file.

        Returns:
            Optional[str]: Recognized text or None if failed.
        """
        try:
            with sr.AudioFile(audio_file_path) as source:
                audio = self.recognizer.record(source)
            text = self.recognizer.recognize_google(audio)
            self.logger.info(f"Recognized speech from '{audio_file_path}': {text}")
            return text
        except sr.UnknownValueError:
            self.logger.warning(f"Speech Recognition could not understand audio in '{audio_file_path}'.")
            return None
        except sr.RequestError as e:
            self.logger.error(f"Could not request results from Speech Recognition service; {e}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to recognize speech from '{audio_file_path}': {e}")
            return None

    # ----------------------------
    # Audio Feature Extraction
    # ----------------------------

    def extract_features(self, audio_file_path: str) -> Optional[np.ndarray]:
        """
        Extracts audio features from an audio file.

        Args:
            audio_file_path (str): Path to the audio file.

        Returns:
            Optional[np.ndarray]: Extracted feature vector or None if failed.
        """
        try:
            y, sr_ = librosa.load(audio_file_path, sr=None)
            mfccs = librosa.feature.mfcc(y=y, sr=sr_, n_mfcc=13)
            mfccs_scaled = np.mean(mfccs.T, axis=0)
            self.logger.debug(f"Extracted MFCCs from '{audio_file_path}': {mfccs_scaled}")
            return mfccs_scaled
        except Exception as e:
            self.logger.error(f"Failed to extract features from '{audio_file_path}': {e}")
            return None

    # ----------------------------
    # Audio Classification
    # ----------------------------

    def train_classifier(self, feature_matrix: np.ndarray, labels: List[str]) -> None:
        """
        Trains the audio classification model.

        Args:
            feature_matrix (np.ndarray): Feature matrix where each row corresponds to an audio sample.
            labels (List[str]): List of labels corresponding to each audio sample.
        """
        try:
            self.classifier.fit(feature_matrix, labels)
            self.model_trained = True
            self.logger.info("Audio classification model trained successfully.")
        except Exception as e:
            self.logger.error(f"Failed to train audio classification model: {e}")
            raise e

    def classify_audio(self, audio_file_path: str) -> Optional[Dict[str, Any]]:
        """
        Classifies an audio file based on extracted features.

        Args:
            audio_file_path (str): Path to the audio file.

        Returns:
            Optional[Dict[str, Any]]: Classification result with probabilities or None if failed.
        """
        if not self.model_trained:
            self.logger.warning("Audio classification model is not trained.")
            return None

        features = self.extract_features(audio_file_path)
        if features is None:
            return None

        try:
            prediction = self.classifier.predict([features])[0]
            probabilities = self.classifier.predict_proba([features])[0]
            result = {
                'prediction': prediction,
                'probabilities': dict(zip(self.classifier.named_steps['svc'].classes_, probabilities))
            }
            self.logger.info(f"Classified audio '{audio_file_path}': {result}")
            return result
        except Exception as e:
            self.logger.error(f"Failed to classify audio '{audio_file_path}': {e}")
            return None

    # ----------------------------
    # Example Usage and Test Cases
    # ----------------------------

    def run_sample_operations(self):
        """
        Demonstrates sample audio processing operations.
        """
        self.logger.info("Running sample audio processing operations.")

        # Sample audio file path
        sample_audio = 'sample_audio.wav'  # Replace with an actual audio file path

        # Speech Recognition
        recognized_text = self.recognize_speech_from_audio(sample_audio)
        if recognized_text:
            self.logger.info(f"Recognized Text: {recognized_text}")

        # Feature Extraction
        features = self.extract_features(sample_audio)
        if features is not None:
            self.logger.debug(f"Extracted Features: {features}")

        # Classification (Assuming a trained model and sample data)
        # For demonstration, we'll create dummy data
        dummy_features = np.array([[0.1]*13, [0.2]*13, [0.3]*13])
        dummy_labels = ['speech', 'music', 'noise']
        self.train_classifier(dummy_features, dummy_labels)

        classification_result = self.classify_audio(sample_audio)
        if classification_result:
            self.logger.info(f"Classification Result: {classification_result}")


# Example Usage and Test Cases
if __name__ == "__main__":
    import os

    # Initialize AudioProcessor
    project_id = os.getenv('PROJECT_ID', 'default_project')  # Ensure PROJECT_ID is set
    audio_processor = AudioProcessor(project_id=project_id)

    # Run sample operations
    audio_processor.run_sample_operations()
