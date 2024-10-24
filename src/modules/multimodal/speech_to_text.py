# src/modules/multimodal/speech_to_text.py

import os
import logging
import speech_recognition as sr
from typing import Optional, Callable
from utils.logger import get_logger
from utils.configuration_manager import ConfigurationManager
from src.modules.multimodal.multimodal_engine import MultimodalEngine


class SpeechToText:
    """
    Handles converting voice commands into text for the system to process,
    expanding Hermod’s interactive capabilities.
    """

    def __init__(self, project_id: str, command_callback: Optional[Callable[[str], None]] = None):
        """
        Initializes the SpeechToText with necessary configurations and tools.

        Args:
            project_id (str): Unique identifier for the project.
            command_callback (Optional[Callable[[str], None]]): Callback function to handle recognized commands.
        """
        self.logger = get_logger(__name__)
        self.project_id = project_id
        self.command_callback = command_callback
        self.config_manager = ConfigurationManager()
        self.config = self.config_manager.get_configuration(project_id)

        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        # Initialize MultimodalEngine or other components if needed
        self.multimodal_engine = MultimodalEngine(project_id=project_id)

        self.logger.info(f"SpeechToText initialized for project '{project_id}'.")

    # ----------------------------
    # Speech Recognition Methods
    # ----------------------------

    def listen_and_recognize(self, timeout: int = 5, phrase_time_limit: Optional[int] = None) -> Optional[str]:
        """
        Listens to microphone input and recognizes speech.

        Args:
            timeout (int, optional): Maximum number of seconds that it will wait for a phrase to start before giving up.
            phrase_time_limit (Optional[int], optional): Maximum number of seconds a phrase can be.

        Returns:
            Optional[str]: Recognized text or None if failed.
        """
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source)
                self.logger.info("Listening for speech...")
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            text = self.recognizer.recognize_google(audio)
            self.logger.info(f"Recognized Speech: {text}")
            return text
        except sr.WaitTimeoutError:
            self.logger.warning("Listening timed out while waiting for phrase to start.")
            return None
        except sr.UnknownValueError:
            self.logger.warning("Speech Recognition could not understand audio.")
            return None
        except sr.RequestError as e:
            self.logger.error(f"Could not request results from Speech Recognition service; {e}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to recognize speech: {e}")
            return None

    def process_command(self, command_text: str) -> None:
        """
        Processes the recognized command text.

        Args:
            command_text (str): Recognized command text.
        """
        self.logger.info(f"Processing Command: {command_text}")
        if self.command_callback:
            self.command_callback(command_text)
        else:
            # Default command processing (can be expanded)
            self._default_command_handler(command_text)

    def _default_command_handler(self, command_text: str) -> None:
        """
        Default handler for processing commands.

        Args:
            command_text (str): Recognized command text.
        """
        # Example: Simple keyword-based command execution
        if "start analysis" in command_text.lower():
            self.logger.info("Executing 'start analysis' command.")
            # Trigger analysis (placeholder)
            # e.g., self.multimodal_engine.run_analysis()
        elif "show commit history" in command_text.lower():
            self.logger.info("Executing 'show commit history' command.")
            # Retrieve and display commit history
            commit_history = self.multimodal_engine.get_commit_history(max_count=5)
            if commit_history:
                for commit in commit_history:
                    self.logger.info(commit)
        else:
            self.logger.warning(f"Unrecognized command: {command_text}")

    # ----------------------------
    # Continuous Listening
    # ----------------------------

    def start_continuous_listening(self, callback: Optional[Callable[[str], None]] = None) -> None:
        """
        Starts continuous listening in a separate thread.

        Args:
            callback (Optional[Callable[[str], None]], optional): Callback function to handle recognized commands.
        """
        import threading

        def listen():
            while True:
                text = self.listen_and_recognize()
                if text:
                    if callback:
                        callback(text)
                    else:
                        self.process_command(text)

        listener_thread = threading.Thread(target=listen, daemon=True)
        listener_thread.start()
        self.logger.info("Started continuous listening thread.")

    # ----------------------------
    # Example Usage and Test Cases
    # ----------------------------

    def run_sample_operations(self):
        """
        Demonstrates sample speech-to-text operations.
        """
        self.logger.info("Running sample speech-to-text operations.")

        # Example: Listen and recognize once
        recognized_text = self.listen_and_recognize(timeout=5, phrase_time_limit=5)
        if recognized_text:
            self.process_command(recognized_text)

        # Example: Start continuous listening with default command handler
        # self.start_continuous_listening()

        # For demonstration, we'll keep it simple and not start continuous listening


# Example Usage and Test Cases
if __name__ == "__main__":
    import os

    # Define a simple command callback function
    def my_command_handler(command: str):
        print(f"Command received: {command}")
        # Implement custom command handling logic here

    # Initialize SpeechToText
    project_id = os.getenv('PROJECT_ID', 'default_project')  # Ensure PROJECT_ID is set
    speech_to_text = SpeechToText(project_id=project_id, command_callback=my_command_handler)

    # Run sample operations
    speech_to_text.run_sample_operations()

    # To start continuous listening, uncomment the following line:
    # speech_to_text.start_continuous_listening()
