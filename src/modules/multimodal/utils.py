# src/modules/multimodal/utils.py

import os
import logging
import numpy as np
import pandas as pd
import cv2
from typing import Any, Dict, List, Optional
from utils.logger import get_logger
from utils.configuration_manager import ConfigurationManager


class MultimodalUtils:
    """
    Contains helper functions and classes used across multimodal processing tasks,
    such as data transformations, format conversions, and common operations.
    """

    def __init__(self, project_id: str):
        """
        Initializes the MultimodalUtils with necessary configurations.

        Args:
            project_id (str): Unique identifier for the project.
        """
        self.logger = get_logger(__name__)
        self.project_id = project_id
        self.config_manager = ConfigurationManager()
        self.config = self.config_manager.get_configuration(project_id)

        # Load labels or other necessary configurations
        self.imagenet_labels = self._load_labels('imagenet_classes.txt')
        self.coco_labels = self._load_labels('coco_labels.txt')

        self.logger.info(f"MultimodalUtils initialized for project '{project_id}'.")

    # ----------------------------
    # Label Loading
    # ----------------------------

    def _load_labels(self, label_file: str) -> List[str]:
        """
        Loads labels from a specified file.

        Args:
            label_file (str): Path to the label file.

        Returns:
            List[str]: List of labels.
        """
        try:
            label_path = os.path.join(os.path.dirname(__file__), label_file)
            with open(label_path, 'r') as f:
                labels = [line.strip() for line in f.readlines()]
            self.logger.info(f"Loaded {len(labels)} labels from '{label_file}'.")
            return labels
        except Exception as e:
            self.logger.error(f"Failed to load labels from '{label_file}': {e}")
            return []

    # ----------------------------
    # Data Transformation
    # ----------------------------

    def normalize_array(self, array: np.ndarray, mean: Optional[List[float]] = None,
                       std: Optional[List[float]] = None) -> np.ndarray:
        """
        Normalizes a NumPy array using provided mean and standard deviation.

        Args:
            array (np.ndarray): The array to normalize.
            mean (Optional[List[float]]): Mean values for normalization.
            std (Optional[List[float]]): Standard deviation values for normalization.

        Returns:
            np.ndarray: Normalized array.
        """
        try:
            if mean is not None and std is not None:
                normalized = (array - mean) / std
            else:
                normalized = (array - np.mean(array)) / np.std(array)
            self.logger.debug(f"Normalized array with mean={mean} and std={std}.")
            return normalized
        except Exception as e:
            self.logger.error(f"Failed to normalize array: {e}")
            return array

    def save_numpy_array(self, array: np.ndarray, file_path: str) -> None:
        """
        Saves a NumPy array to a specified file path.

        Args:
            array (np.ndarray): The array to save.
            file_path (str): Destination file path.
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            np.save(file_path, array)
            self.logger.info(f"Saved NumPy array to '{file_path}'.")
        except Exception as e:
            self.logger.error(f"Failed to save NumPy array to '{file_path}': {e}")

    def load_numpy_array(self, file_path: str) -> Optional[np.ndarray]:
        """
        Loads a NumPy array from a specified file path.

        Args:
            file_path (str): Path to the NumPy file.

        Returns:
            Optional[np.ndarray]: Loaded array or None if failed.
        """
        try:
            array = np.load(file_path)
            self.logger.info(f"Loaded NumPy array from '{file_path}'.")
            return array
        except Exception as e:
            self.logger.error(f"Failed to load NumPy array from '{file_path}': {e}")
            return None

    # ----------------------------
    # Format Conversion
    # ----------------------------

    def convert_audio_format(self, input_path: str, output_path: str, target_sr: int = 22050) -> bool:
        """
        Converts an audio file to a specified format and sample rate.

        Args:
            input_path (str): Path to the input audio file.
            output_path (str): Path to save the converted audio file.
            target_sr (int, optional): Target sample rate. Defaults to 22050.

        Returns:
            bool: True if conversion is successful, False otherwise.
        """
        try:
            import librosa
            y, sr_ = librosa.load(input_path, sr=target_sr)
            librosa.output.write_wav(output_path, y, target_sr)
            self.logger.info(f"Converted audio from '{input_path}' to '{output_path}' with sample rate {target_sr}.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to convert audio format from '{input_path}' to '{output_path}': {e}")
            return False

    def extract_video_frames(self, video_path: str, frames_dir: str, fps: int = 1) -> bool:
        """
        Extracts frames from a video file at a specified frames per second (fps).

        Args:
            video_path (str): Path to the video file.
            frames_dir (str): Directory to save extracted frames.
            fps (int, optional): Number of frames to extract per second. Defaults to 1.

        Returns:
            bool: True if extraction is successful, False otherwise.
        """
        try:
            os.makedirs(frames_dir, exist_ok=True)
            vidcap = cv2.VideoCapture(video_path)
            if not vidcap.isOpened():
                self.logger.error(f"Failed to open video file '{video_path}'.")
                return False
            frame_rate = vidcap.get(cv2.CAP_PROP_FPS)
            interval = int(frame_rate / fps) if frame_rate > 0 else 1
            count = 0
            saved = 0
            while True:
                success, image = vidcap.read()
                if not success:
                    break
                if count % interval == 0:
                    frame_filename = os.path.join(frames_dir, f"frame_{saved:04d}.jpg")
                    cv2.imwrite(frame_filename, image)
                    saved += 1
                count += 1
            vidcap.release()
            self.logger.info(f"Extracted {saved} frames from '{video_path}' to '{frames_dir}' at {fps} fps.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to extract frames from '{video_path}': {e}")
            return False

    # ----------------------------
    # Common Operations
    # ----------------------------

    def get_file_list(self, directory: str, extensions: Optional[List[str]] = None) -> List[str]:
        """
        Retrieves a list of files from a directory with optional filtering by extensions.

        Args:
            directory (str): Directory path.
            extensions (Optional[List[str]], optional): List of file extensions to include. Defaults to None.

        Returns:
            List[str]: List of file paths.
        """
        try:
            files = []
            for root, _, filenames in os.walk(directory):
                for filename in filenames:
                    if extensions:
                        if any(filename.lower().endswith(ext.lower()) for ext in extensions):
                            files.append(os.path.join(root, filename))
                    else:
                        files.append(os.path.join(root, filename))
            self.logger.info(f"Retrieved {len(files)} files from '{directory}' with extensions {extensions}.")
            return files
        except Exception as e:
            self.logger.error(f"Failed to retrieve files from '{directory}': {e}")
            return []

    def save_dataframe_to_csv(self, df: pd.DataFrame, file_path: str) -> None:
        """
        Saves a pandas DataFrame to a CSV file.

        Args:
            df (pd.DataFrame): DataFrame to save.
            file_path (str): Path to the CSV file.
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            df.to_csv(file_path, index=False)
            self.logger.info(f"Saved DataFrame to '{file_path}'.")
        except Exception as e:
            self.logger.error(f"Failed to save DataFrame to '{file_path}': {e}")

    def load_dataframe_from_csv(self, file_path: str) -> Optional[pd.DataFrame]:
        """
        Loads a pandas DataFrame from a CSV file.

        Args:
            file_path (str): Path to the CSV file.

        Returns:
            Optional[pd.DataFrame]: Loaded DataFrame or None if failed.
        """
        try:
            df = pd.read_csv(file_path)
            self.logger.info(f"Loaded DataFrame from '{file_path}'.")
            return df
        except Exception as e:
            self.logger.error(f"Failed to load DataFrame from '{file_path}': {e}")
            return None

    # ----------------------------
    # Example Usage and Test Cases
    # ----------------------------

    def run_sample_operations(self):
        """
        Demonstrates sample utility operations.
        """
        self.logger.info("Running sample utility operations.")

        # Sample normalization
        sample_array = np.array([1, 2, 3, 4, 5], dtype=float)
        normalized = self.normalize_array(sample_array)
        self.logger.debug(f"Normalized Array: {normalized}")

        # Sample save and load NumPy array
        numpy_file = 'processed_data/sample_array.npy'
        self.save_numpy_array(sample_array, numpy_file)
        loaded_array = self.load_numpy_array(numpy_file)
        if loaded_array is not None:
            self.logger.debug(f"Loaded Array: {loaded_array}")

        # Sample format conversion
        # Convert audio format (assuming 'input_audio.wav' exists)
        # success = self.convert_audio_format('input_audio.wav', 'converted_audio.wav')
        # if success:
        #     self.logger.info("Audio format conversion successful.")

        # Extract video frames (assuming 'input_video.mp4' exists)
        # success = self.extract_video_frames('input_video.mp4', 'extracted_frames/', fps=2)
        # if success:
        #     self.logger.info("Video frame extraction successful.")

        # Sample file list retrieval
        files = self.get_file_list('processed_data/', extensions=['.npy', '.csv'])
        self.logger.debug(f"Retrieved Files: {files}")

        # Sample DataFrame operations
        df = pd.DataFrame({
            'Feature1': [0.1, 0.2, 0.3],
            'Feature2': [1, 2, 3],
            'Label': ['A', 'B', 'C']
        })
        csv_path = 'processed_data/sample_df.csv'
        self.save_dataframe_to_csv(df, csv_path)
        loaded_df = self.load_dataframe_from_csv(csv_path)
        if loaded_df is not None:
            self.logger.debug(f"Loaded DataFrame:\n{loaded_df}")


# Example Usage and Test Cases
if __name__ == "__main__":
    import os

    # Initialize MultimodalUtils
    project_id = os.getenv('PROJECT_ID', 'default_project')  # Ensure PROJECT_ID is set
    utils = MultimodalUtils(project_id=project_id)

    # Run sample operations
    utils.run_sample_operations()
