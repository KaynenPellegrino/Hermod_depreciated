# src/modules/multimodal/video_processor.py

import os
import logging
import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
from typing import Any, Dict, List, Optional
from PIL import Image
from utils.logger import get_logger
from utils.configuration_manager import ConfigurationManager


class VideoProcessor:
    """
    Handles video data processing, including tasks like video classification,
    motion detection, and frame extraction. Allows the system to analyze and interpret video inputs.
    """

    def __init__(self, project_id: str):
        """
        Initializes the VideoProcessor with necessary configurations and tools.

        Args:
            project_id (str): Unique identifier for the project.
        """
        self.logger = get_logger(__name__)
        self.project_id = project_id
        self.config_manager = ConfigurationManager()
        self.config = self.config_manager.get_configuration(project_id)

        # Initialize video classification model
        self.classification_model = self._load_classification_model()

        # Initialize motion detection parameters
        self.previous_frame = None
        self.motion_threshold = self.config.get('motion_threshold', 25)

        self.logger.info(f"VideoProcessor initialized for project '{project_id}'.")

    # ----------------------------
    # Model Loading
    # ----------------------------

    def _load_classification_model(self) -> torch.nn.Module:
        """
        Loads a pre-trained video classification model.

        Returns:
            torch.nn.Module: Pre-trained classification model.
        """
        try:
            # Placeholder: Using image classification model for frame-wise classification
            # For actual video classification, consider using 3D CNNs or pre-trained video models
            model = models.video.r2plus1d_18(pretrained=True)
            model.eval()
            self.logger.info("Loaded pre-trained R(2+1)D video classification model.")
            return model
        except Exception as e:
            self.logger.error(f"Failed to load video classification model: {e}")
            raise e

    # ----------------------------
    # Video Classification
    # ----------------------------

    def classify_video(self, video_path: str, num_segments: int = 8) -> Optional[Dict[str, Any]]:
        """
        Classifies a video and returns the top prediction.

        Args:
            video_path (str): Path to the video file.
            num_segments (int, optional): Number of segments to divide the video into for classification. Defaults to 8.

        Returns:
            Optional[Dict[str, Any]]: Classification result or None if failed.
        """
        try:
            # Load video frames
            frames = self._extract_frames(video_path, num_segments)
            if not frames:
                self.logger.warning(f"No frames extracted from '{video_path}' for classification.")
                return None

            # Preprocess frames
            input_tensor = self._preprocess_frames(frames)

            # Perform classification
            with torch.no_grad():
                outputs = self.classification_model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                top_prob, top_catid = torch.topk(probabilities, 1)

            # Load Kinetics-400 labels (assuming 'kinetics_labels.txt' exists)
            labels_path = os.path.join(os.path.dirname(__file__), 'kinetics_labels.txt')
            with open(labels_path, "r") as f:
                categories = [s.strip() for s in f.readlines()]

            prediction = categories[top_catid]
            result = {
                'prediction': prediction.item(),
                'probability': top_prob.item()
            }
            self.logger.info(f"Classified video '{video_path}': {result}")
            return result
        except Exception as e:
            self.logger.error(f"Failed to classify video '{video_path}': {e}")
            return None

    def _extract_frames(self, video_path: str, num_segments: int) -> List[Image.Image]:
        """
        Extracts a specified number of frames from a video file.

        Args:
            video_path (str): Path to the video file.
            num_segments (int): Number of frames to extract.

        Returns:
            List[Image.Image]: List of extracted PIL Image frames.
        """
        try:
            vidcap = cv2.VideoCapture(video_path)
            if not vidcap.isOpened():
                self.logger.error(f"Failed to open video file '{video_path}'.")
                return []
            frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            interval = frame_count // num_segments
            frames = []
            for i in range(num_segments):
                frame_id = i * interval
                vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                success, image = vidcap.read()
                if success:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(image)
                    frames.append(pil_image)
                else:
                    self.logger.warning(f"Failed to read frame {frame_id} from '{video_path}'.")
            vidcap.release()
            self.logger.debug(f"Extracted {len(frames)} frames from '{video_path}'.")
            return frames
        except Exception as e:
            self.logger.error(f"Failed to extract frames from '{video_path}': {e}")
            return []

    def _preprocess_frames(self, frames: List[Image.Image]) -> torch.Tensor:
        """
        Preprocesses frames for video classification.

        Args:
            frames (List[Image.Image]): List of PIL Image frames.

        Returns:
            torch.Tensor: Preprocessed tensor ready for model input.
        """
        try:
            preprocess = transforms.Compose([
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                                     std=[0.22803, 0.22145, 0.216989]),
            ])
            processed_frames = [preprocess(frame) for frame in frames]
            # Stack frames to create a tensor of shape [C, T, H, W]
            input_tensor = torch.stack(processed_frames)  # Shape: [T, C, H, W]
            input_tensor = input_tensor.permute(1, 0, 2, 3)  # Shape: [C, T, H, W]
            input_tensor = input_tensor.unsqueeze(0)  # Shape: [1, C, T, H, W]
            self.logger.debug(f"Preprocessed frames into tensor shape {input_tensor.shape}.")
            return input_tensor
        except Exception as e:
            self.logger.error(f"Failed to preprocess frames: {e}")
            return torch.Tensor()

    # ----------------------------
    # Motion Detection
    # ----------------------------

    def detect_motion(self, video_path: str, output_path: str = 'motion_detection_output.avi') -> bool:
        """
        Detects motion in a video file and saves the output video with motion highlighted.

        Args:
            video_path (str): Path to the video file.
            output_path (str, optional): Path to save the output video. Defaults to 'motion_detection_output.avi'.

        Returns:
            bool: True if motion detection is successful, False otherwise.
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.logger.error(f"Failed to open video file '{video_path}'.")
                return False

            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)

                if self.previous_frame is None:
                    self.previous_frame = gray
                    continue

                # Compute difference between current frame and previous frame
                frame_delta = cv2.absdiff(self.previous_frame, gray)
                thresh = cv2.threshold(frame_delta, self.motion_threshold, 255, cv2.THRESH_BINARY)[1]
                thresh = cv2.dilate(thresh, None, iterations=2)

                # Find contours
                contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    if cv2.contourArea(contour) < 500:
                        continue
                    (x, y, w, h) = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Write the frame into the file 'output_path'
                out.write(frame)

                # Update previous frame
                self.previous_frame = gray

            # Release everything if job is finished
            cap.release()
            out.release()
            self.logger.info(f"Motion detection completed and saved to '{output_path}'.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to perform motion detection on '{video_path}': {e}")
            return False

    # ----------------------------
    # Video Segmentation
    # ----------------------------

    # Implement video segmentation methods if needed using models like DeepLab or Mask R-CNN

    # ----------------------------
    # Helper Methods
    # ----------------------------

    def get_video_length(self, video_path: str) -> Optional[float]:
        """
        Retrieves the length of a video in seconds.

        Args:
            video_path (str): Path to the video file.

        Returns:
            Optional[float]: Length of the video in seconds or None if failed.
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.logger.error(f"Failed to open video file '{video_path}'.")
                return None
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = length / fps if fps > 0 else 0
            cap.release()
            self.logger.info(f"Video '{video_path}' duration: {duration} seconds.")
            return duration
        except Exception as e:
            self.logger.error(f"Failed to get video length for '{video_path}': {e}")
            return None

    # ----------------------------
    # Example Usage and Test Cases
    # ----------------------------

    def run_sample_operations(self):
        """
        Demonstrates sample video processing operations.
        """
        self.logger.info("Running sample video processing operations.")

        # Sample video file path
        sample_video = 'sample_video.mp4'  # Replace with an actual video file path

        # Video Classification
        classification_result = self.classify_video(sample_video, num_segments=8)
        if classification_result:
            self.logger.info(f"Video Classification Result: {classification_result}")

        # Motion Detection
        motion_output = 'motion_detection_output.avi'
        success = self.detect_motion(sample_video, output_path=motion_output)
        if success:
            self.logger.info(f"Motion detection output saved to '{motion_output}'.")

        # Get Video Length
        duration = self.get_video_length(sample_video)
        if duration is not None:
            self.logger.info(f"Video Length: {duration} seconds.")


# Example Usage and Test Cases
if __name__ == "__main__":
    import os

    # Initialize VideoProcessor
    project_id = os.getenv('PROJECT_ID', 'default_project')  # Ensure PROJECT_ID is set
    video_processor = VideoProcessor(project_id=project_id)

    # Run sample operations
    video_processor.run_sample_operations()
