# src/modules/multimodal/image_processor.py

import os
import logging
import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
from typing import Any, Dict, Optional
from PIL import Image
from utils.logger import get_logger
from utils.configuration_manager import ConfigurationManager


class ImageProcessor:
    """
    Processes image data for tasks like object detection, image classification, and image segmentation.
    Provides computer vision capabilities to Hermod.
    """

    def __init__(self, project_id: str):
        """
        Initializes the ImageProcessor with necessary configurations and tools.

        Args:
            project_id (str): Unique identifier for the project.
        """
        self.logger = get_logger(__name__)
        self.project_id = project_id
        self.config_manager = ConfigurationManager()
        self.config = self.config_manager.get_configuration(project_id)

        # Initialize models
        self.classification_model = self._load_classification_model()
        self.object_detection_model = self._load_object_detection_model()
        # Add image segmentation model if needed

        self.logger.info(f"ImageProcessor initialized for project '{project_id}'.")

    # ----------------------------
    # Model Loading
    # ----------------------------

    def _load_classification_model(self) -> torch.nn.Module:
        """
        Loads a pre-trained image classification model.

        Returns:
            torch.nn.Module: Pre-trained classification model.
        """
        try:
            model = models.resnet50(pretrained=True)
            model.eval()
            self.logger.info("Loaded pre-trained ResNet50 classification model.")
            return model
        except Exception as e:
            self.logger.error(f"Failed to load classification model: {e}")
            raise e

    def _load_object_detection_model(self) -> torch.nn.Module:
        """
        Loads a pre-trained object detection model.

        Returns:
            torch.nn.Module: Pre-trained object detection model.
        """
        try:
            model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
            model.eval()
            self.logger.info("Loaded pre-trained Faster R-CNN object detection model.")
            return model
        except Exception as e:
            self.logger.error(f"Failed to load object detection model: {e}")
            raise e

    # ----------------------------
    # Image Classification
    # ----------------------------

    def classify_image(self, image_path: str) -> Optional[Dict[str, Any]]:
        """
        Classifies an image and returns the top prediction.

        Args:
            image_path (str): Path to the image file.

        Returns:
            Optional[Dict[str, Any]]: Classification result or None if failed.
        """
        try:
            image = Image.open(image_path).convert('RGB')
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            input_tensor = preprocess(image)
            input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

            with torch.no_grad():
                output = self.classification_model(input_batch)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            top_prob, top_catid = torch.topk(probabilities, 1)
            # Load ImageNet labels
            labels_path = os.path.join(os.path.dirname(__file__), 'imagenet_classes.txt')
            with open(labels_path, "r") as f:
                categories = [s.strip() for s in f.readlines()]
            prediction = categories[top_catid]
            result = {
                'prediction': prediction,
                'probability': top_prob.item()
            }
            self.logger.info(f"Classified image '{image_path}': {result}")
            return result
        except Exception as e:
            self.logger.error(f"Failed to classify image '{image_path}': {e}")
            return None

    # ----------------------------
    # Object Detection
    # ----------------------------

    def detect_objects(self, image_path: str, threshold: float = 0.5) -> Optional[Dict[str, Any]]:
        """
        Detects objects in an image and returns bounding boxes with labels and scores.

        Args:
            image_path (str): Path to the image file.
            threshold (float, optional): Confidence threshold for detections. Defaults to 0.5.

        Returns:
            Optional[Dict[str, Any]]: Detection results or None if failed.
        """
        try:
            image = Image.open(image_path).convert("RGB")
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            img_tensor = transform(image)
            with torch.no_grad():
                detections = self.object_detection_model([img_tensor])[0]

            # Load COCO labels
            labels_path = os.path.join(os.path.dirname(__file__), 'coco_labels.txt')
            with open(labels_path, "r") as f:
                categories = [s.strip() for s in f.readlines()]

            results = []
            for box, label, score in zip(detections['boxes'], detections['labels'], detections['scores']):
                if score >= threshold:
                    result = {
                        'bounding_box': box.tolist(),
                        'label': categories[label],
                        'score': score.item()
                    }
                    results.append(result)

            self.logger.info(f"Detected objects in '{image_path}': {results}")
            return {'detections': results}
        except Exception as e:
            self.logger.error(f"Failed to detect objects in '{image_path}': {e}")
            return None

    # ----------------------------
    # Image Segmentation
    # ----------------------------

    # Implement image segmentation methods if needed using models like DeepLabV3 or U-Net

    # ----------------------------
    # Example Usage and Test Cases
    # ----------------------------

    def run_sample_operations(self):
        """
        Demonstrates sample image processing operations.
        """
        self.logger.info("Running sample image processing operations.")

        # Sample image file path
        sample_image = 'sample_image.jpg'  # Replace with an actual image file path

        # Image Classification
        classification_result = self.classify_image(sample_image)
        if classification_result:
            self.logger.info(f"Classification Result: {classification_result}")

        # Object Detection
        detection_result = self.detect_objects(sample_image)
        if detection_result:
            self.logger.info(f"Object Detection Result: {detection_result}")


# Example Usage and Test Cases
if __name__ == "__main__":
    import os

    # Initialize ImageProcessor
    project_id = os.getenv('PROJECT_ID', 'default_project')  # Ensure PROJECT_ID is set
    image_processor = ImageProcessor(project_id=project_id)

    # Run sample operations
    image_processor.run_sample_operations()
