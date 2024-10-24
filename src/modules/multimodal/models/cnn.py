# src/modules/multimodal/models/cnn.py

import os
import logging
import torch
import torch.nn as nn
from torchvision import models
from typing import Any, Dict, Optional
from utils.logger import get_logger
from utils.configuration_manager import ConfigurationManager


class CNNModel(nn.Module):
    """
    Defines a customizable Convolutional Neural Network (CNN) architecture.
    Can be initialized with pre-trained weights for transfer learning.
    """

    def __init__(self, model_name: str = 'resnet50', num_classes: int = 1000, pretrained: bool = True):
        """
        Initializes the CNNModel with the specified architecture.

        Args:
            model_name (str, optional): Name of the CNN architecture. Defaults to 'resnet50'.
            num_classes (int, optional): Number of output classes. Defaults to 1000.
            pretrained (bool, optional): Whether to use pre-trained weights. Defaults to True.
        """
        super(CNNModel, self).__init__()
        self.logger = get_logger(__name__)
        self.model_name = model_name.lower()
        self.num_classes = num_classes
        self.pretrained = pretrained

        try:
            if self.model_name == 'resnet50':
                self.model = models.resnet50(pretrained=self.pretrained)
                in_features = self.model.fc.in_features
                self.model.fc = nn.Linear(in_features, self.num_classes)
                self.logger.info(f"Initialized ResNet50 model with {self.num_classes} output classes.")
            elif self.model_name == 'vgg16':
                self.model = models.vgg16(pretrained=self.pretrained)
                in_features = self.model.classifier[6].in_features
                self.model.classifier[6] = nn.Linear(in_features, self.num_classes)
                self.logger.info(f"Initialized VGG16 model with {self.num_classes} output classes.")
            elif self.model_name == 'mobilenet_v2':
                self.model = models.mobilenet_v2(pretrained=self.pretrained)
                in_features = self.model.classifier[1].in_features
                self.model.classifier[1] = nn.Linear(in_features, self.num_classes)
                self.logger.info(f"Initialized MobileNetV2 model with {self.num_classes} output classes.")
            else:
                raise ValueError(f"Model '{self.model_name}' is not supported.")
        except Exception as e:
            self.logger.error(f"Failed to initialize CNN model '{self.model_name}': {e}")
            raise e

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the CNN model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output logits.
        """
        return self.model(x)

    def load_weights(self, weights_path: str) -> None:
        """
        Loads weights from a specified file into the model.

        Args:
            weights_path (str): Path to the weights file.
        """
        try:
            if not os.path.exists(weights_path):
                self.logger.error(f"Weights file '{weights_path}' does not exist.")
                raise FileNotFoundError(f"Weights file '{weights_path}' does not exist.")
            self.model.load_state_dict(torch.load(weights_path))
            self.logger.info(f"Loaded weights from '{weights_path}' into the model.")
        except Exception as e:
            self.logger.error(f"Failed to load weights from '{weights_path}': {e}")
            raise e

    def save_weights(self, save_path: str) -> None:
        """
        Saves the model's weights to a specified file.

        Args:
            save_path (str): Path to save the weights file.
        """
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(self.model.state_dict(), save_path)
            self.logger.info(f"Saved model weights to '{save_path}'.")
        except Exception as e:
            self.logger.error(f"Failed to save weights to '{save_path}': {e}")
            raise e


# Example Usage and Test Cases
if __name__ == "__main__":
    import torch

    # Initialize ConfigurationManager
    project_id = os.getenv('PROJECT_ID', 'default_project')  # Ensure PROJECT_ID is set
    config_manager = ConfigurationManager()
    config = config_manager.get_configuration(project_id)

    # Initialize CNNModel
    try:
        cnn = CNNModel(model_name='resnet50', num_classes=10, pretrained=True)
    except Exception as e:
        print(f"Failed to initialize CNNModel: {e}")
        exit(1)

    # Create a dummy input tensor
    dummy_input = torch.randn(1, 3, 224, 224)

    # Perform a forward pass
    try:
        output = cnn(dummy_input)
        print(f"Output shape: {output.shape}")  # Expected: [1, 10]
    except Exception as e:
        print(f"Forward pass failed: {e}")

    # Save and load weights (optional)
    # cnn.save_weights('models/resnet50_weights.pth')
    # cnn.load_weights('models/resnet50_weights.pth')
