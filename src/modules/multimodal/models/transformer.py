# src/modules/multimodal/models/transformer.py

import os
import logging
import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer, RobertaConfig
from typing import Any, Dict, Optional
from utils.logger import get_logger
from utils.configuration_manager import ConfigurationManager


class TransformerModel(nn.Module):
    """
    Defines a Transformer-based model for processing sequential data such as text or speech.
    Utilizes pre-trained models like RoBERTa for natural language understanding tasks.
    """

    def __init__(self, model_name: str = 'roberta-base', num_classes: int = 10, pretrained: bool = True):
        """
        Initializes the TransformerModel with the specified architecture.

        Args:
            model_name (str, optional): Name of the transformer model. Defaults to 'roberta-base'.
            num_classes (int, optional): Number of output classes. Defaults to 10.
            pretrained (bool, optional): Whether to use pre-trained weights. Defaults to True.
        """
        super(TransformerModel, self).__init__()
        self.logger = get_logger(__name__)
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained

        try:
            if 'roberta' in self.model_name.lower():
                self.roberta = RobertaModel.from_pretrained(self.model_name) if self.pretrained else RobertaModel(RobertaConfig())
                self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name) if self.pretrained else RobertaTokenizer(RobertaConfig())
                self.fc = nn.Linear(self.roberta.config.hidden_size, self.num_classes)
                self.logger.info(f"Initialized RoBERTa model '{self.model_name}' with {self.num_classes} output classes.")
            else:
                raise ValueError(f"Transformer model '{self.model_name}' is not supported.")
        except Exception as e:
            self.logger.error(f"Failed to initialize Transformer model '{self.model_name}': {e}")
            raise e

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the Transformer model.

        Args:
            input_ids (torch.Tensor): Tensor of token IDs.
            attention_mask (torch.Tensor): Tensor indicating which tokens should be attended to.

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        try:
            outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
            cls_output = outputs.last_hidden_state[:, 0, :]  # Extract <s> token representation
            logits = self.fc(cls_output)
            return logits
        except Exception as e:
            self.logger.error(f"Forward pass failed: {e}")
            raise e

    def tokenize(self, texts: list, max_length: int = 128) -> Dict[str, torch.Tensor]:
        """
        Tokenizes a list of texts for input into the Transformer model.

        Args:
            texts (list): List of text strings.
            max_length (int, optional): Maximum sequence length. Defaults to 128.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing 'input_ids' and 'attention_mask'.
        """
        try:
            encoding = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
            return encoding
        except Exception as e:
            self.logger.error(f"Tokenization failed: {e}")
            raise e

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
            self.load_state_dict(torch.load(weights_path))
            self.logger.info(f"Loaded weights from '{weights_path}' into the Transformer model.")
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
            torch.save(self.state_dict(), save_path)
            self.logger.info(f"Saved Transformer model weights to '{save_path}'.")
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

    # Define Transformer parameters
    model_name = 'roberta-base'
    num_classes = 5
    pretrained = True

    # Initialize TransformerModel
    try:
        transformer = TransformerModel(model_name=model_name, num_classes=num_classes, pretrained=pretrained)
    except Exception as e:
        print(f"Failed to initialize TransformerModel: {e}")
        exit(1)

    # Sample texts
    texts = [
        "Hermod is an advanced multimodal AI assistant.",
        "It integrates data from various sources seamlessly.",
        "This enables comprehensive analysis and interaction."
    ]

    # Tokenize inputs
    try:
        inputs = transformer.tokenize(texts, max_length=50)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
    except Exception as e:
        print(f"Tokenization failed: {e}")
        exit(1)

    # Perform a forward pass
    try:
        output = transformer(input_ids=input_ids, attention_mask=attention_mask)
        print(f"Output logits shape: {output.shape}")  # Expected: [batch_size, num_classes]
    except Exception as e:
        print(f"Forward pass failed: {e}")

    # Save and load weights (optional)
    # transformer.save_weights('models/transformer_weights.pth')
    # transformer.load_weights('models/transformer_weights.pth')
