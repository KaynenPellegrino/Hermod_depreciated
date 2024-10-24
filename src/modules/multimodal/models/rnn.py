# src/modules/multimodal/models/rnn.py

import os
import logging
import torch
import torch.nn as nn
from typing import Any, Dict, Optional
from utils.logger import get_logger
from utils.configuration_manager import ConfigurationManager


class RNNModel(nn.Module):
    """
    Defines a customizable Recurrent Neural Network (RNN) architecture.
    Supports Vanilla RNN, LSTM, and GRU layers for sequential data processing.
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 rnn_type: str = 'lstm', num_classes: int = 10, bidirectional: bool = False):
        """
        Initializes the RNNModel with the specified architecture.

        Args:
            input_size (int): Number of expected features in the input.
            hidden_size (int): Number of features in the hidden state.
            num_layers (int, optional): Number of recurrent layers. Defaults to 1.
            rnn_type (str, optional): Type of RNN ('rnn', 'lstm', 'gru'). Defaults to 'lstm'.
            num_classes (int, optional): Number of output classes. Defaults to 10.
            bidirectional (bool, optional): If True, becomes a bidirectional RNN. Defaults to False.
        """
        super(RNNModel, self).__init__()
        self.logger = get_logger(__name__)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()
        self.num_classes = num_classes
        self.bidirectional = bidirectional

        try:
            if self.rnn_type == 'rnn':
                self.rnn = nn.RNN(input_size, hidden_size, num_layers,
                                  batch_first=True, bidirectional=bidirectional)
                self.logger.info(
                    f"Initialized Vanilla RNN with hidden_size={hidden_size}, num_layers={num_layers}, bidirectional={bidirectional}.")
            elif self.rnn_type == 'lstm':
                self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                                   batch_first=True, bidirectional=bidirectional)
                self.logger.info(
                    f"Initialized LSTM with hidden_size={hidden_size}, num_layers={num_layers}, bidirectional={bidirectional}.")
            elif self.rnn_type == 'gru':
                self.rnn = nn.GRU(input_size, hidden_size, num_layers,
                                  batch_first=True, bidirectional=bidirectional)
                self.logger.info(
                    f"Initialized GRU with hidden_size={hidden_size}, num_layers={num_layers}, bidirectional={bidirectional}.")
            else:
                raise ValueError(f"RNN type '{self.rnn_type}' is not supported.")

            direction = 2 if bidirectional else 1
            self.fc = nn.Linear(hidden_size * direction, num_classes)
            self.logger.info(
                f"Added Linear layer with input features={hidden_size * direction} and output features={num_classes}.")
        except Exception as e:
            self.logger.error(f"Failed to initialize RNN model: {e}")
            raise e

    def forward(self, x: torch.Tensor, hidden: Optional[Any] = None) -> torch.Tensor:
        """
        Defines the forward pass of the RNN model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_length, input_size).
            hidden (Optional[Any], optional): Initial hidden state. Defaults to None.

        Returns:
            torch.Tensor: Output logits of shape (batch, num_classes).
        """
        try:
            if self.rnn_type == 'lstm':
                output, (hn, cn) = self.rnn(x, hidden)
            else:
                output, hn = self.rnn(x, hidden)

            # Take the output from the last time step
            if self.bidirectional:
                hn = hn.view(self.num_layers, 2, x.size(0), self.hidden_size)
                hn = torch.cat((hn[-1, 0], hn[-1, 1]), dim=1)
            else:
                hn = hn[-1]

            logits = self.fc(hn)
            return logits
        except Exception as e:
            self.logger.error(f"Forward pass failed: {e}")
            raise e

    def init_hidden(self, batch_size: int) -> Any:
        """
        Initializes the hidden state.

        Args:
            batch_size (int): Batch size.

        Returns:
            Any: Initialized hidden state.
        """
        try:
            num_directions = 2 if self.bidirectional else 1
            if self.rnn_type == 'lstm':
                h0 = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size)
                c0 = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size)
                return (h0, c0)
            else:
                h0 = torch.zeros(self.num_layers * num_directions, batch_size, self.hidden_size)
                return h0
        except Exception as e:
            self.logger.error(f"Failed to initialize hidden state: {e}")
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
            torch.save(self.state_dict(), save_path)
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

    # Define RNN parameters
    input_size = 40  # Example: number of MFCC features for audio
    hidden_size = 128
    num_layers = 2
    rnn_type = 'lstm'
    num_classes = 10
    bidirectional = True

    # Initialize RNNModel
    try:
        rnn = RNNModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                       rnn_type=rnn_type, num_classes=num_classes, bidirectional=bidirectional)
    except Exception as e:
        print(f"Failed to initialize RNNModel: {e}")
        exit(1)

    # Create a dummy input tensor
    batch_size = 16
    seq_length = 50
    dummy_input = torch.randn(batch_size, seq_length, input_size)

    # Initialize hidden state
    hidden = rnn.init_hidden(batch_size)

    # Perform a forward pass
    try:
        output = rnn(dummy_input, hidden)
        print(f"Output shape: {output.shape}")  # Expected: [batch_size, num_classes]
    except Exception as e:
        print(f"Forward pass failed: {e}")

    # Save and load weights (optional)
    # rnn.save_weights('models/rnn_weights.pth')
    # rnn.load_weights('models/rnn_weights.pth')
