"""
model.py - Spatio-Temporal LSTM for Activity Classification

Architecture Overview:
    Input (batch, seq_len=30, features=132)
        ↓
    LSTM Layer 1 (hidden=256, bidirectional=False)
        ↓
    Dropout (0.3)
        ↓
    LSTM Layer 2 (hidden=128)
        ↓
    Dropout (0.3)
        ↓
    FC Layer (128 → 64)
        ↓
    ReLU + Dropout
        ↓
    FC Layer (64 → 3)
        ↓
    Output: [Fall, Seizure, Normal] logits

Why LSTM over CNN3D?
    1. Computational Efficiency: LSTMs process sequences frame-by-frame with
       O(seq_len) complexity, while 3D CNNs have O(seq_len * H * W) complexity.
    2. Edge Deployment: LSTMs have ~10x fewer parameters for similar accuracy,
       crucial for real-time inference on resource-constrained devices.
    3. Variable Length: LSTMs naturally handle variable-length sequences without
       padding overhead during inference.
    4. Temporal Focus: For skeleton data, temporal patterns (velocity, acceleration)
       matter more than spatial convolutions.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config


class PatientSafetyLSTM(nn.Module):
    """
    LSTM-based classifier for fall and seizure detection from skeletal sequences.
    
    The model learns temporal patterns in body landmark movements:
    - Falls: Rapid downward movement, loss of upright posture
    - Seizures: Erratic, repetitive movements with high variance
    - Normal: Smooth, predictable movement patterns
    
    Attributes:
        lstm1: First LSTM layer for low-level temporal features
        lstm2: Second LSTM layer for high-level temporal abstractions
        fc1: First fully connected layer
        fc2: Output layer (logits, no softmax - CrossEntropyLoss handles it)
    """
    
    def __init__(self, 
                 input_size: int = config.INPUT_SIZE,
                 hidden_size_1: int = config.HIDDEN_SIZE_1,
                 hidden_size_2: int = config.HIDDEN_SIZE_2,
                 num_classes: int = config.NUM_CLASSES,
                 dropout: float = config.DROPOUT,
                 bidirectional: bool = False):
        """
        Initialize the LSTM model.
        
        Args:
            input_size: Number of features per timestep (33 landmarks × 4 = 132)
            hidden_size_1: Hidden units in first LSTM layer
            hidden_size_2: Hidden units in second LSTM layer
            num_classes: Number of output classes (Fall, Seizure, Normal)
            dropout: Dropout probability for regularization
            bidirectional: Whether to use bidirectional LSTM (doubles hidden size)
        """
        super(PatientSafetyLSTM, self).__init__()
        
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.num_directions = 2 if bidirectional else 1
        
        # Layer 1: Learn low-level temporal patterns (joint velocities, etc.)
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size_1,
            num_layers=1,
            batch_first=True,  # Input shape: (batch, seq, features)
            dropout=0,         # No dropout for single-layer LSTM
            bidirectional=bidirectional
        )
        
        self.dropout1 = nn.Dropout(dropout)
        
        # Layer 2: Learn high-level temporal patterns (activity signatures)
        self.lstm2 = nn.LSTM(
            input_size=hidden_size_1 * self.num_directions,
            hidden_size=hidden_size_2,
            num_layers=1,
            batch_first=True,
            dropout=0,
            bidirectional=bidirectional
        )
        
        self.dropout2 = nn.Dropout(dropout)
        
        # Classification head
        fc_input_size = hidden_size_2 * self.num_directions
        self.fc1 = nn.Linear(fc_input_size, 64)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, num_classes)
        
        # Initialize weights for better convergence
        self._init_weights()
    
    def _init_weights(self):
        """Initialize LSTM and FC weights using Xavier/Glorot initialization."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                # Input-hidden weights: Xavier uniform
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                # Hidden-hidden weights: Orthogonal (helps with vanishing gradients)
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                # Biases: Zero, except forget gate (set to 1 for better gradient flow)
                nn.init.zeros_(param)
                # Set forget gate bias to 1 (it's the second quarter of the bias vector)
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)
            elif 'fc' in name and 'weight' in name:
                nn.init.xavier_uniform_(param)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, features)
               where features = 132 (33 landmarks × 4 values)
               
        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        # LSTM Layer 1
        # lstm_out: (batch, seq_len, hidden_size_1 * num_directions)
        # h_n: (num_directions, batch, hidden_size_1)
        lstm_out1, (h_n1, c_n1) = self.lstm1(x)
        lstm_out1 = self.dropout1(lstm_out1)
        
        # LSTM Layer 2
        lstm_out2, (h_n2, c_n2) = self.lstm2(lstm_out1)
        lstm_out2 = self.dropout2(lstm_out2)
        
        # Take the last timestep's output for classification
        # This captures the accumulated temporal information
        # Shape: (batch, hidden_size_2 * num_directions)
        if self.num_directions == 2:
            # For bidirectional: concatenate forward and backward final states
            last_output = torch.cat([h_n2[0], h_n2[1]], dim=1)
        else:
            last_output = h_n2[-1]
        
        # Classification head
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout3(out)
        out = self.fc2(out)  # Raw logits (no softmax)
        
        return out
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get class probabilities (for inference).
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, features)
            
        Returns:
            Probability tensor of shape (batch_size, num_classes)
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get predictions with confidence scores.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, features)
            
        Returns:
            Tuple of (predicted_classes, confidence_scores)
        """
        probs = self.predict_proba(x)
        confidence, predictions = torch.max(probs, dim=1)
        return predictions, confidence


class LightweightLSTM(nn.Module):
    """
    A lighter version for extremely resource-constrained edge devices.
    
    Reduces parameters by ~60% while maintaining reasonable accuracy.
    Use this for deployment on devices with <1GB RAM.
    """
    
    def __init__(self,
                 input_size: int = config.INPUT_SIZE,
                 hidden_size: int = 64,
                 num_classes: int = config.NUM_CLASSES,
                 dropout: float = 0.2):
        super(LightweightLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_output = h_n[-1]
        out = self.dropout(last_output)
        out = self.fc(out)
        return out


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model: nn.Module, input_shape: Tuple[int, ...]) -> str:
    """Generate a summary of the model architecture."""
    total_params = count_parameters(model)
    
    summary = f"""
{'=' * 60}
MODEL SUMMARY
{'=' * 60}
Architecture: {model.__class__.__name__}
Input Shape: {input_shape}
Total Parameters: {total_params:,}
Estimated Size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)
{'=' * 60}

Layer Details:
"""
    for name, module in model.named_modules():
        if isinstance(module, (nn.LSTM, nn.Linear)):
            params = sum(p.numel() for p in module.parameters())
            summary += f"  {name}: {module.__class__.__name__} -> {params:,} params\n"
    
    return summary


if __name__ == "__main__":
    # Quick test of the model
    print("Testing PatientSafetyLSTM...")
    
    model = PatientSafetyLSTM()
    
    # Create dummy input: (batch=4, seq_len=30, features=132)
    dummy_input = torch.randn(4, config.SEQUENCE_LENGTH, config.INPUT_SIZE)
    
    # Forward pass
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: (4, {config.NUM_CLASSES})")
    
    # Get predictions
    predictions, confidence = model.predict(dummy_input)
    print(f"Predictions: {predictions}")
    print(f"Confidence: {confidence}")
    
    # Model summary
    print(get_model_summary(model, (config.SEQUENCE_LENGTH, config.INPUT_SIZE)))
    
    # Test lightweight model
    print("\nTesting LightweightLSTM...")
    light_model = LightweightLSTM()
    print(get_model_summary(light_model, (config.SEQUENCE_LENGTH, config.INPUT_SIZE)))
