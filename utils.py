"""
utils.py - Shared Utility Functions

This module provides common utilities used across preprocessing,
training, and inference components.
"""

import time
from pathlib import Path
from typing import Optional, Tuple, Callable
from functools import wraps
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import config


def timer(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.
    
    Usage:
        @timer
        def my_function():
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {(end - start) * 1000:.2f}ms")
        return result
    return wrapper


class InferenceTimer:
    """
    Context manager for precise inference timing.
    
    Usage:
        with InferenceTimer() as timer:
            # inference code
        print(f"Inference took {timer.elapsed_ms}ms")
    """
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.elapsed_ms = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.end_time = time.perf_counter()
        self.elapsed_ms = (self.end_time - self.start_time) * 1000


class LatencyTracker:
    """
    Track and report inference latencies over time.
    
    Maintains a rolling window of latency measurements and
    provides statistics (mean, max, min, percentiles).
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.latencies = []
    
    def add(self, latency_ms: float):
        """Add a latency measurement."""
        self.latencies.append(latency_ms)
        if len(self.latencies) > self.window_size:
            self.latencies.pop(0)
    
    @property
    def mean(self) -> float:
        """Average latency in ms."""
        return np.mean(self.latencies) if self.latencies else 0.0
    
    @property
    def max(self) -> float:
        """Maximum latency in ms."""
        return np.max(self.latencies) if self.latencies else 0.0
    
    @property
    def min(self) -> float:
        """Minimum latency in ms."""
        return np.min(self.latencies) if self.latencies else 0.0
    
    @property
    def p95(self) -> float:
        """95th percentile latency in ms."""
        return np.percentile(self.latencies, 95) if self.latencies else 0.0
    
    def summary(self) -> str:
        """Get a summary string of latency statistics."""
        return (f"Latency (ms): mean={self.mean:.1f}, "
                f"min={self.min:.1f}, max={self.max:.1f}, p95={self.p95:.1f}")


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Apply hip-center and torso-scale normalization to landmarks.
    
    This is the same normalization used in preprocessing, extracted
    here for use during real-time inference.
    
    Args:
        landmarks: Shape (33, 4) with [x, y, z, visibility]
        
    Returns:
        Normalized landmarks of shape (33, 4)
    """
    normalized = landmarks.copy()
    
    # Hip center
    left_hip = landmarks[config.LEFT_HIP, :3]
    right_hip = landmarks[config.RIGHT_HIP, :3]
    hip_center = (left_hip + right_hip) / 2.0
    
    # Torso length
    left_shoulder = landmarks[config.LEFT_SHOULDER, :3]
    right_shoulder = landmarks[config.RIGHT_SHOULDER, :3]
    shoulder_center = (left_shoulder + right_shoulder) / 2.0
    
    torso_length = np.linalg.norm(shoulder_center - hip_center)
    if torso_length < 0.01:
        torso_length = 0.5
    
    # Apply normalization
    normalized[:, :3] -= hip_center
    normalized[:, :3] /= torso_length
    
    return normalized


def flatten_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Flatten landmarks from (33, 4) to (132,) for LSTM input.
    
    Args:
        landmarks: Shape (33, 4)
        
    Returns:
        Flattened array of shape (132,)
    """
    return landmarks.flatten().astype(np.float32)


def load_model(
    model_path: Path,
    device: Optional[torch.device] = None
) -> torch.nn.Module:
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path: Path to .pth checkpoint file
        device: Device to load model to
        
    Returns:
        Loaded model in eval mode
    """
    from model import PatientSafetyLSTM
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model with saved config
    model_config = checkpoint.get('config', {})
    model = PatientSafetyLSTM(
        input_size=model_config.get('input_size', config.INPUT_SIZE),
        hidden_size_1=model_config.get('hidden_size_1', config.HIDDEN_SIZE_1),
        hidden_size_2=model_config.get('hidden_size_2', config.HIDDEN_SIZE_2),
        num_classes=model_config.get('num_classes', config.NUM_CLASSES)
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from {model_path}")
    print(f"Validation accuracy: {checkpoint.get('val_acc', 'N/A'):.1%}")
    
    return model


def get_class_color(class_idx: int) -> Tuple[int, int, int]:
    """
    Get BGR color for visualization based on class.
    
    Args:
        class_idx: Class index (0=Normal, 1=Fall, 2=Seizure)
        
    Returns:
        BGR color tuple
    """
    colors = {
        0: (0, 255, 0),    # Green for Normal
        1: (0, 0, 255),    # Red for Fall
        2: (0, 165, 255),  # Orange for Seizure
    }
    return colors.get(class_idx, (255, 255, 255))


def format_prediction(
    class_idx: int,
    confidence: float,
    latency_ms: float
) -> str:
    """
    Format prediction for display overlay.
    
    Args:
        class_idx: Predicted class index
        confidence: Prediction confidence (0-1)
        latency_ms: Inference latency in milliseconds
        
    Returns:
        Formatted string for display
    """
    class_name = config.CLASS_LABELS.get(class_idx, "Unknown")
    return f"{class_name}: {confidence*100:.0f}% ({latency_ms:.0f}ms)"


if __name__ == "__main__":
    # Test utilities
    print("Testing utilities...")
    
    # Test timer decorator
    @timer
    def slow_function():
        time.sleep(0.1)
        return "done"
    
    result = slow_function()
    
    # Test inference timer
    with InferenceTimer() as t:
        time.sleep(0.05)
    print(f"InferenceTimer test: {t.elapsed_ms:.2f}ms")
    
    # Test latency tracker
    tracker = LatencyTracker(window_size=10)
    for i in range(20):
        tracker.add(np.random.uniform(10, 50))
    print(tracker.summary())
    
    # Test normalization
    mock_landmarks = np.random.rand(33, 4).astype(np.float32)
    normalized = normalize_landmarks(mock_landmarks)
    print(f"Normalized landmarks shape: {normalized.shape}")
    
    # Test flattening
    flattened = flatten_landmarks(normalized)
    print(f"Flattened shape: {flattened.shape}")
    
    print("\n✓ All utility tests passed!")
