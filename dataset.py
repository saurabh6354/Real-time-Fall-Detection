"""
dataset.py - PyTorch Dataset for Skeletal Sequence Data

This module provides:
1. SkeletonDataset: Loads preprocessed .npy files for training
2. MockDataset: Generates synthetic data for testing without real videos
3. Data augmentation utilities for improved generalization

The dataset handles the conversion from (frames, 33, 4) landmark arrays
to flattened (frames, 132) feature vectors expected by the LSTM.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import random
import sys

sys.path.insert(0, str(Path(__file__).parent))
import config


class SkeletonDataset(Dataset):
    """
    PyTorch Dataset for loading preprocessed skeletal sequences.
    
    Expects .npy files with shape (num_frames, 33, 4) in the processed
    data directories, organized by class label.
    
    Attributes:
        sequences: List of (sequence_tensor, label) tuples
        transform: Optional augmentation function
    """
    
    def __init__(self, 
                 data_dir: Path = config.PROCESSED_DIR,
                 sequence_length: int = config.SEQUENCE_LENGTH,
                 stride: int = config.STRIDE,
                 transform: Optional[callable] = None):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Root directory containing class subdirectories
            sequence_length: Number of frames per sequence
            stride: Sliding window stride for creating sequences
            transform: Optional augmentation function
        """
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.stride = stride
        self.transform = transform
        
        self.sequences: List[Tuple[np.ndarray, int]] = []
        self._load_data()
    
    def _load_data(self):
        """Load all .npy files and create fixed-length sequences."""
        # Expected structure:
        # data/processed/
        #   ├── falls/
        #   │   ├── video1_Fall.npy
        #   │   └── video2_Fall.npy
        #   └── adls/
        #       ├── video3_Normal.npy
        #       └── video4_Normal.npy
        
        label_dirs = {
            "falls": config.LABEL_TO_IDX.get("Fall", 1),
            "adls": config.LABEL_TO_IDX.get("Normal", 0),
            "seizures": config.LABEL_TO_IDX.get("Seizure", 2)
        }
        
        for subdir_name, label_idx in label_dirs.items():
            subdir = self.data_dir / subdir_name
            if not subdir.exists():
                continue
            
            for npy_file in subdir.glob("*.npy"):
                try:
                    # Load landmarks: (num_frames, 33, 4)
                    landmarks = np.load(npy_file)
                    
                    # Flatten to (num_frames, 132) for LSTM
                    flattened = landmarks.reshape(landmarks.shape[0], -1)
                    
                    # Create sliding window sequences
                    sequences = self._create_sequences(flattened)
                    
                    for seq in sequences:
                        self.sequences.append((seq, label_idx))
                        
                except Exception as e:
                    print(f"Warning: Failed to load {npy_file}: {e}")
        
        print(f"Loaded {len(self.sequences)} sequences from {self.data_dir}")
    
    def _create_sequences(self, data: np.ndarray) -> List[np.ndarray]:
        """Create fixed-length overlapping sequences from variable-length data."""
        num_frames = data.shape[0]
        sequences = []
        
        if num_frames < self.sequence_length:
            # Pad short sequences
            padding = np.zeros((self.sequence_length - num_frames, data.shape[1]), 
                              dtype=np.float32)
            padded = np.vstack([data, padding])
            sequences.append(padded)
        else:
            # Sliding window
            for start in range(0, num_frames - self.sequence_length + 1, self.stride):
                end = start + self.sequence_length
                sequences.append(data[start:end])
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sequence, label = self.sequences[idx]
        
        # Convert to tensor
        sequence_tensor = torch.from_numpy(sequence.astype(np.float32))
        
        # Apply augmentation if specified
        if self.transform:
            sequence_tensor = self.transform(sequence_tensor)
        
        return sequence_tensor, label


class MockDataset(Dataset):
    """
    Synthetic dataset for testing the model architecture.
    
    Generates realistic-looking skeletal sequences with distinct patterns
    for each class:
    - Normal: Smooth, low-variance movements
    - Fall: Sudden downward acceleration followed by stillness
    - Seizure: High-frequency oscillations with high variance
    
    This allows testing the full training pipeline without real data.
    """
    
    def __init__(self,
                 num_samples: int = 500,
                 sequence_length: int = config.SEQUENCE_LENGTH,
                 num_features: int = config.INPUT_SIZE,
                 random_seed: int = config.RANDOM_SEED):
        """
        Initialize mock dataset.
        
        Args:
            num_samples: Total number of sequences to generate
            sequence_length: Frames per sequence
            num_features: Features per frame (132 for skeleton)
            random_seed: Seed for reproducibility
        """
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.num_features = num_features
        
        np.random.seed(random_seed)
        self.sequences, self.labels = self._generate_data()
    
    def _generate_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic movement patterns for each class."""
        sequences = []
        labels = []
        
        samples_per_class = self.num_samples // 3
        
        # Class 0: Normal activities
        # Characteristics: Smooth, low-frequency movements
        for _ in range(samples_per_class):
            # Base position with slow drift
            base = np.random.randn(1, self.num_features) * 0.1
            
            # Smooth movement using cumulative sum of small steps
            steps = np.random.randn(self.sequence_length, self.num_features) * 0.02
            smooth_movement = np.cumsum(steps, axis=0)
            
            sequence = base + smooth_movement
            sequences.append(sequence.astype(np.float32))
            labels.append(0)
        
        # Class 1: Falls
        # Characteristics: Sudden acceleration, then stillness
        for _ in range(samples_per_class):
            sequence = np.zeros((self.sequence_length, self.num_features), dtype=np.float32)
            
            # Fall starts at random point in sequence
            fall_start = np.random.randint(5, self.sequence_length - 10)
            fall_duration = np.random.randint(3, 8)
            
            # Pre-fall: normal movement
            sequence[:fall_start] = np.cumsum(
                np.random.randn(fall_start, self.num_features) * 0.02, axis=0
            )
            
            # During fall: rapid downward acceleration
            # Focus on Y-coordinates (every 4th feature starting at index 1)
            for t in range(fall_start, min(fall_start + fall_duration, self.sequence_length)):
                sequence[t] = sequence[t-1].copy()
                # Simulate downward movement (negative Y velocity)
                y_indices = np.arange(1, self.num_features, 4)
                sequence[t, y_indices] -= 0.1 * (t - fall_start + 1)
            
            # Post-fall: stillness (small noise)
            for t in range(fall_start + fall_duration, self.sequence_length):
                sequence[t] = sequence[fall_start + fall_duration - 1] + \
                             np.random.randn(self.num_features) * 0.005
            
            sequences.append(sequence)
            labels.append(1)
        
        # Class 2: Seizures
        # Characteristics: High-frequency oscillations, high variance
        for _ in range(samples_per_class + (self.num_samples % 3)):
            # High-frequency oscillation
            t = np.linspace(0, 4 * np.pi, self.sequence_length)
            
            # Random frequency for each landmark
            freqs = np.random.uniform(1, 3, self.num_features)
            phases = np.random.uniform(0, 2 * np.pi, self.num_features)
            amplitudes = np.random.uniform(0.1, 0.3, self.num_features)
            
            sequence = np.zeros((self.sequence_length, self.num_features), dtype=np.float32)
            for f in range(self.num_features):
                sequence[:, f] = amplitudes[f] * np.sin(freqs[f] * t + phases[f])
            
            # Add noise
            sequence += np.random.randn(self.sequence_length, self.num_features) * 0.05
            
            sequences.append(sequence)
            labels.append(2)
        
        return np.array(sequences), np.array(labels)
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sequence = torch.from_numpy(self.sequences[idx])
        label = int(self.labels[idx])
        return sequence, label


# ============================================================================
# DATA AUGMENTATION
# ============================================================================

class SkeletonAugmentation:
    """
    Data augmentation transforms for skeletal sequences.
    
    These augmentations preserve the semantic meaning of movements
    while increasing data diversity.
    """
    
    def __init__(self, 
                 noise_std: float = 0.01,
                 time_warp_prob: float = 0.3,
                 scale_range: Tuple[float, float] = (0.9, 1.1)):
        """
        Initialize augmentation parameters.
        
        Args:
            noise_std: Standard deviation of Gaussian noise
            time_warp_prob: Probability of applying time warping
            scale_range: Range for random scaling
        """
        self.noise_std = noise_std
        self.time_warp_prob = time_warp_prob
        self.scale_range = scale_range
    
    def __call__(self, sequence: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations to a sequence."""
        # Add Gaussian noise
        if self.noise_std > 0:
            noise = torch.randn_like(sequence) * self.noise_std
            sequence = sequence + noise
        
        # Random scaling
        if self.scale_range != (1.0, 1.0):
            scale = random.uniform(*self.scale_range)
            sequence = sequence * scale
        
        return sequence


def get_data_loaders(
    data_dir: Path = config.PROCESSED_DIR,
    batch_size: int = config.BATCH_SIZE,
    train_split: float = config.TRAIN_SPLIT,
    use_mock: bool = False,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.
    
    Args:
        data_dir: Directory with processed data
        batch_size: Batch size
        train_split: Fraction of data for training
        use_mock: Whether to use mock data (for testing)
        num_workers: Number of data loading workers
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    if use_mock:
        dataset = MockDataset(num_samples=500)
    else:
        dataset = SkeletonDataset(data_dir=data_dir)
    
    # Split into train/val
    total_size = len(dataset)
    train_size = int(total_size * train_split)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.RANDOM_SEED)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Train samples: {train_size}, Validation samples: {val_size}")
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test the mock dataset
    print("Testing MockDataset...")
    mock_ds = MockDataset(num_samples=100)
    print(f"Dataset size: {len(mock_ds)}")
    
    sample, label = mock_ds[0]
    print(f"Sample shape: {sample.shape}")
    print(f"Label: {label} ({config.CLASS_LABELS[label]})")
    
    # Test data loaders
    print("\nTesting data loaders with mock data...")
    train_loader, val_loader = get_data_loaders(use_mock=True, batch_size=16)
    
    for batch_x, batch_y in train_loader:
        print(f"Batch X shape: {batch_x.shape}")
        print(f"Batch Y shape: {batch_y.shape}")
        print(f"Labels in batch: {batch_y.tolist()}")
        break
    
    print("\n✓ Dataset tests passed!")
