"""
Configuration constants for the Patient Safety System.

This file centralizes all hyperparameters and settings to ensure
consistency across preprocessing, training, and inference.
"""

from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
FALLS_DIR = DATA_DIR / "falls"
ADLS_DIR = DATA_DIR / "adls"  # Activities of Daily Living (normal behavior)
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"

# =============================================================================
# MEDIAPIPE POSE CONFIGURATION
# =============================================================================
NUM_LANDMARKS = 33  # MediaPipe Pose has 33 body landmarks
LANDMARK_DIMS = 4   # x, y, z, visibility
FEATURES_PER_FRAME = NUM_LANDMARKS * LANDMARK_DIMS  # 132 features

# Key landmark indices for normalization
# Reference: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

# =============================================================================
# SEQUENCE CONFIGURATION
# =============================================================================
SEQUENCE_LENGTH = 30  # ~1 second at 30 FPS
STRIDE = 15           # Overlap between sequences (50% overlap for training)
MIN_VISIBILITY = 0.5  # Minimum visibility threshold for valid landmarks

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================
INPUT_SIZE = FEATURES_PER_FRAME  # 132
HIDDEN_SIZE_1 = 256
HIDDEN_SIZE_2 = 128
NUM_LSTM_LAYERS = 2
DROPOUT = 0.3
NUM_CLASSES = 3  # Fall, Seizure, Normal

# Class labels mapping
CLASS_LABELS = {
    0: "Normal",
    1: "Fall",
    2: "Seizure"
}
LABEL_TO_IDX = {v: k for k, v in CLASS_LABELS.items()}

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
TRAIN_SPLIT = 0.8
RANDOM_SEED = 42

# =============================================================================
# INFERENCE CONFIGURATION
# =============================================================================
CONFIDENCE_THRESHOLD = 0.8  # Alert only if confidence > 80%
MAX_INFERENCE_TIME_MS = 150  # Target latency constraint
WEBCAM_FPS = 30

# Supported video extensions for preprocessing
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
