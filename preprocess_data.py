"""
preprocess_data.py - Privacy-Preserving Data Pipeline

This script converts raw video files into anonymous skeletal landmark sequences.
It is the CRITICAL privacy layer of our system - raw video is NEVER stored.

Pipeline:
    Video → MediaPipe Pose → 33 Landmarks (x,y,z,vis) → Normalize → Save as .npy

Normalization Strategy:
    1. Hip-Center Translation: All coordinates become relative to the midpoint
       of left and right hips. This makes the data position-invariant.
    2. Torso Scaling: Divide by torso length (hip-to-shoulder distance) to make
       the data height-invariant and camera-distance-invariant.

Usage:
    python preprocess_data.py                    # Process default data directories
    python preprocess_data.py --input path/to/videos --output path/to/output
    python preprocess_data.py --visualize        # Show pose overlay while processing
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional, Tuple, List
import time

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))
import config


class PoseExtractor:
    """
    Extracts and normalizes skeletal landmarks from video frames using MediaPipe.
    
    This class is the core of our privacy-preserving approach. It transforms
    pixel-based video into abstract numerical coordinates that cannot be used
    to reconstruct a person's appearance.
    
    Attributes:
        mp_pose: MediaPipe Pose solution instance
        pose: Configured pose detector
        min_visibility: Threshold for considering a landmark as "detected"
    """
    
    def __init__(self, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 min_visibility: float = config.MIN_VISIBILITY):
        """
        Initialize the pose extractor with MediaPipe.
        
        Args:
            min_detection_confidence: Minimum confidence for initial pose detection
            min_tracking_confidence: Minimum confidence for pose tracking across frames
            min_visibility: Minimum visibility score to consider landmark valid
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,  # Video mode for better tracking
            model_complexity=1,       # 0=lite, 1=full, 2=heavy (balance accuracy/speed)
            smooth_landmarks=True,    # Temporal smoothing for stability
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.min_visibility = min_visibility
    
    def extract_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract 33 pose landmarks from a single frame.
        
        Args:
            frame: BGR image from OpenCV (height, width, 3)
            
        Returns:
            np.ndarray of shape (33, 4) with [x, y, z, visibility] per landmark,
            or None if no pose detected.
        """
        # MediaPipe expects RGB, OpenCV gives BGR
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame - this is where the magic happens
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks is None:
            return None
        
        # Convert landmarks to numpy array
        landmarks = np.array([
            [lm.x, lm.y, lm.z, lm.visibility]
            for lm in results.pose_landmarks.landmark
        ], dtype=np.float32)
        
        return landmarks  # Shape: (33, 4)
    
    def normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Apply hip-center translation and torso scaling normalization.
        
        Why this normalization?
        1. Hip-center translation makes poses position-invariant. A person at
           the left of the frame has the same normalized pose as one on the right.
        2. Torso scaling makes poses scale-invariant. A tall person far from the
           camera has the same normalized pose as a short person close to it.
        
        Args:
            landmarks: Raw landmarks of shape (33, 4)
            
        Returns:
            Normalized landmarks of shape (33, 4). Visibility is preserved unchanged.
        """
        normalized = landmarks.copy()
        
        # Step 1: Compute hip center (midpoint of left and right hips)
        left_hip = landmarks[config.LEFT_HIP, :3]   # x, y, z only
        right_hip = landmarks[config.RIGHT_HIP, :3]
        hip_center = (left_hip + right_hip) / 2.0
        
        # Step 2: Compute torso length for scale normalization
        # Torso = distance from hip center to shoulder center
        left_shoulder = landmarks[config.LEFT_SHOULDER, :3]
        right_shoulder = landmarks[config.RIGHT_SHOULDER, :3]
        shoulder_center = (left_shoulder + right_shoulder) / 2.0
        
        torso_length = np.linalg.norm(shoulder_center - hip_center)
        
        # Avoid division by zero (if torso length is too small, use default scale)
        if torso_length < 0.01:
            torso_length = 0.5  # Reasonable default in normalized coordinates
        
        # Step 3: Apply translation (subtract hip center)
        normalized[:, :3] -= hip_center
        
        # Step 4: Apply scaling (divide by torso length)
        normalized[:, :3] /= torso_length
        
        # Visibility (column 4) remains unchanged
        return normalized
    
    def draw_pose(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Draw skeleton overlay on frame for visualization/debugging.
        
        Args:
            frame: Original BGR image
            landmarks: Optional landmarks for overlay
            
        Returns:
            Frame with skeleton overlay drawn
        """
        annotated = frame.copy()
        h, w = frame.shape[:2]
        
        # Convert back to MediaPipe landmark format for drawing
        mp_landmarks = self.mp_pose.PoseLandmark
        connections = self.mp_pose.POSE_CONNECTIONS
        
        # Draw landmarks as circles
        for i, (x, y, z, vis) in enumerate(landmarks):
            if vis >= self.min_visibility:
                cx, cy = int(x * w), int(y * h)
                cv2.circle(annotated, (cx, cy), 5, (0, 255, 0), -1)
        
        # Draw connections
        for connection in connections:
            start_idx, end_idx = connection
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            
            if start[3] >= self.min_visibility and end[3] >= self.min_visibility:
                start_point = (int(start[0] * w), int(start[1] * h))
                end_point = (int(end[0] * w), int(end[1] * h))
                cv2.line(annotated, start_point, end_point, (255, 0, 0), 2)
        
        return annotated
    
    def close(self):
        """Release MediaPipe resources."""
        self.pose.close()


def process_video(video_path: Path, 
                  extractor: PoseExtractor,
                  visualize: bool = False) -> Tuple[Optional[np.ndarray], dict]:
    """
    Process a single video file and extract normalized skeletal sequences.
    
    Args:
        video_path: Path to video file
        extractor: PoseExtractor instance
        visualize: Whether to show real-time visualization
        
    Returns:
        Tuple of:
            - np.ndarray of shape (num_frames, 33, 4) or None if failed
            - dict with metadata (fps, total_frames, valid_frames, etc.)
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"  ✗ Failed to open: {video_path}")
        return None, {"error": "Failed to open video"}
    
    # Get video metadata
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    landmarks_sequence = []
    valid_frames = 0
    
    # Create progress bar for this video
    pbar = tqdm(total=total_frames, desc=f"  {video_path.name}", leave=False)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract landmarks
        landmarks = extractor.extract_landmarks(frame)
        
        if landmarks is not None:
            # Normalize before storing (privacy + invariance)
            normalized = extractor.normalize_landmarks(landmarks)
            landmarks_sequence.append(normalized)
            valid_frames += 1
            
            if visualize:
                vis_frame = extractor.draw_pose(frame, landmarks)
                cv2.imshow("Pose Extraction", vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            # No pose detected - append zeros as placeholder
            # This preserves temporal alignment
            landmarks_sequence.append(np.zeros((config.NUM_LANDMARKS, config.LANDMARK_DIMS), 
                                               dtype=np.float32))
        
        pbar.update(1)
    
    pbar.close()
    cap.release()
    
    if visualize:
        cv2.destroyAllWindows()
    
    metadata = {
        "source_video": str(video_path),
        "fps": fps,
        "resolution": f"{width}x{height}",
        "total_frames": total_frames,
        "valid_frames": valid_frames,
        "detection_rate": valid_frames / total_frames if total_frames > 0 else 0
    }
    
    if len(landmarks_sequence) == 0:
        return None, metadata
    
    # Stack into single array: (num_frames, 33, 4)
    sequence_array = np.stack(landmarks_sequence, axis=0)
    
    return sequence_array, metadata


def process_directory(input_dir: Path, 
                      output_dir: Path,
                      label: str,
                      extractor: PoseExtractor,
                      visualize: bool = False) -> List[dict]:
    """
    Process all videos in a directory.
    
    Args:
        input_dir: Directory containing video files
        output_dir: Directory to save .npy files
        label: Class label for all videos in this directory
        extractor: PoseExtractor instance
        visualize: Whether to show real-time visualization
        
    Returns:
        List of metadata dicts for each processed video
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all video files
    video_files = [
        f for f in input_dir.iterdir() 
        if f.suffix.lower() in config.VIDEO_EXTENSIONS
    ]
    
    if not video_files:
        print(f"  ⚠ No video files found in {input_dir}")
        return []
    
    print(f"\nProcessing {len(video_files)} videos from: {input_dir}")
    print(f"Label: {label}")
    
    all_metadata = []
    
    for video_path in video_files:
        sequence, metadata = process_video(video_path, extractor, visualize)
        
        if sequence is not None:
            # Save as .npy file
            output_name = f"{video_path.stem}_{label}.npy"
            output_path = output_dir / output_name
            np.save(output_path, sequence)
            
            metadata["output_file"] = str(output_path)
            metadata["label"] = label
            metadata["label_idx"] = config.LABEL_TO_IDX.get(label, -1)
            metadata["shape"] = sequence.shape
            
            print(f"  ✓ {video_path.name} → {output_name} | Shape: {sequence.shape}")
        else:
            metadata["output_file"] = None
            metadata["label"] = label
            print(f"  ✗ {video_path.name} - No valid poses extracted")
        
        all_metadata.append(metadata)
    
    return all_metadata


def save_metadata_csv(metadata_list: List[dict], output_path: Path):
    """
    Save processing metadata to CSV for later analysis.
    
    Args:
        metadata_list: List of metadata dicts from processing
        output_path: Path to save CSV file
    """
    if not metadata_list:
        return
    
    fieldnames = [
        "source_video", "output_file", "label", "label_idx",
        "fps", "resolution", "total_frames", "valid_frames", 
        "detection_rate", "shape"
    ]
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(metadata_list)
    
    print(f"\nMetadata saved to: {output_path}")


def create_sequences(landmarks_array: np.ndarray, 
                     sequence_length: int = config.SEQUENCE_LENGTH,
                     stride: int = config.STRIDE) -> np.ndarray:
    """
    Convert a long landmark sequence into fixed-length overlapping windows.
    
    This is crucial for training - the LSTM expects fixed-length inputs.
    
    Args:
        landmarks_array: Shape (total_frames, 33, 4)
        sequence_length: Number of frames per sequence
        stride: Step size between sequences
        
    Returns:
        np.ndarray of shape (num_sequences, sequence_length, 33, 4)
    """
    total_frames = landmarks_array.shape[0]
    
    if total_frames < sequence_length:
        # Pad short videos with zeros
        padding = np.zeros((sequence_length - total_frames, 
                           config.NUM_LANDMARKS, 
                           config.LANDMARK_DIMS), dtype=np.float32)
        padded = np.concatenate([landmarks_array, padding], axis=0)
        return padded[np.newaxis, ...]  # Shape: (1, seq_len, 33, 4)
    
    sequences = []
    for start in range(0, total_frames - sequence_length + 1, stride):
        end = start + sequence_length
        sequences.append(landmarks_array[start:end])
    
    return np.stack(sequences, axis=0)


def main():
    """Main entry point for the preprocessing pipeline."""
    parser = argparse.ArgumentParser(
        description="Convert videos to privacy-preserving skeletal sequences"
    )
    parser.add_argument(
        "--falls-dir", type=Path, default=config.FALLS_DIR,
        help="Directory containing fall videos"
    )
    parser.add_argument(
        "--adls-dir", type=Path, default=config.ADLS_DIR,
        help="Directory containing normal activity (ADL) videos"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=config.PROCESSED_DIR,
        help="Output directory for .npy files"
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Show pose overlay while processing"
    )
    parser.add_argument(
        "--create-sequences", action="store_true",
        help="Also create fixed-length sequence files for training"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  PRIVACY-PRESERVING DATA PIPELINE")
    print("  Converting videos → skeletal landmarks")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Landmarks per frame: {config.NUM_LANDMARKS}")
    print(f"  Features per landmark: {config.LANDMARK_DIMS}")
    print(f"  Total features per frame: {config.FEATURES_PER_FRAME}")
    print(f"  Sequence length: {config.SEQUENCE_LENGTH} frames")
    
    # Initialize pose extractor
    extractor = PoseExtractor()
    all_metadata = []
    
    start_time = time.time()
    
    # Process falls directory
    if args.falls_dir.exists():
        fall_metadata = process_directory(
            args.falls_dir, 
            args.output_dir / "falls",
            label="Fall",
            extractor=extractor,
            visualize=args.visualize
        )
        all_metadata.extend(fall_metadata)
    else:
        print(f"\n⚠ Falls directory not found: {args.falls_dir}")
        print("  Create this directory and add fall videos to process them.")
    
    # Process ADLs (normal activities) directory
    if args.adls_dir.exists():
        adl_metadata = process_directory(
            args.adls_dir,
            args.output_dir / "adls",
            label="Normal",
            extractor=extractor,
            visualize=args.visualize
        )
        all_metadata.extend(adl_metadata)
    else:
        print(f"\n⚠ ADLs directory not found: {args.adls_dir}")
        print("  Create this directory and add normal activity videos.")
    
    # Save metadata
    if all_metadata:
        save_metadata_csv(all_metadata, args.output_dir / "processing_metadata.csv")
    
    # Cleanup
    extractor.close()
    
    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"  Processing complete in {elapsed:.2f} seconds")
    print(f"  Total videos processed: {len(all_metadata)}")
    print(f"  Output directory: {args.output_dir}")
    print(f"{'=' * 60}")
    
    # Privacy confirmation
    print("\n🔒 PRIVACY GUARANTEE:")
    print("   No raw video frames were stored.")
    print("   Only numerical landmark coordinates are saved.")
    print("   Original videos remain untouched.\n")


if __name__ == "__main__":
    main()
