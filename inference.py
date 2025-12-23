"""
inference.py - Real-time Fall & Seizure Detection Engine

This script provides real-time inference with <150ms latency target.

Architecture:
    ┌─────────────────────────────────────────────────────┐
    │                    MAIN THREAD                       │
    │  ┌─────────┐    ┌──────────┐    ┌───────────────┐  │
    │  │ Webcam  │───►│ MediaPipe│───►│ Sliding Window│  │
    │  │ Capture │    │   Pose   │    │    Buffer     │  │
    │  └─────────┘    └──────────┘    └───────┬───────┘  │
    │                                         │          │
    │                                         ▼          │
    │  ┌─────────┐    ┌──────────┐    ┌───────────────┐  │
    │  │ Display │◄───│  Alert   │◄───│ LSTM Inference│  │
    │  │ Output  │    │  Logic   │    │   (Threaded)  │  │
    │  └─────────┘    └──────────┘    └───────────────┘  │
    └─────────────────────────────────────────────────────┘

Optimization Strategies:
1. Sliding window buffer (deque) for O(1) append/pop
2. Model inference on separate thread to prevent frame drops
3. torch.no_grad() + model.eval() for inference mode
4. Frame skipping if processing falls behind

Usage:
    python inference.py                     # Use webcam
    python inference.py --video path.mp4   # Use video file
    python inference.py --no-display       # Headless mode (for testing)
"""

import argparse
import threading
import queue
import time
from collections import deque
from pathlib import Path
from typing import Optional, Tuple, Deque
import sys

import cv2
import numpy as np
import torch

# Handle MediaPipe import (not available in all environments)
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: MediaPipe not available. Using mock pose estimation.")

sys.path.insert(0, str(Path(__file__).parent))
import config
from model import PatientSafetyLSTM
from utils import (
    normalize_landmarks, 
    flatten_landmarks, 
    InferenceTimer,
    LatencyTracker,
    get_class_color,
    format_prediction
)


class PoseEstimator:
    """
    Wrapper for MediaPipe Pose estimation.
    
    Provides a consistent interface for extracting and normalizing
    skeletal landmarks from video frames.
    """
    
    def __init__(self):
        if MEDIAPIPE_AVAILABLE:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.mp_drawing = mp.solutions.drawing_utils
        else:
            self.pose = None
    
    def extract(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract normalized landmarks from a frame.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            Flattened normalized landmarks (132,) or None if no pose
        """
        if not MEDIAPIPE_AVAILABLE:
            # Return mock data for testing
            return np.random.randn(config.FEATURES_PER_FRAME).astype(np.float32)
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        
        if results.pose_landmarks is None:
            return None
        
        # Convert to numpy array
        landmarks = np.array([
            [lm.x, lm.y, lm.z, lm.visibility]
            for lm in results.pose_landmarks.landmark
        ], dtype=np.float32)
        
        # Normalize and flatten
        normalized = normalize_landmarks(landmarks)
        flattened = flatten_landmarks(normalized)
        
        return flattened
    
    def draw_skeleton(self, frame: np.ndarray) -> np.ndarray:
        """Draw skeleton overlay on frame (for visualization)."""
        if not MEDIAPIPE_AVAILABLE:
            return frame
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
        
        return frame
    
    def close(self):
        """Release resources."""
        if MEDIAPIPE_AVAILABLE and self.pose:
            self.pose.close()


class InferenceEngine:
    """
    Real-time inference engine with sliding window buffer.
    
    Manages the sequence buffer and runs LSTM inference when
    the buffer is full. Designed for <150ms latency.
    """
    
    def __init__(self, 
                 model: torch.nn.Module,
                 device: torch.device,
                 sequence_length: int = config.SEQUENCE_LENGTH,
                 confidence_threshold: float = config.CONFIDENCE_THRESHOLD):
        """
        Initialize the inference engine.
        
        Args:
            model: Trained LSTM model
            device: Device for inference
            sequence_length: Number of frames in sliding window
            confidence_threshold: Minimum confidence for alerts
        """
        self.model = model
        self.device = device
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        
        # Sliding window buffer - O(1) append and pop
        self.buffer: Deque[np.ndarray] = deque(maxlen=sequence_length)
        
        # Latest prediction (updated by inference)
        self.latest_prediction: Optional[Tuple[int, float]] = None
        self.latest_latency: float = 0.0
        
        # Latency tracking
        self.latency_tracker = LatencyTracker(window_size=100)
        
        # Thread-safe prediction updates
        self.lock = threading.Lock()
    
    def add_frame(self, features: np.ndarray) -> bool:
        """
        Add a frame to the buffer.
        
        Args:
            features: Flattened landmarks (132,)
            
        Returns:
            True if buffer is full and ready for inference
        """
        self.buffer.append(features)
        return len(self.buffer) >= self.sequence_length
    
    def run_inference(self) -> Optional[Tuple[int, float, float]]:
        """
        Run LSTM inference on the current buffer.
        
        Returns:
            Tuple of (class_idx, confidence, latency_ms) or None
        """
        if len(self.buffer) < self.sequence_length:
            return None
        
        with InferenceTimer() as timer:
            # Convert buffer to tensor
            # Shape: (1, seq_len, features)
            sequence = np.stack(list(self.buffer), axis=0)
            tensor = torch.from_numpy(sequence).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                predictions, confidence = self.model.predict(tensor)
            
            class_idx = predictions[0].item()
            conf_score = confidence[0].item()
        
        # Update latency tracker
        self.latency_tracker.add(timer.elapsed_ms)
        
        # Thread-safe update of latest prediction
        with self.lock:
            self.latest_prediction = (class_idx, conf_score)
            self.latest_latency = timer.elapsed_ms
        
        return class_idx, conf_score, timer.elapsed_ms
    
    def get_latest_prediction(self) -> Optional[Tuple[int, float, float]]:
        """Get the latest prediction in a thread-safe manner."""
        with self.lock:
            if self.latest_prediction is None:
                return None
            return (*self.latest_prediction, self.latest_latency)
    
    def should_alert(self) -> bool:
        """Check if an alert should be triggered."""
        pred = self.get_latest_prediction()
        if pred is None:
            return False
        
        class_idx, confidence, _ = pred
        
        # Alert only for Fall or Seizure with high confidence
        return (class_idx in [1, 2]) and (confidence >= self.confidence_threshold)


class RealTimeDetector:
    """
    Main class for real-time fall and seizure detection.
    
    Orchestrates video capture, pose estimation, and inference
    with proper threading and latency management.
    """
    
    def __init__(self,
                 model_path: Optional[Path] = None,
                 video_source: int = 0,
                 display: bool = True):
        """
        Initialize the detector.
        
        Args:
            model_path: Path to trained model checkpoint
            video_source: Webcam index or video file path
            display: Whether to show visualization
        """
        self.video_source = video_source
        self.display = display
        
        # Device selection
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load or create model
        if model_path and Path(model_path).exists():
            self.model = self._load_model(model_path)
        else:
            print("No model found. Creating untrained model for testing.")
            self.model = PatientSafetyLSTM().to(self.device)
            self.model.eval()
        
        # Initialize components
        self.pose_estimator = PoseEstimator()
        self.inference_engine = InferenceEngine(self.model, self.device)
        
        # Threading for non-blocking inference
        self.inference_queue = queue.Queue(maxsize=1)
        self.running = False
    
    def _load_model(self, model_path: Path) -> torch.nn.Module:
        """Load trained model from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        model = PatientSafetyLSTM()
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        print(f"Loaded model from {model_path}")
        if 'val_acc' in checkpoint:
            print(f"Validation accuracy: {checkpoint['val_acc']:.1%}")
        
        return model
    
    def _inference_thread(self):
        """Background thread for running inference."""
        while self.running:
            try:
                # Wait for new frame data
                _ = self.inference_queue.get(timeout=0.1)
                
                # Run inference
                self.inference_engine.run_inference()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Inference error: {e}")
    
    def _draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw prediction overlay on frame."""
        pred = self.inference_engine.get_latest_prediction()
        
        h, w = frame.shape[:2]
        
        if pred is not None:
            class_idx, confidence, latency = pred
            
            # Get color and text
            color = get_class_color(class_idx)
            text = format_prediction(class_idx, confidence, latency)
            
            # Draw prediction text
            cv2.putText(frame, text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Draw alert if triggered
            if self.inference_engine.should_alert():
                alert_text = "⚠ ALERT: " + config.CLASS_LABELS[class_idx].upper()
                
                # Flash red background
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, h-80), (w, h), (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
                
                cv2.putText(frame, alert_text, (w//4, h-30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        else:
            # Buffer filling
            buffer_status = f"Buffer: {len(self.inference_engine.buffer)}/{config.SEQUENCE_LENGTH}"
            cv2.putText(frame, buffer_status, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw latency stats
        stats = self.inference_engine.latency_tracker.summary()
        cv2.putText(frame, stats, (10, h-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def run(self):
        """Main detection loop."""
        # Open video source
        cap = cv2.VideoCapture(self.video_source)
        
        if not cap.isOpened():
            print(f"Failed to open video source: {self.video_source}")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        print(f"Video FPS: {fps}")
        
        # Start inference thread
        self.running = True
        inference_thread = threading.Thread(target=self._inference_thread, daemon=True)
        inference_thread.start()
        
        print("\n" + "=" * 50)
        print("  REAL-TIME PATIENT SAFETY MONITOR")
        print("  Press 'q' to quit")
        print("=" * 50 + "\n")
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    # Loop video files
                    if isinstance(self.video_source, str):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break
                
                frame_count += 1
                
                # Extract pose features
                features = self.pose_estimator.extract(frame)
                
                if features is not None:
                    # Add to buffer
                    buffer_ready = self.inference_engine.add_frame(features)
                    
                    # Trigger inference (non-blocking)
                    if buffer_ready:
                        try:
                            self.inference_queue.put_nowait(True)
                        except queue.Full:
                            pass  # Skip if inference is busy
                
                # Draw visualization
                if self.display:
                    # Draw skeleton
                    frame = self.pose_estimator.draw_skeleton(frame)
                    
                    # Draw prediction overlay
                    frame = self._draw_overlay(frame)
                    
                    # Calculate actual FPS
                    elapsed = time.time() - start_time
                    actual_fps = frame_count / elapsed if elapsed > 0 else 0
                    cv2.putText(frame, f"FPS: {actual_fps:.1f}", (frame.shape[1]-100, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    cv2.imshow("Patient Safety Monitor", frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
        
        finally:
            # Cleanup
            self.running = False
            inference_thread.join(timeout=1)
            cap.release()
            cv2.destroyAllWindows()
            self.pose_estimator.close()
            
            # Print final stats
            elapsed = time.time() - start_time
            print(f"\nProcessed {frame_count} frames in {elapsed:.1f}s")
            print(f"Average FPS: {frame_count/elapsed:.1f}")
            print(self.inference_engine.latency_tracker.summary())


def benchmark_inference(model_path: Optional[Path] = None, num_iterations: int = 100):
    """
    Benchmark inference latency without video capture.
    
    This isolates the LSTM inference time for accurate measurement.
    """
    print("\n" + "=" * 50)
    print("  INFERENCE LATENCY BENCHMARK")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create model
    model = PatientSafetyLSTM().to(device)
    model.eval()
    
    if model_path and Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded weights from {model_path}")
    
    # Create dummy input
    dummy_input = torch.randn(1, config.SEQUENCE_LENGTH, config.INPUT_SIZE).to(device)
    
    # Warmup
    print("\nWarming up...")
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
    
    # Benchmark
    print(f"\nRunning {num_iterations} inference iterations...")
    latencies = []
    
    for _ in range(num_iterations):
        with InferenceTimer() as timer:
            with torch.no_grad():
                _ = model(dummy_input)
        latencies.append(timer.elapsed_ms)
    
    latencies = np.array(latencies)
    
    print(f"\nResults:")
    print(f"  Mean latency:   {np.mean(latencies):.2f} ms")
    print(f"  Std deviation:  {np.std(latencies):.2f} ms")
    print(f"  Min latency:    {np.min(latencies):.2f} ms")
    print(f"  Max latency:    {np.max(latencies):.2f} ms")
    print(f"  95th percentile:{np.percentile(latencies, 95):.2f} ms")
    
    # Check against target
    target = config.MAX_INFERENCE_TIME_MS
    if np.mean(latencies) < target:
        print(f"\n✓ PASS: Mean latency ({np.mean(latencies):.1f}ms) < target ({target}ms)")
    else:
        print(f"\n✗ FAIL: Mean latency ({np.mean(latencies):.1f}ms) >= target ({target}ms)")


def main():
    parser = argparse.ArgumentParser(description="Real-time Fall & Seizure Detection")
    parser.add_argument(
        '--model', type=Path, default=config.MODELS_DIR / "best_model.pth",
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--video', type=str, default="0",
        help='Video source: webcam index (0) or video file path'
    )
    parser.add_argument(
        '--no-display', action='store_true',
        help='Run without visualization (for headless systems)'
    )
    parser.add_argument(
        '--benchmark', action='store_true',
        help='Run inference latency benchmark instead of live detection'
    )
    
    args = parser.parse_args()
    
    if args.benchmark:
        benchmark_inference(args.model)
        return
    
    # Parse video source
    try:
        video_source = int(args.video)
    except ValueError:
        video_source = args.video
    
    # Create and run detector
    detector = RealTimeDetector(
        model_path=args.model,
        video_source=video_source,
        display=not args.no_display
    )
    
    detector.run()


if __name__ == "__main__":
    main()
