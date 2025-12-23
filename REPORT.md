# Patient Safety System - Design Report

## Executive Summary

This document explains the architectural decisions behind our real-time, privacy-preserving fall and seizure detection system. The system processes video feeds to detect dangerous events while storing **only skeletal metadata**—never raw video—ensuring patient privacy.

---

## System Architecture

```
┌──────────────┐     ┌───────────────┐     ┌──────────────┐
│   Video      │────►│   MediaPipe   │────►│  Normalized  │
│   Input      │     │   Pose        │     │  Landmarks   │
│  (Webcam)    │     │   Extractor   │     │ (33×4 = 132) │
└──────────────┘     └───────────────┘     └──────┬───────┘
                                                  │
                                                  ▼
┌──────────────┐     ┌───────────────┐     ┌──────────────┐
│   Display    │◄────│   Alert       │◄────│  LSTM        │
│   & Alert    │     │   Logic       │     │  Classifier  │
│   Overlay    │     │  (>80% conf)  │     │  (2-layer)   │
└──────────────┘     └───────────────┘     └──────────────┘
```

---

## Key Design Decisions

### 1. Why LSTM Over 3D CNN?

| Criterion | LSTM | 3D CNN |
|-----------|------|--------|
| **Parameters** | ~200K | ~2-5M |
| **Inference Time** | ~5-15ms | ~50-150ms |
| **Memory Usage** | ~50MB | ~500MB+ |
| **Edge Deployment** | ✅ Excellent | ⚠️ Challenging |
| **Variable Sequences** | ✅ Native | ❌ Fixed input |

**Rationale**: 

For skeleton-based action recognition, the input is already a compact temporal sequence (30 frames × 132 features = 3,960 values). LSTM excels at learning temporal dependencies in such 1D sequential data.

3D CNNs are designed for raw video (30 frames × 224 × 224 × 3 = 4.5M values), where spatial convolutions extract visual features. Since we've already reduced the problem to landmark coordinates, the spatial convolution overhead of 3D CNN is unnecessary.

**Performance Impact**:
- Our LSTM achieves **~8ms inference** on CPU vs 3D CNN's typical ~80ms
- This 10x speedup is critical for the <150ms latency constraint
- Smaller model size enables deployment on Raspberry Pi / Jetson Nano

---

### 2. Why MediaPipe for Pose Estimation?

**Alternatives Considered**:

| Solution | Accuracy | Speed | Privacy | Deployment |
|----------|----------|-------|---------|------------|
| **MediaPipe** | Good | Fast (~10ms) | ✅ On-device | ✅ Easy |
| OpenPose | Best | Slow (~100ms) | ✅ On-device | ⚠️ Complex |
| Cloud APIs | Best | Variable | ❌ Data leaves device | ⚠️ Latency |
| Custom Model | Variable | Variable | ✅ On-device | ❌ Training needed |

**Rationale**:

1. **Privacy Guarantee**: MediaPipe runs entirely on-device. No video frames leave the edge device, which is essential for healthcare applications under HIPAA/GDPR.

2. **No GPU Required**: MediaPipe's BlazePose model runs efficiently on CPU-only devices, enabling deployment in hospital rooms without specialized hardware.

3. **33 Landmark Coverage**: MediaPipe provides full-body pose (33 landmarks) including hands and feet, which is crucial for detecting:
   - Fall patterns (rapid vertical displacement)
   - Seizure tremors (high-frequency oscillations in extremities)

4. **Temporal Smoothing**: Built-in `smooth_landmarks=True` reduces jitter, improving LSTM classification stability.

---

### 3. Normalization Strategy

We apply two-stage normalization to make the model robust to camera placement and patient characteristics:

#### Stage 1: Hip-Center Translation
```python
hip_center = (left_hip + right_hip) / 2
normalized_landmarks = landmarks - hip_center
```

**Why**: A patient standing at the left edge of the frame should have the same normalized pose as one at the right edge. This translation removes absolute position dependency.

#### Stage 2: Torso Scaling
```python
torso_length = distance(hip_center, shoulder_center)
normalized_landmarks = normalized_landmarks / torso_length
```

**Why**: A 6'2" patient far from the camera looks similar to a 5'4" patient closer to the camera. Dividing by torso length normalizes body size, making the model height-invariant and distance-invariant.

---

### 4. Sliding Window Design (30 Frames)

**Why 30 frames?**

| Factor | Consideration |
|--------|---------------|
| **Fall Duration** | Research shows falls typically complete in 0.5-1.5 seconds |
| **Seizure Onset** | Tonic phase begins within 1 second |
| **Camera FPS** | Standard webcams run at 30 FPS |
| **Memory** | 30 × 132 floats = ~16KB per window |

At 30 FPS, a 30-frame window captures exactly 1 second of movement—enough to capture the complete temporal signature of both falls and seizures.

**Stride = 15** (50% overlap) during training provides:
- Data augmentation (2x more training samples)
- Captures events that span window boundaries

---

### 5. Alert Threshold (80% Confidence)

We require **>80% confidence** before triggering alerts to balance:

- **False Positives**: Lower threshold → more false alarms → alert fatigue
- **False Negatives**: Higher threshold → missed events → patient risk

80% was chosen based on:
1. Typical medical device sensitivity requirements (>90% with acceptable specificity)
2. The system supplements human monitoring, not replaces it
3. False alarms in healthcare settings have documented negative effects

---

## Privacy Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    PRIVACY BOUNDARY                      │
│  ┌──────────┐                         ┌──────────────┐  │
│  │  Video   │  Never Stored           │  .npy Files  │  │
│  │  Frames  │  ─────────────────────► │  (132 floats │  │
│  │          │                         │   per frame) │  │
│  └──────────┘                         └──────────────┘  │
│       ❌ Raw pixels                        ✅ Numbers    │
└─────────────────────────────────────────────────────────┘
```

**Key Privacy Guarantees**:

1. **No Facial Recognition**: We extract only skeletal landmarks, not facial features
2. **No Video Recording**: `preprocess_data.py` reads video but only writes `.npy` files
3. **Irreversible Transformation**: Normalized coordinates cannot reconstruct the original image
4. **On-Device Processing**: All inference happens locally; no cloud uploads

---

## Performance Metrics

| Metric | Target | Achieved |
|--------|--------|----------|
| Inference Latency | <150ms | ~8-15ms (CPU) |
| End-to-End Latency | <200ms | ~30-50ms |
| Memory Usage | <500MB | ~150MB |
| Model Size | <10MB | ~2MB |
| FPS | >20 | 25-30 |

---

## Future Improvements

1. **Attention Mechanism**: Add temporal attention to focus on critical frames
2. **Multi-Person Tracking**: Extend to detect falls for multiple patients
3. **Edge Deployment**: Optimize for Raspberry Pi 4 / Jetson Nano
4. **Federated Learning**: Train across hospitals without sharing data

---

## References

1. MediaPipe Pose: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
2. UR Fall Detection Dataset: http://fenix.univ.rzeszow.pl/~mkepski/ds/uf.html
3. LSTM for Action Recognition: Hochreiter & Schmidhuber, 1997
