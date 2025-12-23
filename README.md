# Real-time Fall Detection System

A privacy-preserving, real-time fall detection system using LSTM neural networks and MediaPipe pose estimation. Designed for patient safety monitoring in healthcare environments.

## 🎯 Features

- **Real-time Detection**: <150ms latency for immediate alerts
- **Privacy-Preserving**: Stores only skeletal metadata, never raw video
- **Lightweight**: ~200K parameters, runs on edge devices (Raspberry Pi, Jetson Nano)
- **High Accuracy**: LSTM-based temporal analysis of human poses

## 🏗️ Architecture

```
Video Input → MediaPipe Pose → Normalized Landmarks → LSTM Classifier → Alert System
   (Webcam)     Extractor        (33×4 = 132)         (2-layer)
```

## 📁 Project Structure

```
├── config.py           # Configuration parameters
├── dataset.py          # Dataset loading and preprocessing
├── model.py            # LSTM model architecture
├── train.py            # Training script
├── inference.py        # Real-time inference
├── preprocess_data.py  # Data preprocessing utilities
├── utils.py            # Helper functions
├── requirements.txt    # Dependencies
├── REPORT.md           # Detailed design report
├── DEFENSE.md          # Technical defense document
├── data/               # Dataset directory
│   ├── adls/           # Activities of daily living
│   ├── falls/          # Fall event data
│   └── processed/      # Preprocessed data
└── models/             # Saved model checkpoints
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/saurabh6354/Real-time-Fall-Detection.git
cd Real-time-Fall-Detection

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
python train.py
```

### Real-time Inference

```bash
python inference.py
```

## 🛠️ Tech Stack

- **Deep Learning**: PyTorch
- **Pose Estimation**: MediaPipe
- **Computer Vision**: OpenCV
- **Data Processing**: NumPy, Pandas

## 📊 Model Performance

| Metric         | Value      |
| -------------- | ---------- |
| Parameters     | ~200K      |
| Inference Time | ~8ms (CPU) |
| Memory Usage   | ~50MB      |

## 📄 Documentation

- [Design Report](REPORT.md) - Detailed architectural decisions
- [Technical Defense](DEFENSE.md) - Technical justifications

## 📝 License

This project is for educational and research purposes.

## 👤 Author

**Saurabh** - [GitHub](https://github.com/saurabh6354)
