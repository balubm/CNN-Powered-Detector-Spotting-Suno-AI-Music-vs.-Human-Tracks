# CNN-Powered Detector: Spotting Suno AI Music vs. Human Tracks

This project implements a **Convolutional Neural Network (CNN)** to distinguish AI-generated music by **Suno** from human-composed tracks. Using spectrogram-based audio representations, the model can reliably classify short audio clips as either AI-generated or human-made.

LinkedIn Article: https://www.linkedin.com/pulse/cnn-powered-detector-spotting-suno-ai-music-vs-human-tracks-balu-brvnc/?trackingId=Z%2FqRYAy7Rx6wETXkHnzZQQ%3D%3D

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Dataset](#dataset)  
- [Data Preparation](#data-preparation)  
- [Model Architecture](#model-architecture)  
- [Training](#training)  
- [Results](#results)  
- [Future Directions](#future-directions)  

---

## Project Overview

As AI music generation improves, it becomes increasingly challenging to differentiate AI-generated tracks from human-composed music. This project uses a CNN trained on spectrograms to classify music clips as either Suno-generated or human-composed.

The model achieves high accuracy and per-class performance, providing a tool for AI-content verification, tagging, and monitoring AI-generated content.

---

## Dataset

The dataset combines:

- **Suno AI-generated tracks** (from Suno 21K)  
- **Human-composed tracks** (from FMA Medium)  

A total of **4,000 clips** were selected (2,000 Suno + 2,000 Human).

---

## Data Preparation

### Balanced Sampling
- Randomly select 2,000 Suno tracks and 2,000 human tracks.  
- Shuffle to prevent ordering bias.

### Audio Processing
- Resample all audio to **44.1 kHz**.  
- Extract **20-second segments**.  
- Skip files shorter than 20 seconds.

### Feature Extraction
- Compute **STFT spectrograms** (`n_fft=4096`, `hop_length=512`).  
- Convert to **power spectrograms in decibels**.  
- Assign binary labels: `1` for Suno, `0` for Human.

### Batch Saving
- Data processed in batches of **1,000 samples** to manage memory.  
- Each batch saved as `.npz` containing `spectrograms` and `labels`.

---

## Model Architecture

The CNN uses **grayscale spectrograms** of shape `(2049, 1723, 1)` as input.  

**Architecture:**
- **Conv Block 1:** 32 filters, 3×3 kernel, stride 2 → MaxPooling 4×4  
- **Conv Block 2:** 64 filters, 3×3 kernel, stride 2 → MaxPooling 2×2  
- **Conv Block 3:** 128 filters, 3×3 kernel, stride 2 → MaxPooling 4×4 (padding=same)  
- **Conv Block 4:** 256 filters, 3×3 kernel, stride 2 → MaxPooling 4×4 (padding=same)  
- **Dense Layers:** 128 → 64 → 1 (sigmoid activation)  

**Training Settings:**
- **Optimizer:** Adam (lr=0.001)  
- **Loss:** Binary Cross-Entropy  
- **Metrics:** Accuracy, Precision, Recall

---

## Training

- **Epochs:** 30  
- **Batch Size:** 32  

**Data Split:**  
- Training: 56%  
- Validation: 24%  
- Testing: 20%  

Training and validation loss curves indicate low bias and controlled variance, with minor overfitting in later epochs.

---

## Results

- High recall for **AI-generated Suno tracks**  
- Slightly lower recall for **human tracks**, indicating minor misclassification  
- **Generalization** confirmed by evaluating additional batches (2–4)

**Per-class metrics example:**

| Class | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| Human | 0.97      | 0.95   | 0.96     |
| Suno  | 0.96      | 0.98   | 0.97     |

---

## Future Directions

- Explore **compact representations** (e.g., mel-spectrograms) to reduce computation.  
- Extend dataset to include music from **other AI models** for generalization.  
- Investigate **model interpretability**: spectral patterns, AI fingerprints, or watermarks.  
- Compare with **explicit watermark detection** systems.  
- Update model for **domain adaptation** as AI-generated music evolves.
