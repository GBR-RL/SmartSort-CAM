<link rel="icon" type="image/x-icon" href="favicon.ico" />

# SmartSort-CAM Documentation

Welcome to the official documentation site for **SmartSort-CAM** — an end-to-end intelligent computer vision system for classifying industrial parts with explainability.

---

## Overview

SmartSort-CAM is a modular CV system built for industry-grade inspection workflows. It combines:

- Synthetic data generation using Blender
- ConvNeXt-based classification
- Grad-CAM explainability
- FastAPI inference server
- Docker containerization

---

## Repository Structure

Refer to the [README](../README.md) for full details on each folder's responsibility.

---

## Key Features

- **High-fidelity synthetic rendering** (HD lighting, metallic texture, randomized views)
- **ConvNeXt-Large classifier** trained on synthetic parts
- **REST API interface** for deployment
- **Grad-CAM overlays** for explainability
- **Inference tracking + automation** using Docker and Makefile

---

## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/gbr-rl/SmartSort-CAM
cd SmartSort-CAM
```

2. Build the container:
```bash
make build
```

3. Run with GPU support:
```bash
make run-gpu
```

4. Send image to API:
```bash
python app/main.py path/to/image.png
```

---

## Model Performance

The ConvNeXt model was trained on high-quality synthetic images and achieved:

- **100% accuracy** on validation set
- **99.5% accuracy** on shifted-domain images (blurred/low-light)
- Precision, Recall, F1: **All = 1.0**

---

## Contact

Built by Rohiith Gettala, MSc Robotics, RWTH Aachen University.  
Passionate about computer vision, synthetic data, and real-time inference systems.
