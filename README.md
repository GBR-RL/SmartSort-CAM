# SmartSort-CAM: Intelligent Industrial Part Classification with Grad-CAM Explainability

[![Docker](https://img.shields.io/badge/docker-ready-blue?logo=docker)](https://hub.docker.com/)
[![FastAPI](https://img.shields.io/badge/fastapi-async--api-success?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Made with Blender](https://img.shields.io/badge/rendered%20with-blender-orange?logo=blender)](https://www.blender.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg?logo=python)](https://www.python.org/downloads/release/python-3100/)

![SmartSort-CAM Banner](https://raw.githubusercontent.com/gbr-rl/SmartSort-CAM/main/docs/banner.png)

SmartSort-CAM is a comprehensive computer vision pipeline designed to classify industrial parts (bolts, nuts, washers, gears, bearings etc.) using a ConvNeXt-based deep learning model. The system includes a Blender-based synthetic dataset generator, a REST API for deployment, Grad-CAM visualization for explainability, Docker integration, and detailed exploratory analysis.

---

## Project Structure Overview

```plaintext
SmartSort-CAM/
├── app/                        # REST API handler
│   └── main.py                # Inference API client (calls FastAPI endpoint)
│
├── cv_pipeline/               # Core CV logic
│   └── inference.py           # Model loading, Grad-CAM, and FastAPI server
│
├── assets/                    # Blender rendering assets
│   ├── hdri/                  # HDR lighting maps
│   ├── textures/              # Steel surface textures
│   └── stl/                   # 3D models of industrial parts
│
├── data/                      # Synthetic dataset samples and label CSV
│   └── part_labels.csv
│   ├── test_images/
│   ├── dataset_samples/
│
├── outputs/                   # Inference results
│   ├── predictions/           # JSON output from API inference
│   └── visualizations/        # Grad-CAM image overlays
│
├── notebooks/                 # Exploratory and training notebooks
│   ├── exploratory.ipynb      # EDA of generated dataset
│   └── model_training.ipynb   # Training & evaluation of ConvNeXt
│
├── scripts/                   # Dataset rendering logic
│   └── Render_dataset.py      # Synthetic render generator using Blender
│
├── docker/                    # Containerization support
│   ├── Dockerfile
│   └── Makefile
│
├── requirements.txt           # Python dependencies
├── .gitignore
└── README.md
```

---

## Synthetic Dataset Generation with Blender

Located in `scripts/Render_dataset.py`, the rendering pipeline utilizes:

- **Blender's Cycles Renderer** for high-fidelity image generation.
- **Randomized lighting (HDRI)** via `setup_hdri_white_background()`.
- **Random camera angles & object jitter** using `setup_camera_aimed_at()` and `apply_positional_jitter()`.
- **Steel texture application** to STL meshes using `import_stl()`.
- **HDRI strength and exposure control** to simulate real-world lighting variance.
- **Positional randomness and material reflectivity** to increase dataset diversity.

The output directory is organized by class and tag (e.g., `data/dataset_samples/bolt/good/bolt_001.png`).

Finally, the script auto-generates a CSV:
```
filepath,label
bolt/good/bolt_001.png,bolt
...
```

This CSV is used for model training.

---

## Exploratory Data Analysis (EDA)

Found in `notebooks/exploratory.ipynb`, this notebook analyzes:
- Class distribution bar plots
- Image shape distributions
- Random render samples
- Augmentation previews
- Trained model sanity checks on generated images

This helped verify:
- Data cleanliness
- Label balance
- Visual consistency across categories

---

## Model Training: ConvNeXt + Mixed Precision(FP16)

Defined in `notebooks/model_training.ipynb`:

- **Backbone**: `convnext_large` from TIMM
- **Image Size**: 512x512
- **Loss**: CrossEntropy
- **Optim**: AdamW
- **Regularization**: Dropout (0.3), weight decay (1e-5)
- **Augmentations**: Flip, rotation, color jitter
- **Split**: 80/20 train-validation split

### Results:
- **Validation Accuracy**: 100% on synthetic validation set
- **F1 / Precision / Recall**: All 1.0 (confirmed clean split)
- **Domain shift test (blur, noise, rotation)**: ~99.5% accuracy with high confidence

---

## Inference Pipeline: FastAPI + Grad-CAM

Found in `cv_pipeline/inference.py`, the pipeline offers:

- **API Endpoint**: `/predict`
- **Input**: Multipart image upload
- **Outputs**:
  - `predicted_class`
  - `confidence`
  - `entropy`
  - `top-3 predictions`
  - `gradcam_overlay` (base64 PNG)

### Grad-CAM Details:
- Hooked into the last `conv_dw` block
- Produces pixel-level heatmaps
- Visualizations show part regions influencing predictions

---

## Outputs: Results and Visuals

All inference outputs are saved to `outputs/`:
- Grad-CAM overlays: `outputs/visualizations/gradcam_bolt.png`
- API JSONs: `outputs/predictions/result_bolt.json`

These can be used for audits, reports, or additional UI visualization.

---

## Docker + Automation

In `docker/`, you’ll find:

### Dockerfile (CUDA compatible)
```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "cv_pipeline.inference:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Makefile
```makefile
make build        # Builds Docker image
make run-gpu      # Runs inference server with GPU
make stop         # Stops all containers
```

## 🚀 Getting Started

```bash
# Clone repo
https://github.com/gbr-rl/SmartSort-CAM

# Build Docker image
make build

# Run container (GPU)
make run-gpu

# Send test request
python app/main.py bolt.png
```

---

## 📬 Contact
I’m excited to connect and collaborate!  
- **Email**: [gbrohiith@gmail.com](mailto:your.email@example.com)  
- **LinkedIn**: [https://www.linkedin.com/in/rohiithgb/](https://linkedin.com/in/yourprofile)  
- **GitHub**: [https://github.com/GBR-RL/](https://github.com/yourusername)

---

## 📚 License
This project is open-source and available under the [MIT License](LICENSE).  

---

🌟 **If you like this project, please give it a star!** 🌟

