# Use NVIDIA-compatible PyTorch base with CUDA 11.8 support
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Copy all files to the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8000 for the API
EXPOSE 8000

# Run FastAPI with uvicorn
CMD ["uvicorn", "inference_api:app", "--host", "0.0.0.0", "--port", "8000"]
