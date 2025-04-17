import io
import os
import time
import torch
import timm
import base64
import uvicorn
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from torchvision.transforms.functional import to_pil_image
from torch.nn.functional import softmax

# === CONFIG ===
MODEL_PATH = "E:\VisionInspect\models\convnext_large.pth"
IMAGE_SIZE = 512
NUM_CLASSES = 6
#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cuda")
CLASS_NAMES = ['bolt', 'gear', 'nut', 'washer', 'bearing', 'connector']
MODEL_VERSION = os.path.splitext(os.path.basename(MODEL_PATH))[0]  # Dynamic version

# === LOAD MODEL ===
model = timm.create_model('convnext_large', pretrained=False, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval().to(DEVICE)

# === GRAD-CAM HOOKS ===
gradients = None
activations = None

def save_gradient(module, grad_in, grad_out):
    global gradients
    gradients = grad_out[0]

def save_activation(module, input, output):
    global activations
    activations = output

target_layer = model.stages[-1].blocks[-1].conv_dw  # ✅ correct hook layer
target_layer.register_forward_hook(save_activation)
target_layer.register_backward_hook(save_gradient)

# === PREPROCESSING ===
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# === FASTAPI INIT ===
app = FastAPI(title="ConvNeXt Inference API with Explainability")

# === ROUTES ===
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    cam: bool = Query(False, description="Include Grad-CAM overlay (true/false)")
):
    if not file.filename.lower().endswith(('png', 'jpg', 'jpeg')):
        raise HTTPException(status_code=400, detail="Invalid image format.")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Unable to read image.") from e

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    #with torch.no_grad():
    start = time.time()
    output = model(input_tensor)
    probs = softmax(output, dim=1)[0]
    end = time.time()

    confidence, pred_idx = torch.max(probs, dim=0)
    topk = torch.topk(probs, k=3)
    top3 = [{"class": CLASS_NAMES[i], "score": round(p.item(), 4)} for i, p in zip(topk.indices, topk.values)]
    entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
    is_confident = confidence.item() >= 0.8

    response = {
        "predicted_class": CLASS_NAMES[pred_idx.item()],
        "confidence": round(confidence.item(), 4),
        "is_confident": is_confident,
        "top3": top3,
        "entropy": round(entropy, 4),
        "model_version": MODEL_VERSION,
        "preprocessing": {
            "resized_to": [IMAGE_SIZE, IMAGE_SIZE],
            "normalized_with": "[0.5, 0.5, 0.5]"
        },
        "inference_time_ms": round((end - start) * 1000, 2)
    }

    if cam:
        # === Generate Grad-CAM ===
        model.zero_grad()
        output[0, pred_idx].backward(retain_graph=True)

        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = heatmap.detach().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        heatmap = np.uint8(255 * heatmap)
        heatmap = Image.fromarray(heatmap).resize(image.size, resample=Image.BILINEAR).convert("RGB")

        # Overlay heatmap
        blended = Image.blend(image, heatmap, alpha=0.5)

        # Convert to base64
        buffered = io.BytesIO()
        blended.save(buffered, format="PNG")
        cam_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        response["gradcam_overlay"] = f"data:image/png;base64,{cam_base64}"

    return JSONResponse(response)

@app.get("/")
def root():
    return {"message": "ConvNeXt REST API with Grad-CAM is live."}

if __name__ == "__main__":
    uvicorn.run("inference_api:app", host="0.0.0.0", port=8000)
