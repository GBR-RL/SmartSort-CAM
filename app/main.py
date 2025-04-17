
import requests
import base64
import json
from PIL import Image
from io import BytesIO
import sys
import os

# === CONFIG ===
API_URL = "http://localhost:8000/predict?cam=true"
DEFAULT_IMAGE = "bolt.png"  # You can pass a different image as CLI arg

def send_image(image_path):
    with open(image_path, "rb") as f:
        files = {"file": (os.path.basename(image_path), f, "image/png")}
        response = requests.post(API_URL, files=files)

    if response.status_code != 200:
        print(" Error:", response.text)
        return

    data = response.json()

    # === Print summary ===
    print("Prediction Response:")
    for key, value in data.items():
        if key != "gradcam_overlay":
            print(f"{key}: {value}")

    # === Save JSON result ===
    json_filename = f"result_{os.path.splitext(os.path.basename(image_path))[0]}.json"
    with open(json_filename, "w") as jf:
        json.dump(data, jf, indent=2)
    print(f"Saved JSON to {json_filename}")

    # === Save Grad-CAM ===
    if "gradcam_overlay" in data:
        cam_data = data["gradcam_overlay"].split(",")[1]
        cam_image = Image.open(BytesIO(base64.b64decode(cam_data)))

        output_filename = f"gradcam_{os.path.basename(image_path)}"
        cam_image.save(output_filename)
        print(f" Saved Grad-CAM overlay to {output_filename}")
        cam_image.show(title="Grad-CAM Overlay")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE
    send_image(path)
