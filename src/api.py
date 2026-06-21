import sys
import os
from io import BytesIO

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from PIL import Image

# ── Import your existing predictor ──────────────────────────
sys.path.append(os.path.dirname(__file__))
try:
    from predictor import predict
except ImportError:
    # Fallback mock if predictor not available yet
    def predict(img, lang="English"):
        return {
            "is_healthy": False,
            "disease": "Apple Scab",
            "crop": "Apple",
            "confidence": 98.5,
            "severity": "High",
            "advice": "Apply fungicide immediately. Remove infected leaves."
        }

# ── App Setup ────────────────────────────────────────────────
app = FastAPI(title="CropAI API", version="1.0.0")

# Allow requests from index.html (opened as a local file or any origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve index.html at root — this lets you open http://127.0.0.1:8000
# and have it work, instead of opening the file directly from disk.
# The index.html must be in the project root folder (one level up from src/).
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))  # D:\crop_disease_detector\


# ── Routes ───────────────────────────────────────────────────

@app.get("/")
def serve_frontend():
    """Serve the index.html from the project root."""
    html_path = os.path.join(ROOT_DIR, "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path, media_type="text/html")
    return {"message": "CropAI API is running. Place index.html in project root."}


@app.get("/health")
def health_check():
    """Health check — the HTML calls this to show 'API Connected' status."""
    return {"status": "ok", "model": "ResNet50 + Groq Llama4", "accuracy": "98.78%"}


@app.post("/screen-file")
async def detect_from_file(
    file: UploadFile = File(...),
    language: str = "English"
):
    """
    Primary endpoint — HTML sends image as multipart FormData.
    Returns JSON that index.html's showResult() function expects:
      { category, confidence, severity, advice, crop, is_healthy }
    """
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/webp", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Only JPG, PNG, WEBP images are supported.")

    # Read and open image
    contents = await file.read()
    try:
        img = Image.open(BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not open image. Please upload a valid image file.")

    # Run your existing predict() function
    result = predict(img, language)

    # Map your predict() output → JSON fields the HTML expects
    # HTML uses: data.category || data.disease  for the disease name
    # HTML checks if category.toLowerCase().includes('healthy')
    if result.get("is_healthy"):
        category = f"{result.get('crop', 'Crop')} - Healthy"
    else:
        category = result.get("disease", "Unknown Disease")

    return {
        "status":     "success",
        "category":   category,          # what index.html reads for the banner
        "disease":    result.get("disease", ""),
        "crop":       result.get("crop", ""),
        "confidence": result.get("confidence", 0),
        "severity":   result.get("severity", "Medium"),
        "advice":     result.get("advice", ""),
        "is_healthy": result.get("is_healthy", False),
    }


@app.post("/screen")
async def detect_from_json(payload: dict):
    """
    Fallback endpoint — HTML sends base64 JSON if /screen-file fails.
    Returns same format as /screen-file.
    """
    # For the fallback we return a demo result since base64 crop image
    # detection needs the same PIL flow; extend this if needed.
    return {
        "status":     "success",
        "category":   "Unknown - Please use image upload",
        "confidence": 0,
        "severity":   "Unknown",
        "advice":     "Please upload the image using the file uploader for accurate detection.",
        "is_healthy": False,
    }
