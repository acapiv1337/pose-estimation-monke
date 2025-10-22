from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import numpy as np
# import torch

app = FastAPI()

# Allow Vue frontend to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===========================
# Load your model (example)
# ===========================
# Replace this with your actual model loading logic
# e.g. model = torch.load("model.pt", map_location="cpu")
# model.eval()
model = None  # Placeholder for now

@app.on_event("startup")
def load_model():
    global model
    print("🔹 Loading model...")
    # Example: load YOLOv5 or any other PyTorch model
    # model = torch.hub.load('ultralytics/yolov5', 'custom', path='model.pt', force_reload=False)
    model = "dummy_model"
    print("✅ Model loaded successfully!")

# ===========================
# Predict endpoint
# ===========================
@app.post("/predict")
async def predict(frame: UploadFile = File(...)):
    # Read the image bytes
    image_bytes = await frame.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Convert to numpy array (for processing)
    img_np = np.array(image)

    # Example placeholder prediction logic
    # Replace this with your model inference code
    # Example: results = model(img_np)
    # Here, we’ll just simulate a result
    dummy_result = {
        "status": "success",
        "prediction": "Standing",
        "confidence": 0.92
    }

    return dummy_result


@app.get("/")
def root():
    return {"message": "Backend running successfully 🚀"}
