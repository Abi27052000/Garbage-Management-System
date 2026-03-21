from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io
import base64
import numpy as np
import cv2
import os
import torch
from pathlib import Path

# IMPORTANT: Monkey-patch torch.load to use weights_only=False
# This is needed for PyTorch 2.6+ to load ultralytics models
# Safe for your own trained models
_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    # Force weights_only=False for loading models
    kwargs.setdefault('weights_only', False)
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

# Now import ultralytics after patching
from ultralytics import YOLO

app = FastAPI(title="Object Detection API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
# Get the absolute path to the model
CURRENT_DIR = Path(__file__).parent
MODEL_PATH = CURRENT_DIR.parent / "model" / "model" / "best.pt"

print(f"Looking for model at: {MODEL_PATH}")
print(f"Model exists: {MODEL_PATH.exists()}")

try:
    if not MODEL_PATH.exists():
        print(f"ERROR: Model file not found at {MODEL_PATH}")
        model = None
    else:
        model = YOLO(str(MODEL_PATH))
        print(f"✓ Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    import traceback
    traceback.print_exc()
    model = None

@app.get("/")
def read_root():
    return {"message": "Object Detection API is running"}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint to perform object detection on uploaded image
    """
    print(f"\n=== New prediction request ===")
    print(f"Filename: {file.filename}")
    print(f"Content type: {file.content_type}")
    
    if model is None:
        print("ERROR: Model is not loaded!")
        raise HTTPException(status_code=500, detail="Model not loaded. Check server logs for model loading errors.")
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image file
        print("Reading image file...")
        contents = await file.read()
        print(f"File size: {len(contents)} bytes")
        
        image = Image.open(io.BytesIO(contents))
        print(f"Image opened: {image.size}, mode: {image.mode}")
        
        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")
            print("Converted image to RGB")
        
        # Perform inference
        print("Running model inference...")
        results = model(image)
        print(f"Inference complete. Got {len(results)} result(s)")
        
        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            print(f"Found {len(boxes)} detection(s)")
            for box in boxes:
                detection = {
                    "class": result.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                }
                detections.append(detection)
                print(f"  - {detection['class']}: {detection['confidence']:.2f}")
        
        # Generate annotated image
        print("Generating annotated image...")
        annotated_image = results[0].plot()
        
        # Convert annotated image to base64
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(annotated_image_rgb)
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        print(f"✓ Success! Returning {len(detections)} detection(s)")
        
        return JSONResponse(content={
            "detections": detections,
            "num_detections": len(detections),
            "annotated_image": f"data:image/png;base64,{img_str}"
        })
    
    except Exception as e:
        print(f"✗ ERROR in predict endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/predict-batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """
    Endpoint to perform object detection on multiple images
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    results_list = []
    
    for file in files:
        if not file.content_type.startswith("image/"):
            continue
        
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            results = model(image)
            
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    detection = {
                        "class": result.names[int(box.cls[0])],
                        "confidence": float(box.conf[0]),
                        "bbox": box.xyxy[0].tolist()
                    }
                    detections.append(detection)
            
            annotated_image = results[0].plot()
            annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(annotated_image_rgb)
            buffered = io.BytesIO()
            pil_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            results_list.append({
                "filename": file.filename,
                "detections": detections,
                "num_detections": len(detections),
                "annotated_image": f"data:image/png;base64,{img_str}"
            })
        except Exception as e:
            results_list.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return JSONResponse(content={"results": results_list})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
