import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
# Heavy ML imports are deferred when SKIP_MODEL is enabled
cv2 = None
torch = None
YOLO = None
import logging
import sys
import uuid
import json
import os

# --- CONFIGURATION ---
# UPDATE THIS to your local .pt file path
MODEL_PATH = r"D:\AMD\hvac-ai\ai_model\models\yolo11m_run_v10\weights\best.pt"
PORT = 8000
CONF_THRES = 0.50
IOU_THRES = 0.45
IMG_SIZE = 1024
# HALF will be set after importing torch (if model is loaded); default false

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [BACKEND] - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("HVAC-Backend")

app = FastAPI(title="HVAC Local Inference API")

# Allow CORS for localhost frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
HALF = False

# Dev mode: skip loading heavy ML stack
# Can be enabled via environment variable SKIP_MODEL=1 or CLI arg --no-model
SKIP_MODEL = os.environ.get("SKIP_MODEL", "0") == "1" or any(a in ("--no-model", "--skip-model") for a in sys.argv)


@app.on_event("startup")
async def load_model():
    global model
    if SKIP_MODEL:
        logger.warning("SKIP_MODEL is enabled — skipping heavy ML imports and model load (dev mode).")
        return

    if not os.path.exists(MODEL_PATH):
        logger.error(f"❌ Model not found at: {MODEL_PATH}")
        # We don't exit here to keep the server alive for health checks,
        # but inference will fail.
        return

    logger.info(f"Loading model from {MODEL_PATH}...")
    try:
        # Import heavy dependencies here so they can be skipped in dev mode
        global torch, cv2, YOLO, HALF
        import cv2 as _cv2
        import torch as _torch
        from ultralytics import YOLO as _YOLO
        cv2 = _cv2
        torch = _torch
        YOLO = _YOLO

        HALF = torch.cuda.is_available()

        model = YOLO(MODEL_PATH)
        if torch.cuda.is_available():
            model.to('cuda')
            logger.info(f"✅ Model loaded on GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("⚠️ GPU not found. Running on CPU (Slower).")
        
        # Warmup
        model.predict(np.zeros((640,640,3), dtype=np.uint8), verbose=False, half=HALF)
        logger.info("🔥 Model Warmup Complete")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")


@app.get("/health")
def health_check():
    status = "healthy" if model else "model_not_loaded"
    return {"status": status, "device": str(model.device) if model else "none"}


async def process_image(file_bytes):
    # If model isn't loaded (either because SKIP_MODEL or failure), raise
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image")
    
    height, width = img.shape[:2]

    results = model.predict(
        img, 
        conf=CONF_THRES, 
        iou=IOU_THRES, 
        imgsz=IMG_SIZE,
        half=HALF
    )
    
    result = results[0]
    detections = []

    # Handle OBB
    if hasattr(result, 'obb') and result.obb is not None:
        for box in result.obb:
            poly = box.xyxyxyxy[0].cpu().numpy().tolist()
            xywhr = box.xywhr[0].cpu().numpy().tolist()
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            label = result.names[cls_id]

            # Calculate standard bbox for fallback
            x_coords = [p[0] for p in poly]
            y_coords = [p[1] for p in poly]
            bbox_rect = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

            detections.append({
                "id": str(uuid.uuid4()),
                "label": label,
                "score": conf,
                "bbox": bbox_rect,
                "points": poly,
                "obb": {
                    "x_center": xywhr[0],
                    "y_center": xywhr[1],
                    "width": xywhr[2],
                    "height": xywhr[3],
                    "rotation": xywhr[4]
                }
            })
            
    return {
        "success": True,
        "image": {"width": width, "height": height},
        "count": len(detections),
        "detections": detections
    }

# Match the path your frontend expects
@app.post("/api/hvac/analyze") 
async def analyze_image(request: Request, file: UploadFile = File(None), image: UploadFile = File(None)):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    target_file = file or image
    if not target_file:
        raise HTTPException(status_code=422, detail="No file uploaded")
    
    try:
        contents = await target_file.read()
        response_data = await process_image(contents)
        logger.info(f"Analyzed {target_file.filename}: Found {response_data['count']} items")

        # SSE Stream support
        if request.query_params.get("stream") == "1":
            async def event_generator():
                yield f"data: {json.dumps(response_data)}\n\n"
                yield "event: close\ndata: [DONE]\n\n"
            return StreamingResponse(event_generator(), media_type="text/event-stream")
        
        return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Ensure stdout is flushed immediately for the parent process to read
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_config=None)
