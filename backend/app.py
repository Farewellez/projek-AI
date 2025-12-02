import os
import pickle
import base64
import logging
import time
import numpy as np
import cv2
import mediapipe as mp
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, Body, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from datetime import datetime

# --- Konfigurasi Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- Konfigurasi Path Model ---
MODEL_PATH = os.path.join("models", "rf_model.pkl")

# --- Global Variables ---
ml_resources = {}

def load_model():
    if not os.path.exists(MODEL_PATH):
        logger.warning(f"Model not found at {MODEL_PATH}")
        return None
    try:
        with open(MODEL_PATH, 'rb') as f:
            data = pickle.load(f)
        resources = {}
        if isinstance(data, dict):
            resources['model'] = data.get('model')
            resources['encoder'] = data.get('encoder')
            resources['classes'] = data.get('classes')
        else:
            resources['model'] = data
            resources['encoder'] = None
            resources['classes'] = getattr(data, 'classes_', None)
        logger.info("Model loaded successfully.")
        return resources
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_resources.update(load_model() or {})
    ml_resources['hands'] = mp.solutions.hands.Hands(
        static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5
    )
    yield
    if 'hands' in ml_resources:
        ml_resources['hands'].close()
    ml_resources.clear()

app = FastAPI(title="IsyaratKu API", lifespan=lifespan)

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Exception Handlers (Debug 400 Error) ---
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation Error: {exc}")
    body = await request.body()
    logger.error(f"Body content: {body.decode('utf-8')[:100]}...") # Print depan body biar tau isinya
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc.errors()), "body": str(body)},
    )

# --- Preprocessing ---
def extract_features(image_rgb):
    hands = ml_resources.get('hands')
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0]
        points = np.array([[p.x, p.y, p.z] for p in lm.landmark])
        # Normalize
        base = points[0]
        points = points - base
        max_val = np.max(np.abs(points))
        if max_val > 0:
            points /= max_val
        return points.flatten().reshape(1, -1)
    return None

class ImagePayload(BaseModel):
    image_base64: str

def decode_image(base64_str):
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    return cv2.cvtColor(cv2.imdecode(np.frombuffer(base64.b64decode(base64_str), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

@app.post("/predict")
async def predict(data: ImagePayload):
    # Kita pisahkan endpoint JSON murni agar tidak bentrok dengan UploadFile
    start = time.time()
    try:
        img = decode_image(data.image_base64)
        model = ml_resources.get('model')
        if not model:
            raise HTTPException(503, "Model not loaded")

        features = extract_features(img)
        if features is None:
            # Return 200 dengan status khusus biar frontend gak error console
            return {"text": "-", "confidence": 0.0, "status": "no_hand"}

        # Inference
        pred_idx = model.predict(features)[0]
        
        # Decoding Label
        label = str(pred_idx)
        encoder = ml_resources.get('encoder')
        classes = ml_resources.get('classes')
        
        if encoder:
            label = encoder.inverse_transform([pred_idx])[0]
        elif classes is not None:
            # Fallback manual jika classes tersedia
            classes_list = classes.tolist() if hasattr(classes, 'tolist') else classes
            if isinstance(pred_idx, (int, np.integer)) and 0 <= pred_idx < len(classes_list):
                label = classes_list[pred_idx]

        conf = 0.5
        if hasattr(model, "predict_proba"):
            conf = float(np.max(model.predict_proba(features)[0]))

        logger.info(f"Pred: {label} ({conf:.2f}) - {time.time()-start:.3f}s")
        return {
            "text": str(label), 
            "confidence": conf,
            "status": "ok"
        }

    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(500, str(e))