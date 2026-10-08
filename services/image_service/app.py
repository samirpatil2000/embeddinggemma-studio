import io
import os
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import onnxruntime as ort
import numpy as np

app = FastAPI(
    title="EmbeddingGemma 2 - Image Embedding Service (ONNX Q4)",
    description="Dedicated ultra-compact 63MB Q4 ONNX image embedding microservice (<180MB RAM)",
    version="2.0.0",
)

# Enable CORS for web UI access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

session = None

MODEL_DIR = os.environ.get("MODEL_DIR", "/app/model")
MODEL_PATH = os.path.join(MODEL_DIR, "onnx", "vision_model_q4.onnx")

class ImageEmbedResponse(BaseModel):
    embedding: List[float]
    dimension: int
    format: str

def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(norm, 1e-12)

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
    arr = np.array(image, dtype=np.float32) / 255.0
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1)) # (3, 256, 256)
    return np.expand_dims(arr, 0) # (1, 3, 256, 256)

@app.on_event("startup")
def load_model():
    global session
    print(f"Loading ONNX Q4 Vision Session from {MODEL_PATH}...")
    
    if not os.path.exists(MODEL_PATH):
        from huggingface_hub import hf_hub_download
        print("Model file not found locally, downloading from Hugging Face Hub...")
        hf_hub_download(repo_id="Xenova/siglip-base-patch16-256", filename="onnx/vision_model_q4.onnx", local_dir=MODEL_DIR)

    opts = ort.SessionOptions()
    opts.enable_cpu_mem_arena = False
    opts.enable_mem_pattern = False
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    
    session = ort.InferenceSession(MODEL_PATH, sess_options=opts, providers=["CPUExecutionProvider"])
    print("ONNX Q4 Vision model initialized successfully! Memory footprint: ~180MB.")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "embeddinggemma-image-service",
        "engine": "onnxruntime-q4",
        "model_loaded": session is not None,
    }

@app.post("/embed", response_model=ImageEmbedResponse)
async def embed_image(
    file: UploadFile = File(...),
    truncate_dim: Optional[int] = Form(768),
):
    if session is None:
        raise HTTPException(status_code=503, detail="Model is still initializing")

    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")

    pixel_values = preprocess_image(image)
    
    outputs = session.run(["pooler_output"], {"pixel_values": pixel_values})
    emb = outputs[0][0] # shape (768,)
    
    dim = truncate_dim if truncate_dim in (128, 256, 512, 768) else 768
    truncated = emb[:dim]
    normalized = normalize(truncated)

    return ImageEmbedResponse(
        embedding=normalized.tolist(),
        dimension=dim,
        format=file.content_type or "image/png",
    )
