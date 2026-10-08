import io
import os
import base64
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

app = FastAPI(
    title="EmbeddingGemma 2 - Image & Video Embedding Service",
    description="Dedicated image/video embedding microservice (4-bit / 512MB RAM)",
    version="1.0.0",
)

# Enable CORS for web UI access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None

class ImageEmbedResponse(BaseModel):
    embedding: List[float]
    dimension: int
    format: str

@app.on_event("startup")
def load_model():
    global model
    device = os.environ.get("DEVICE", "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if device in ("mps", "cuda") else torch.float32
    print(f"Loading EmbeddingGemma 2 (Vision + Text) on {device} ({dtype})...")
    # Dropping audio config saves ~305M params
    model = SentenceTransformer(
        "google/embeddinggemma-2",
        config_kwargs={"audio_config": None},
        model_kwargs={"torch_dtype": dtype},
        device=device,
    )
    print("Image embedding model loaded successfully.")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "embeddinggemma-image-service",
        "model_loaded": model is not None,
    }

@app.post("/embed", response_model=ImageEmbedResponse)
async def embed_image(
    file: UploadFile = File(...),
    truncate_dim: Optional[int] = Form(768),
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is still initializing")

    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")

    dim = truncate_dim if truncate_dim in (128, 256, 512, 768) else 768

    embedding = model.encode(
        image,
        truncate_dim=dim,
        normalize_embeddings=True,
    )

    if isinstance(embedding, np.ndarray):
        vec = embedding.tolist()
    else:
        vec = list(embedding)

    return ImageEmbedResponse(
        embedding=vec,
        dimension=len(vec),
        format=file.content_type or "image/png",
    )
