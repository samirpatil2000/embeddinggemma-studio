import os
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

app = FastAPI(
    title="EmbeddingGemma 2 - Audio Embedding Service",
    description="Dedicated audio embedding microservice (16kHz mono) for Render machines",
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

class AudioEmbedResponse(BaseModel):
    embedding: List[float]
    dimension: int
    filename: str

@app.on_event("startup")
def load_model():
    global model
    device = os.environ.get("DEVICE", "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if device in ("mps", "cuda") else torch.float32
    print(f"Loading EmbeddingGemma 2 (Audio + Text) on {device} ({dtype})...")
    # Dropping vision config keeps parameters down
    model = SentenceTransformer(
        "google/embeddinggemma-2",
        config_kwargs={"vision_config": None},
        model_kwargs={"torch_dtype": dtype},
        device=device,
    )
    print("Audio embedding model loaded successfully.")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "embeddinggemma-audio-service",
        "model_loaded": model is not None,
    }

@app.post("/embed", response_model=AudioEmbedResponse)
async def embed_audio(
    file: UploadFile = File(...),
    truncate_dim: Optional[int] = Form(768),
):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is still initializing")

    content = await file.read()
    dim = truncate_dim if truncate_dim in (128, 256, 512, 768) else 768

    try:
        temp_audio_path = f"/tmp/{file.filename or 'temp_audio.wav'}"
        with open(temp_audio_path, "wb") as f:
            f.write(content)

        embedding = model.encode(
            temp_audio_path,
            truncate_dim=dim,
            normalize_embeddings=True,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Audio processing error: {e}")

    if isinstance(embedding, np.ndarray):
        vec = embedding.tolist()
    else:
        vec = list(embedding)

    return AudioEmbedResponse(
        embedding=vec,
        dimension=len(vec),
        filename=file.filename or "audio.wav",
    )
