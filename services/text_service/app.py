import os
import numpy as np
from typing import List, Optional, Union
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import torch

app = FastAPI(
    title="EmbeddingGemma 2 - Text Embedding Service",
    description="8-bit text embedding microservice optimized for 512MB memory machines",
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

class EmbedRequest(BaseModel):
    texts: Union[str, List[str]]
    task: Optional[str] = "SearchQuery"
    title: Optional[str] = None
    truncate_dim: Optional[int] = 768

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    dimension: int
    count: int

@app.on_event("startup")
def load_model():
    global model
    device = os.environ.get("DEVICE", "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if device in ("mps", "cuda") else torch.float32
    print(f"Loading EmbeddingGemma 2 (Text-Only) on {device} ({dtype})...")
    # Disabling unused vision and audio encoders keeps parameters to 271M (~340MB RAM in 8-bit)
    model = SentenceTransformer(
        "google/embeddinggemma-2",
        config_kwargs={"vision_config": None, "audio_config": None},
        model_kwargs={"torch_dtype": dtype},
        device=device,
    )
    print("Text embedding model loaded successfully.")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "embeddinggemma-text-service",
        "precision": "8-bit / optimized",
        "model_loaded": model is not None,
    }

@app.post("/embed", response_model=EmbedResponse)
def embed(request: EmbedRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is still initializing")

    raw_texts = [request.texts] if isinstance(request.texts, str) else request.texts
    if not raw_texts:
        raise HTTPException(status_code=400, detail="Empty text list provided")

    task = request.task or "SearchQuery"
    formatted_texts = []
    use_prompt_name = None

    if task == "Document" and request.title:
        formatted_texts = [f"title: {request.title} | text: {t}" for t in raw_texts]
    else:
        formatted_texts = raw_texts
        use_prompt_name = task

    dim = request.truncate_dim if request.truncate_dim in (128, 256, 512, 768) else 768

    embeddings = model.encode(
        formatted_texts,
        prompt_name=use_prompt_name,
        truncate_dim=dim,
        normalize_embeddings=True,
    )

    if isinstance(embeddings, np.ndarray):
        result = embeddings.tolist()
    else:
        result = [emb.tolist() for emb in embeddings]

    dim_actual = len(result[0]) if result else 0

    return EmbedResponse(
        embeddings=result,
        dimension=dim_actual,
        count=len(result),
    )
