import os
import numpy as np
from typing import List, Optional, Union
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tokenizers import Tokenizer
import onnxruntime as ort

app = FastAPI(
    title="EmbeddingGemma 2 - Text Embedding Service (ONNX Q4)",
    description="Ultra-compact 102MB Q4 ONNX text embedding microservice (<250MB RAM)",
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
tokenizer = None

MODEL_DIR = os.environ.get("MODEL_DIR", "/app/model")
MODEL_PATH = os.path.join(MODEL_DIR, "onnx", "model_q4.onnx")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.json")

class EmbedRequest(BaseModel):
    texts: Union[str, List[str]]
    task: Optional[str] = "SearchQuery"
    title: Optional[str] = None
    truncate_dim: Optional[int] = 768

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    dimension: int
    count: int

def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(norm, 1e-12)

@app.on_event("startup")
def load_model():
    global session, tokenizer
    print(f"Loading ONNX Q4 Session from {MODEL_PATH}...")
    
    # Fallback to downloading if not baked into image (e.g. local dev)
    if not os.path.exists(MODEL_PATH) or not os.path.exists(TOKENIZER_PATH):
        from huggingface_hub import hf_hub_download
        print("Model files not found locally, downloading from Hugging Face Hub...")
        hf_hub_download(repo_id="tooape/embeddinggemma-300m-qat-q8-ONNX", filename="onnx/model_q4.onnx", local_dir=MODEL_DIR)
        hf_hub_download(repo_id="tooape/embeddinggemma-300m-qat-q8-ONNX", filename="tokenizer.json", local_dir=MODEL_DIR)

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    
    opts = ort.SessionOptions()
    opts.enable_cpu_mem_arena = False
    opts.enable_mem_pattern = False
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    
    session = ort.InferenceSession(MODEL_PATH, sess_options=opts, providers=["CPUExecutionProvider"])
    print("ONNX Q4 model initialized successfully! Memory footprint: ~220MB.")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "embeddinggemma-text-service",
        "engine": "onnxruntime-q4",
        "model_loaded": session is not None and tokenizer is not None,
    }

@app.post("/embed", response_model=EmbedResponse)
def embed(request: EmbedRequest):
    if session is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is still initializing")

    raw_texts = [request.texts] if isinstance(request.texts, str) else request.texts
    if not raw_texts:
        raise HTTPException(status_code=400, detail="Empty text list provided")

    task = request.task or "SearchQuery"
    formatted_texts = []
    
    for t in raw_texts:
        if task == "Document":
            prefix = f"title: {request.title or 'none'} | text: "
            formatted_texts.append(f"{prefix}{t}")
        elif task == "SearchQuery":
            formatted_texts.append(f"task: search result | query: {t}")
        else:
            formatted_texts.append(f"task: {task} | query: {t}")

    dim = request.truncate_dim if request.truncate_dim in (128, 256, 512, 768) else 768
    all_embeddings = []

    for text in formatted_texts:
        encoded = tokenizer.encode(text, add_special_tokens=True)
        input_ids = np.array([encoded.ids], dtype=np.int64)
        attention_mask = np.ones_like(input_ids)

        outputs = session.run(["sentence_embedding"], {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        })
        emb = outputs[0][0] # shape (768,)
        
        # MRL dimension truncation
        truncated = emb[:dim]
        # L2 normalization
        normalized = normalize(truncated)
        all_embeddings.append(normalized.tolist())

    return EmbedResponse(
        embeddings=all_embeddings,
        dimension=dim,
        count=len(all_embeddings),
    )
