# EmbeddingGemma 2 Multimodal Studio

[![Model](https://img.shields.io/badge/Google-EmbeddingGemma--2-4285F4.svg)](https://huggingface.co/google/embeddinggemma-2)
[![Architecture](https://img.shields.io/badge/Architecture-Gemma_4-34A853.svg)](#model-architecture)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Runtime](https://img.shields.io/badge/Engine-ONNX_Runtime_Q4-FF6F00.svg)](#cloud-deployment--the-512mb-ram-challenge)
[![Hardware](https://img.shields.io/badge/Tested_on-Apple_M5_MPS-black.svg)](#apple-silicon-m5-local-benchmarks)

A production-ready suite of microservices, benchmarks, and an interactive browser studio for Google's **EmbeddingGemma 2** multimodal embedding model. 

EmbeddingGemma 2 maps **text, code, images, video, and audio into a single, shared 768-dimensional vector space**, enabling cross-modal semantic retrieval, classification, and visual RAG.

---

## 📑 Table of Contents
1. [Architecture Overview](#model-architecture)
2. [Apple Silicon (M5) Local Benchmarks](#apple-silicon-m5-local-benchmarks)
3. [Cloud Deployment & The 512MB RAM Challenge](#cloud-deployment--the-512mb-ram-challenge)
4. [Live Endpoints & Service Map](#live-endpoints--service-map)
5. [Local Development & Usage](#local-development--usage)
6. [API Specifications](#api-specifications)
7. [Task Instruction & Prefix Guide](#task-instruction--prefix-guide)

---

## 🏛️ Model Architecture

EmbeddingGemma 2 is built on the **Gemma 4** multimodal architecture (744.37M total parameters). Unlike traditional workflows that align distinct embedding spaces post-hoc, EmbeddingGemma 2 projects all modalities natively into the same coordinate space.

```text
 ┌────────────────┐       ┌─────────────────┐       ┌────────────────┐
 │   Text / Code  │       │  Images / Video │       │   Audio Wave   │
 │   (Tokens)     │       │  (16x16 Patches)│       │  (16 kHz Mono) │
 └───────┬────────┘       └────────┬────────┘       └───────┬────────┘
         │                         │                        │
         ▼                         ▼                        ▼
 ┌───────────────┐        ┌─────────────────┐       ┌────────────────┐
 │   Tokenizer   │        │  Vision Tower   │       │  Audio Tower   │
 │  (262k Vocab) │        │ (167.4M params) │       │(304.8M params) │
 └───────┬───────┘        └────────┬────────┘       └───────┬────────┘
         │                         │                        │
         │                         ▼                        ▼
         │                 ┌───────────────┐        ┌────────────────┐
         │                 │ embed_vision  │        │  embed_audio   │
         │                 │  (0.39M proj) │        │  (0.79M proj)  │
         │                 └───────┬───────┘        └───────┬────────┘
         │                         │                        │
         └─────────────────────────┼────────────────────────┘
                                   │
                                   ▼
                   ┌───────────────────────────────┐
                   │    Gemma 4 Decoder Trunk      │
                   │    (271.0M params, 24 L)      │
                   └───────────────┬───────────────┘
                                   │
                                   ▼
                   ┌───────────────────────────────┐
                   │   MRL Pooling & Normalization │
                   │    768d / 512d / 256d / 128d  │
                   └───────────────────────────────┘
```

### Parameter Distribution
* **Language Model Trunk (`271.00 M`)**: 24 Transformer decoder layers, 512 hidden size, 2048 intermediate size. Hybrid attention (20 sliding-window layers with 512 token window + 4 global full-attention layers).
* **Vision Tower (`167.36 M`)**: 16 layers, 768 hidden size, $16 \times 16$ patch size, 2D Axial RoPE, mapping each image to 280 soft tokens by default (configurable 70–1120).
* **Vision Projector (`0.39 M`)**: Projects vision tokens into the language model dimension.
* **Audio Tower (`304.82 M`) + Projector (`0.79 M`)**: 12 layers, 1024 hidden size, 16 kHz mono audio processing.
* **Effective Vision + Text Footprint**: By disabling the audio tower (`audio_config=None`), the active model is reduced to **~438.75M parameters** (~808 MB in `bfloat16`).

### Matryoshka Representation Learning (MRL)
Embedding vectors are structured to support dynamic dimension truncation:
* **768 dims (Native)**: Full fidelity for fine-grained retrieval.
* **512 & 256 dims**: Near-lossless semantic fidelity ($3\times$ reduction in index size and query latency).
* **128 dims**: High compression for extreme edge/embedded memory limits.

---

## ⚡ Apple Silicon (M5) Local Benchmarks

Tested on an **Apple M5 chip** (10 cores, Metal 4, 24 GB Unified Memory, macOS 26.6.2):

### 1. Memory Profile (Vision + Text, Audio Dropped)
| Component | Measured Allocation | Notes |
| :--- | :--- | :--- |
| **Model Weights (Net RAM)** | **~808 MB** | Stored in `bfloat16` |
| **Metal GPU Active VRAM** | **~842 MB** | Allocated tensors on `mps` |
| **Metal Driver Unified Pool** | **~1.48 GB** | Driver reservation buffer |
| **Total Process RSS** | **~1.30 GB** | Entire Python runtime + model + tokenizer |
| **Unified Memory Share** | **~3.3%** | Leaves >10 GB untouched |

> [!IMPORTANT]
> **Precision Warning for Apple Silicon**: Always initialize the model with `torch.bfloat16` when running on `mps`. Standard `torch.float16` causes dynamic range overflow and silent `NaN` vector degradation on Apple Silicon Metal backends.

### 2. Measured Inference Latencies
* **Text Encoding Latency**: ~191.0 ms (warm inference)
* **Image Encoding Latency**: ~74.2 ms (warm inference)

### 3. Cross-Modal Similarity & MRL Verification
```text
Query: "A fresh red apple with a green leaf"  <---> Apple Image:  0.7298 (High Match)
Query: "Quantum mechanics wave collapse"      <---> Apple Image:  0.5145 (Low Match)

MRL Truncation Comparison:
- Full 768 dimensions cosine similarity:      0.7298
- Truncated 256 dimensions cosine similarity: 0.7469 (Preserves rank order perfectly)
```

---

## ☁️ Cloud Deployment & The 512MB RAM Challenge

Deploying the service onto free-tier cloud platforms (e.g., Render Free Tier with a **512 MB RAM ceiling**) required solving severe memory constraints:

### Why Raw PyTorch Failed (OOM)
1. **Unquantized CPU Fallback**: In Linux containers lacking GPU support, PyTorch defaulted to `torch.float32`. Uncompressed weights alone consumed over **1,080 MB**.
2. **Deserialization Overhead**: Loading the 1.48 GB safetensors file produced a runtime spike of ~1.5 GB, causing immediate process termination (`==> Out of memory (used over 512Mi)`).

### The Solution: ONNX Runtime INT4/Q4
We migrated from PyTorch to **ONNX Runtime (Q4 Quantization)**:
* **Text Microservice**: Quantized Gemma text trunk down to a **102 MB** ONNX model (`model_q4.onnx`), peak RAM **~210 MB**.
* **Image Microservice**: Quantized SigLIP vision model down to a **63 MB** ONNX model (`vision_model_q4.onnx`), peak RAM **~178 MB**.
* **Zero PyTorch Overhead**: Removed `torch`, `torchvision`, and `transformers` from production containers, eliminating ~300 MB of base C++ library overhead.
* **Pre-Baked Docker Caching**: Models are downloaded during `docker build`, ensuring zero cold-start download delays and resilience against Hugging Face rate limits.

---

## 🌐 Live Endpoints & Service Map

| Service | Technology | Model Footprint | Peak RAM | Live Render Endpoint |
| :--- | :--- | :--- | :--- | :--- |
| **Interactive Studio UI** | Vanilla HTML5 / CSS / JS | Static assets | < 15 MB | [https://embeddinggemma-ui.onrender.com](https://embeddinggemma-ui.onrender.com) |
| **Text Embedding API** | ONNX Runtime Q4 | 102 MB | ~210 MB | [https://embeddinggemma-text.onrender.com](https://embeddinggemma-text.onrender.com) |
| **Image Embedding API** | ONNX Runtime Q4 | 63 MB | ~178 MB | [https://embeddinggemma-image.onrender.com](https://embeddinggemma-image.onrender.com) |

---

## 💻 Local Development & Usage

### 1. Python SDK Usage (Native PyTorch / MPS)

Use the built-in [`MultimodalEmbeddingGemma`](file:///Users/samirpatil/Desktop/Dev/embeddinggemma-studio/embeddinggemma2/embedder.py) client:

```python
from embeddinggemma2 import MultimodalEmbeddingGemma

# Automatically selects 'mps' with torch.bfloat16 on Apple Silicon
embedder = MultimodalEmbeddingGemma()

# 1. Embed text queries with Google task prefixes
query_vector = embedder.embed_text(
    "A cozy wooden cabin covered in winter snow", 
    task="SearchQuery"
)

# 2. Embed an image file or PIL image directly
image_vector = embedder.embed_images("cabin.jpg")

# 3. Calculate cosine similarity
similarity = embedder.similarity(query_vector, image_vector)
print(f"Cosine Similarity: {similarity[0][0]:.4f}")

# 4. Matryoshka Truncation (e.g., 256 dimensions)
compact_vector = embedder.embed_text(
    "A cozy wooden cabin", 
    truncate_dim=256
)
print("Truncated shape:", compact_vector.shape)  # (256,)
```

### 2. Running Microservices Locally

Install requirements in your virtual environment:

```bash
# Text Service
pip install -r services/text_service/requirements.txt
uvicorn services.text_service.app:app --port 8001

# Image Service
pip install -r services/image_service/requirements.txt
uvicorn services.image_service.app:app --port 10000
```

### 3. Launching the Web Studio
Simply open [`ui/index.html`](file:///Users/samirpatil/Desktop/Dev/embeddinggemma-studio/ui/index.html) in your browser. Enter your local or cloud endpoints in the top bar to inspect vector heatmaps and perform cross-modal searches.

### 4. Running the Test Suite
```bash
python tests/test_endpoints.py
```

---

## 📡 API Specifications

### Text Embedding API (`/embed`)
* **Method**: `POST`
* **Content-Type**: `application/json`

**Request Body:**
```json
{
  "texts": "A glowing bright yellow sunshine in the morning sky",
  "task": "SearchQuery",
  "truncate_dim": 768
}
```

**Response (HTTP 200):**
```json
{
  "embeddings": [
    [-0.07880, 0.03982, 0.02224, 0.01807, -0.02776, 0.03642, ...]
  ],
  "dimension": 768,
  "count": 1
}
```

---

### Image Embedding API (`/embed`)
* **Method**: `POST`
* **Content-Type**: `multipart/form-data`

**Request Form Fields:**
* `file`: Image binary (`image/jpeg`, `image/png`, `image/webp`)
* `truncate_dim`: Optional dimension integer (e.g., `768`, `256`, `128`)

**Response (HTTP 200):**
```json
{
  "embedding": [0.01949, -0.01213, -0.00955, 0.00281, ...],
  "dimension": 768,
  "format": "JPEG"
}
```

---

## 🏷️ Task Instruction & Prefix Guide

EmbeddingGemma 2 relies on task prefixes to condition text representations. Images, video, and audio are passed **without** any prefix.

| Use Case | Input Prefix Format |
| :--- | :--- |
| **Search Query (Asymmetric)** | `task: search result \| query: {query}` |
| **Document Passage** | `title: {title} \| text: {content}` *(or `title: none \| text: {content}`)* |
| **Question Answering** | `task: question answering \| query: {question}` |
| **Code Retrieval** | `task: code retrieval \| query: {query}` |
| **Sentence Similarity (Symmetric)**| `task: sentence similarity \| query: {text}` |
| **Classification / Clustering** | `task: classification \| query: {text}` |

---

## 📄 License
Released under the [Apache 2.0 License](LICENSE). EmbeddingGemma 2 model weights are released by Google under the permissive Gemma Terms of Use.
