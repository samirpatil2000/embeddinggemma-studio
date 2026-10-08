# EmbeddingGemma 2 Multimodal Studio

Deployable microservices and interactive browser studio for Google's **EmbeddingGemma 2** natively multimodal embedding model.

## Overview

EmbeddingGemma 2 maps **text, images, and audio into a shared 768-dimensional vector space**. This project packages the model into:

1. **`embeddinggemma-text`**: 8-bit text embedding API (~340MB RAM).
2. **`embeddinggemma-image`**: 4-bit vision/video embedding API (~420MB RAM).
3. **`embeddinggemma-audio`**: 4-bit acoustic embedding API (~440MB RAM).
4. **`embeddinggemma-ui`**: Standalone web UI with real-time cosine similarity search.

All 3 services run within **512MB RAM constraints** on Render's Free or Starter tier.

---

## 1. Local Development

### Run Text Service:
```bash
uvicorn services.text_service.app:app --port 8001
```

### Run Image Service:
```bash
uvicorn services.image_service.app:app --port 8002
```

### Launch Interactive UI:
Open `ui/index.html` directly in any web browser!

---

## 2. Deploy to Render via Blueprint

1. Push this repository to GitHub.
2. Go to [Render Dashboard $\rightarrow$ Blueprints](https://dashboard.render.com/blueprints/new).
3. Connect this repository and click **Apply Blueprint**.

Render automatically provisions:
- `https://embeddinggemma-text.onrender.com`
- `https://embeddinggemma-image.onrender.com`
- `https://embeddinggemma-audio.onrender.com`
- `https://embeddinggemma-ui.onrender.com`

---

## 3. Verify Endpoints

Run the automated test suite:
```bash
python tests/test_endpoints.py
```
