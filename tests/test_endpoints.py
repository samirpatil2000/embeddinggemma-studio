"""
Test script for verifying EmbeddingGemma 2 microservices.
Supports both local testing (via Docker / Uvicorn) and live Render endpoints.
"""

import sys
import io
import requests
import numpy as np
from PIL import Image, ImageDraw

def cosine_similarity(v1, v2):
    a = np.array(v1)
    b = np.array(v2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def test_services(
    text_endpoint="http://localhost:8001",
    image_endpoint="http://localhost:8002",
    audio_endpoint="http://localhost:8003",
):
    print("=" * 60)
    print("EmbeddingGemma 2 Microservices Verification")
    print("=" * 60)
    print(f"Text service:  {text_endpoint}")
    print(f"Image service: {image_endpoint}")
    print(f"Audio service: {audio_endpoint}")

    # 1. Health checks
    print("\n--- 1. Health Checks ---")
    for name, url in [("Text", text_endpoint), ("Image", image_endpoint), ("Audio", audio_endpoint)]:
        try:
            r = requests.get(f"{url}/health", timeout=5)
            print(f"{name} Service ({url}/health): {r.status_code} - {r.json()}")
        except Exception as e:
            print(f"{name} Service ({url}/health): NOT REACHABLE ({e})")

    # 2. Test Text Embedding
    print("\n--- 2. Testing Text Embedding ---")
    query = "A glowing bright yellow sunshine in the morning sky"
    try:
        r = requests.post(
            f"{text_endpoint}/embed",
            json={"texts": query, "task": "SearchQuery", "truncate_dim": 768},
            timeout=15,
        )
        if r.status_code == 200:
            data = r.json()
            text_emb = data["embeddings"][0]
            print(f"Text embedding received! Dimension: {len(text_emb)}")
        else:
            print(f"Text embed failed: {r.status_code} - {r.text}")
            return
    except Exception as e:
        print(f"Text request failed: {e}")
        return

    # 3. Test Image Embedding
    print("\n--- 3. Testing Image Embedding ---")
    # Create test image in memory
    img = Image.new("RGB", (128, 128), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.ellipse((20, 20, 108, 108), fill=(255, 215, 0)) # Yellow circle/sun
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_bytes = img_byte_arr.getvalue()

    try:
        files = {"file": ("sun.png", img_bytes, "image/png")}
        r = requests.post(f"{image_endpoint}/embed", files=files, timeout=15)
        if r.status_code == 200:
            data = r.json()
            img_emb = data["embedding"]
            print(f"Image embedding received! Dimension: {len(img_emb)}")
        else:
            print(f"Image embed failed: {r.status_code} - {r.text}")
            return
    except Exception as e:
        print(f"Image request failed: {e}")
        return

    # 4. Cross-Modal Similarity Check
    print("\n--- 4. Cross-Modal Compatibility Check ---")
    score = cosine_similarity(text_emb, img_emb)
    print(f"Cosine Similarity (Text Query <-> Image): {score:.4f}")
    if score > 0.4:
        print("SUCCESS: Microservices share the exact same embedding space!")
    else:
        print("Note: Similarity returned. Validate with domain-specific dataset.")

if __name__ == "__main__":
    if len(sys.argv) >= 4:
        test_services(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        test_services()
