"""
Comprehensive Local API Test for EmbeddingGemma 2 Microservices.
Tests endpoints via FastAPI TestClient directly without external network dependency.
"""

import io
import numpy as np
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient

def cosine_similarity(v1, v2):
    a = np.array(v1)
    b = np.array(v2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def run_api_tests():
    print("=" * 65)
    print("TESTING EMBEDDINGGEMMA 2 MICROSERVICES (LOCAL FASTAPI TESTCLIENT)")
    print("=" * 65)

    # -------------------------------------------------------------
    # 1. TEST TEXT SERVICE
    # -------------------------------------------------------------
    print("\n[1/3] Testing Text Embedding Service (deploy/text_service/app.py)...")
    from deploy.text_service.app import app as text_app

    with TestClient(text_app) as text_client:
        # Health check
        h_resp = text_client.get("/health")
        print(f"  GET /health: {h_resp.status_code} -> {h_resp.json()}")
        assert h_resp.status_code == 200

        # Query embedding (768d)
        q_apple = "A fresh red apple with a green leaf"
        resp_q1 = text_client.post(
            "/embed",
            json={"texts": q_apple, "task": "SearchQuery", "truncate_dim": 768},
        )
        assert resp_q1.status_code == 200, f"Error: {resp_q1.text}"
        data_q1 = resp_q1.json()
        emb_q_apple = data_q1["embeddings"][0]
        print(f"  POST /embed ('{q_apple}'): dim={len(emb_q_apple)} count={data_q1['count']}")

        # Unrelated query embedding
        q_physics = "Quantum mechanics wave collapse and general relativity"
        resp_q2 = text_client.post(
            "/embed",
            json={"texts": q_physics, "task": "SearchQuery", "truncate_dim": 768},
        )
        assert resp_q2.status_code == 200
        emb_q_physics = resp_q2.json()["embeddings"][0]
        print(f"  POST /embed ('{q_physics}'): dim={len(emb_q_physics)}")

        # Document embedding with title
        resp_doc = text_client.post(
            "/embed",
            json={
                "texts": "Apples are widely cultivated across temperate regions.",
                "title": "Domestic Apple Cultivation",
                "task": "Document",
                "truncate_dim": 768,
            },
        )
        assert resp_doc.status_code == 200
        print(f"  POST /embed (Document with title): dim={resp_doc.json()['dimension']}")

        # MRL 256-d Truncation
        resp_mrl = text_client.post(
            "/embed",
            json={"texts": q_apple, "task": "SearchQuery", "truncate_dim": 256},
        )
        assert resp_mrl.status_code == 200
        emb_q_256 = resp_mrl.json()["embeddings"][0]
        print(f"  POST /embed (MRL 256d truncation): dim={len(emb_q_256)}")
        assert len(emb_q_256) == 256

    # -------------------------------------------------------------
    # 2. TEST IMAGE SERVICE
    # -------------------------------------------------------------
    print("\n[2/3] Testing Image Embedding Service (deploy/image_service/app.py)...")
    from deploy.image_service.app import app as image_app

    # Generate sample apple image in memory
    img = Image.new("RGB", (256, 256), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.ellipse((48, 48, 208, 208), fill=(220, 30, 30))
    draw.polygon([(128, 48), (145, 20), (160, 45)], fill=(34, 139, 34))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    with TestClient(image_app) as img_client:
        # Health check
        h_resp = img_client.get("/health")
        print(f"  GET /health: {h_resp.status_code} -> {h_resp.json()}")
        assert h_resp.status_code == 200

        # Upload image (768d)
        resp_img = img_client.post(
            "/embed",
            files={"file": ("apple.png", img_bytes, "image/png")},
            data={"truncate_dim": 768},
        )
        assert resp_img.status_code == 200, f"Error: {resp_img.text}"
        data_img = resp_img.json()
        emb_img_768 = data_img["embedding"]
        print(f"  POST /embed (Image multipart upload): dim={len(emb_img_768)} format={data_img['format']}")

        # Upload image with MRL (256d)
        resp_img_256 = img_client.post(
            "/embed",
            files={"file": ("apple.png", img_bytes, "image/png")},
            data={"truncate_dim": 256},
        )
        assert resp_img_256.status_code == 200
        emb_img_256 = resp_img_256.json()["embedding"]
        print(f"  POST /embed (Image MRL 256d): dim={len(emb_img_256)}")
        assert len(emb_img_256) == 256

    # -------------------------------------------------------------
    # 3. CROSS-SERVICE SIMILARITY VERIFICATION
    # -------------------------------------------------------------
    print("\n[3/3] Cross-Service Embedding Alignment (Text Service <-> Image Service)...")
    sim_match = cosine_similarity(emb_q_apple, emb_img_768)
    sim_unrelated = cosine_similarity(emb_q_physics, emb_img_768)
    sim_mrl_256 = cosine_similarity(emb_q_256, emb_img_256)

    print(f"  Cosine Similarity (Apple Text Query <-> Apple Image): {sim_match:.4f}")
    print(f"  Cosine Similarity (Physics Text Query <-> Apple Image): {sim_unrelated:.4f}")
    print(f"  Cosine Similarity at 256d (MRL Truncation):           {sim_mrl_256:.4f}")

    assert sim_match > sim_unrelated, "Semantic mismatch: Apple text should score higher than physics!"
    print(f"  Delta: +{(sim_match - sim_unrelated):.4f} margin for relevant match!")

    print("\n" + "=" * 65)
    print("ALL API ENDPOINTS TESTED AND VERIFIED SUCCESSFULLY!")
    print("=" * 65)

if __name__ == "__main__":
    run_api_tests()
