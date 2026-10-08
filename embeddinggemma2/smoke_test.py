import os
import time
import torch
from PIL import Image, ImageDraw
from sentence_transformers import SentenceTransformer

def run_smoke_test():
    print("=" * 60)
    print("EmbeddingGemma 2 (Vision + Text) Smoke Test on Apple Silicon M5")
    print("=" * 60)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Target device: {device}")
    print(f"Torch dtype: torch.bfloat16 (safe precision for EmbeddingGemma 2)")

    # Create a simple test image: a red circle on white background representing an apple/fruit
    img_path = "embeddinggemma2/test_red_apple.png"
    img = Image.new("RGB", (256, 256), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.ellipse((48, 48, 208, 208), fill=(220, 30, 30), outline=(150, 0, 0), width=3)
    # Green leaf on top
    draw.polygon([(128, 48), (145, 20), (160, 45)], fill=(34, 139, 34))
    img.save(img_path)
    print(f"Generated test image: {img_path}")

    # Load model with audio encoder disabled (vision + text only, ~440M params)
    print("\nLoading google/embeddinggemma-2 (vision + text, audio_config=None)...")
    start_load = time.time()
    
    # We load with torch.bfloat16 to prevent float16 dynamic range overflow
    model = SentenceTransformer(
        "google/embeddinggemma-2",
        model_kwargs={"torch_dtype": torch.bfloat16},
        config_kwargs={"audio_config": None},
        device=device
    )
    load_time = time.time() - start_load
    print(f"Model loaded successfully in {load_time:.2f} seconds!")

    # 1. Test Text Embeddings
    print("\n--- 1. Testing Text Embeddings ---")
    query = "A fresh red apple with a green leaf"
    unrelated_query = "Quantum mechanics and wave function collapse in physics"

    t0 = time.time()
    q_emb = model.encode(query, prompt_name="SearchQuery", normalize_embeddings=True)
    unrelated_emb = model.encode(unrelated_query, prompt_name="SearchQuery", normalize_embeddings=True)
    text_time = (time.time() - t0) * 1000

    print(f"Text encoding latency: {text_time:.1f} ms")
    print(f"Query embedding shape: {q_emb.shape}, norm: {float(torch.linalg.norm(torch.tensor(q_emb))):.4f}")
    assert not torch.isnan(torch.tensor(q_emb)).any(), "Query embedding contains NaN!"

    # 2. Test Image Embedding
    print("\n--- 2. Testing Image Embedding ---")
    t0 = time.time()
    img_emb = model.encode(img, normalize_embeddings=True)
    img_time = (time.time() - t0) * 1000

    print(f"Image encoding latency: {img_time:.1f} ms")
    print(f"Image embedding shape: {img_emb.shape}, norm: {float(torch.linalg.norm(torch.tensor(img_emb))):.4f}")
    assert not torch.isnan(torch.tensor(img_emb)).any(), "Image embedding contains NaN!"

    # 3. Cross-Modal Similarity Check
    print("\n--- 3. Cross-Modal Similarity (Text <-> Image) ---")
    sim_match = float(model.similarity(q_emb, img_emb)[0][0])
    sim_unrelated = float(model.similarity(unrelated_emb, img_emb)[0][0])

    print(f"Similarity ('{query}' <-> Image): {sim_match:.4f}")
    print(f"Similarity ('{unrelated_query}' <-> Image): {sim_unrelated:.4f}")

    if sim_match > sim_unrelated:
        print("PASS: Relevant text has significantly higher semantic similarity to the image!")
    else:
        print("WARNING: Relevant text did not score higher than unrelated text.")

    # 4. Matryoshka Truncation (MRL) Check (e.g. 256 dimensions)
    print("\n--- 4. Matryoshka Representation Learning (256 dims) ---")
    q_emb_256 = model.encode(query, prompt_name="SearchQuery", truncate_dim=256, normalize_embeddings=True)
    img_emb_256 = model.encode(img, truncate_dim=256, normalize_embeddings=True)
    sim_256 = float(model.similarity(q_emb_256, img_emb_256)[0][0])
    print(f"Truncated shapes: {q_emb_256.shape}, {img_emb_256.shape}")
    print(f"Truncated similarity (256d): {sim_256:.4f} (Original 768d: {sim_match:.4f})")

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_smoke_test()
