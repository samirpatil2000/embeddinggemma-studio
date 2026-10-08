import os
import torch
import numpy as np
from PIL import Image, ImageDraw
from embeddinggemma2.embedder import MultimodalEmbeddingGemma

def test_pipeline():
    embedder = MultimodalEmbeddingGemma()

    # Generate 2 distinct images
    os.makedirs("embeddinggemma2/samples", exist_ok=True)
    
    # 1. Blue circle (sky/ocean token)
    blue_img = Image.new("RGB", (128, 128), color=(255, 255, 255))
    draw = ImageDraw.Draw(blue_img)
    draw.ellipse((20, 20, 108, 108), fill=(30, 144, 255))
    blue_path = "embeddinggemma2/samples/blue_circle.png"
    blue_img.save(blue_path)

    # 2. Golden star / yellow sun
    sun_img = Image.new("RGB", (128, 128), color=(255, 255, 255))
    draw = ImageDraw.Draw(sun_img)
    draw.ellipse((20, 20, 108, 108), fill=(255, 215, 0))
    sun_path = "embeddinggemma2/samples/yellow_sun.png"
    sun_img.save(sun_path)

    # Embed images
    img_embs = embedder.embed_images([blue_path, sun_path])
    
    # Queries
    queries = [
        "A clear blue sky or bright blue sphere",
        "A bright warm yellow radiant sunshine",
    ]
    q_embs = embedder.embed_text(queries, task="SearchQuery")

    sim_matrix = embedder.similarity(q_embs, img_embs)
    print("\nCosine Similarity Matrix [Queries x Images]:")
    print(f"               Blue Circle  |  Yellow Sun")
    print(f"Blue Query:     {float(sim_matrix[0][0]):.4f}     |   {float(sim_matrix[0][1]):.4f}")
    print(f"Sun Query:      {float(sim_matrix[1][0]):.4f}     |   {float(sim_matrix[1][1]):.4f}")

    assert sim_matrix[0][0] > sim_matrix[0][1], "Blue query should match blue circle best!"
    assert sim_matrix[1][1] > sim_matrix[1][0], "Sun query should match sun image best!"
    print("\nVerification Passed: Embeddings match correct visual concepts across modalities!")

if __name__ == "__main__":
    test_pipeline()
