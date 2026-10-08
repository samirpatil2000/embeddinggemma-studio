"""
Multimodal Semantic Search Engine using EmbeddingGemma 2
Demonstrates:
  1. Indexing images and text documents into a unified vector index
  2. Text-to-Image Search ("find photos of a bright sunrise")
  3. Image-to-Image Search (reverse image search)
  4. Text-to-Document Search (semantic document retrieval)
"""

import os
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image
from embeddinggemma2.embedder import MultimodalEmbeddingGemma


class MultimodalSearchIndex:
    """
    Lightweight, in-memory vector index for multimodal search.
    Stores normalized 768-d (or 256-d) vectors with metadata.
    """

    def __init__(self, embedder: Optional[MultimodalEmbeddingGemma] = None, dim: int = 768):
        self.embedder = embedder or MultimodalEmbeddingGemma()
        self.dim = dim
        self.vectors: List[np.ndarray] = []
        self.metadata: List[Dict[str, Any]] = []

    def add_image(self, image_path: str, item_id: str, tags: Optional[str] = None):
        """Encode an image (no prefix) and add to index."""
        emb = self.embedder.embed_images(image_path, truncate_dim=self.dim)
        self.vectors.append(np.array(emb, dtype=np.float32))
        self.metadata.append({
            "id": item_id,
            "type": "image",
            "path": image_path,
            "description": tags or os.path.basename(image_path),
        })

    def add_document(self, text: str, item_id: str, title: Optional[str] = None):
        """Encode a document with Document task prefix and add to index."""
        emb = self.embedder.embed_text(
            text,
            task="Document",
            title=title,
            truncate_dim=self.dim,
        )
        self.vectors.append(np.array(emb, dtype=np.float32))
        self.metadata.append({
            "id": item_id,
            "type": "document",
            "title": title or "Untitled",
            "snippet": text[:120] + ("..." if len(text) > 120 else ""),
        })

    def search_by_text(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Text Query Search:
        Embeds query with SearchQuery prompt prefix and computes cosine similarity
        against ALL items in index (images and documents alike).
        """
        if not self.vectors:
            return []

        # Encode query
        q_emb = self.embedder.embed_text(query, task="SearchQuery", truncate_dim=self.dim)
        q_vec = np.array(q_emb, dtype=np.float32)

        return self._rank_and_score(q_vec, top_k)

    def search_by_image(self, query_image_path: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Image Query Search (Reverse image search / visual search):
        Embeds image directly and computes cosine similarity against stored index.
        """
        if not self.vectors:
            return []

        img_emb = self.embedder.embed_images(query_image_path, truncate_dim=self.dim)
        img_vec = np.array(img_emb, dtype=np.float32)

        return self._rank_and_score(img_vec, top_k)

    def _rank_and_score(self, query_vec: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        # Cosine similarity matrix multiplication: (N, D) @ (D,) -> (N,)
        matrix = np.vstack(self.vectors)
        scores = np.dot(matrix, query_vec)

        # Sort indices in descending order
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in ranked_indices:
            res = dict(self.metadata[idx])
            res["score"] = float(scores[idx])
            results.append(res)
        return results


def demo_search():
    print("=" * 65)
    print("DEMO: MULTIMODAL SEARCH ENGINE (EMBEDDINGGEMMA 2)")
    print("=" * 65)

    index = MultimodalSearchIndex()

    # 1. Index Sample Media (Images & Documents)
    print("\n--- 1. Indexing Corpus into Shared Vector Space ---")
    
    # Add Images
    index.add_image("embeddinggemma2/test_red_apple.png", "img_apple", "Fresh red apple with leaf")
    index.add_image("embeddinggemma2/samples/blue_circle.png", "img_ocean", "Deep blue sky and ocean sphere")
    index.add_image("embeddinggemma2/samples/yellow_sun.png", "img_sun", "Radiant morning yellow sun")

    # Add Text Documents
    index.add_document(
        "Apples are rich in fiber and vitamin C, and have been cultivated since antiquity across Central Asia.",
        "doc_nutrition",
        title="Nutrition Guide: Apples & Orchard Fruits"
    )
    index.add_document(
        "Solar radiation provides light and heat energy driving Earth's weather, ocean currents, and photosynthesis.",
        "doc_astronomy",
        title="The Sun and Solar Energy"
    )
    index.add_document(
        "Quantum superposition allows particles to exist in linear combinations of states until measurement.",
        "doc_physics",
        title="Quantum Mechanics Fundamentals"
    )

    print(f"Index complete: {len(index.vectors)} multimodal items stored.\n")

    # 2. Test Text-to-Image / Text-to-Doc Search Queries
    test_queries = [
        "A picture of the hot yellow sun shining bright",
        "Healthy fruit snacks with vitamins and dietary fiber",
        "Wave functions and atomic state probability collapse",
    ]

    for q in test_queries:
        print(f"\n🔎 Query: \"{q}\"")
        matches = index.search_by_text(q, top_k=2)
        for rank, match in enumerate(matches, 1):
            if match["type"] == "image":
                print(f"   [{rank}] (IMAGE) Score: {match['score']:.4f} | ID: {match['id']} | File: {match['path']}")
            else:
                print(f"   [{rank}] (DOC)   Score: {match['score']:.4f} | ID: {match['id']} | Title: {match['title']}")

    # 3. Test Image Query (Reverse visual search)
    print("\n🔎 Image Query: [Using apple image as query input]")
    img_matches = index.search_by_image("embeddinggemma2/test_red_apple.png", top_k=2)
    for rank, match in enumerate(img_matches, 1):
        if match["type"] == "image":
            print(f"   [{rank}] (IMAGE) Score: {match['score']:.4f} | ID: {match['id']} | File: {match['path']}")
        else:
            print(f"   [{rank}] (DOC)   Score: {match['score']:.4f} | ID: {match['id']} | Title: {match['title']}")


if __name__ == "__main__":
    demo_search()
