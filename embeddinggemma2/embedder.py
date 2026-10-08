"""
EmbeddingGemma 2 Multimodal Embedder (Vision + Text)
Optimized for Apple Silicon (M-series / MPS) using bfloat16.
"""

from typing import Union, List, Optional
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer


class MultimodalEmbeddingGemma:
    """
    Lightweight client for Google's EmbeddingGemma 2.
    Loads vision + text encoders (440M params), bypassing unused audio encoder.
    """

    def __init__(
        self,
        model_name: str = "google/embeddinggemma-2",
        device: Optional[str] = None,
        truncate_dim: Optional[int] = None,
    ):
        if device is None:
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = device

        self.truncate_dim = truncate_dim

        # Crucial: Use bfloat16. float16 suffers dynamic range overflow causing silent NaNs.
        self.torch_dtype = torch.bfloat16

        print(f"Initializing {model_name} on {self.device} (dtype={self.torch_dtype})...")
        self.model = SentenceTransformer(
            model_name,
            model_kwargs={"torch_dtype": self.torch_dtype},
            config_kwargs={"audio_config": None},  # Drop audio encoder (saves memory)
            device=self.device,
        )
        print("Model initialized successfully.")

    def embed_text(
        self,
        texts: Union[str, List[str]],
        task: Optional[str] = "SearchQuery",
        title: Optional[str] = None,
        truncate_dim: Optional[int] = None,
    ):
        """
        Embed queries or documents with appropriate task prompt prefixes.
        Task types:
          - 'SearchQuery' / 'query' (asymmetric query)
          - 'Document' / 'passage' (asymmetric document, uses title: none if no title given)
          - 'SentenceSimilarity' (symmetric comparison)
          - 'CodeRetrieval' (code search)
        """
        dim = truncate_dim or self.truncate_dim

        if title is not None:
            if isinstance(texts, str):
                formatted = f"title: {title} | text: {texts}"
            else:
                formatted = [f"title: {title} | text: {t}" for t in texts]
            prompt_name = None
        else:
            formatted = texts
            if task in ("SearchQuery", "query"):
                prompt_name = "SearchQuery"
            elif task in ("Document", "passage", "doc"):
                prompt_name = "Document"
            else:
                prompt_name = task

        return self.model.encode(
            formatted,
            prompt_name=prompt_name,
            truncate_dim=dim,
            normalize_embeddings=True,
        )

    def embed_images(
        self,
        images: Union[str, Image.Image, List[Union[str, Image.Image]]],
        truncate_dim: Optional[int] = None,
    ):
        """
        Embed images into the exact same 768-d semantic vector space.
        Accepts PIL Images or file paths. Images do not take task prefixes.
        """
        dim = truncate_dim or self.truncate_dim
        if isinstance(images, (str, Image.Image)):
            images = [images]

        loaded_images = []
        for img in images:
            if isinstance(img, str):
                loaded_images.append(Image.open(img).convert("RGB"))
            else:
                loaded_images.append(img.convert("RGB"))

        embeddings = self.model.encode(
            loaded_images,
            truncate_dim=dim,
            normalize_embeddings=True,
        )
        return embeddings[0] if len(images) == 1 else embeddings

    def similarity(self, emb1, emb2):
        """Compute cosine similarity between two embeddings or batches."""
        return self.model.similarity(emb1, emb2)
