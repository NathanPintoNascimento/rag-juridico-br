import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL

_model = None


def get_model():
    global _model
    if _model is None:
        print(f"Carregando modelo de embeddings: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Modelo carregado | Dispositivo: {device}")
    return _model


def embed_texts(texts: list[str], batch_size: int = 16) -> np.ndarray:
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings


def embed_query(query: str) -> np.ndarray:
    model = get_model()
    vec = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vec[0]
