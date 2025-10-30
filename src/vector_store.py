import os
import json
from typing import List, Dict, Tuple

import numpy as np
import faiss

from .utils import load_config, ensure_dir


class VectorStore:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.embedder = None  # Lazy init to reduce startup issues
        self.index: faiss.Index = None
        # Store entries with text and metadata for reconstruction
        self.entries: List[Dict] = []  # [{"text": str, "metadata": dict}]

    def build(self, chunks: List[Dict]) -> None:
        # Lazy import to avoid streamlit watcher issues at startup
        from sentence_transformers import SentenceTransformer
        if self.embedder is None:
            self.embedder = SentenceTransformer(self.model_name)
        texts = [c["text"] for c in chunks]
        self.entries = chunks
        embeddings = self.embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype(np.float32))

    def save(self, index_path: str, metadata_path: str) -> None:
        ensure_dir(os.path.dirname(index_path))
        ensure_dir(os.path.dirname(metadata_path))
        faiss.write_index(self.index, index_path)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.entries, f, ensure_ascii=False, indent=2)

    def load(self, index_path: str, metadata_path: str) -> None:
        self.index = faiss.read_index(index_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.entries = json.load(f)

    def search(self, query: str, k: int) -> List[Tuple[int, float]]:
        # Ensure embedder is ready (needed for query embedding)
        if self.embedder is None:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer(self.model_name)
        q_emb = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        scores, idxs = self.index.search(q_emb, k)
        results = []
        for i, score in zip(idxs[0], scores[0]):
            if i == -1:
                continue
            results.append((int(i), float(score)))
        return results


def ensure_vector_store(config_path: str = "config.yaml") -> Tuple[VectorStore, Dict]:
    cfg = load_config(config_path)
    model_name = cfg["embeddings"]["model_name"]
    vs = VectorStore(model_name=model_name)
    return vs, cfg


def build_and_persist_vector_store(chunks: List[Dict], config_path: str = "config.yaml") -> VectorStore:
    vs, cfg = ensure_vector_store(config_path)
    vs.build(chunks)
    paths = cfg["paths"]
    vs.save(paths["faiss_index_path"], paths["faiss_metadata_path"])
    return vs


def load_vector_store(config_path: str = "config.yaml") -> VectorStore:
    vs, cfg = ensure_vector_store(config_path)
    paths = cfg["paths"]
    index_path = paths["faiss_index_path"]
    metadata_path = paths["faiss_metadata_path"]
    if os.path.exists(index_path) and os.path.exists(metadata_path):
        vs.load(index_path, metadata_path)
    else:
        raise FileNotFoundError("Vector store index or metadata not found. Build it first.")
    return vs