import os
from typing import List, Dict, Optional

from .utils import (
    load_config,
    ensure_dir,
    chunk_text,
    read_pdf,
    read_docx,
    read_txt,
    file_extension,
)


def load_documents_from_dir(directory: str) -> List[Dict]:
    docs: List[Dict] = []
    for root, _, files in os.walk(directory):
        for fn in files:
            path = os.path.join(root, fn)
            ext = file_extension(path)
            if ext == ".pdf":
                pages = read_pdf(path)
                for p in pages:
                    docs.append({
                        "source": os.path.relpath(path, directory),
                        "page": p["page"],
                        "text": p["text"],
                        "type": "pdf",
                    })
            elif ext == ".docx":
                text = read_docx(path)
                docs.append({
                        "source": os.path.relpath(path, directory),
                        "page": None,
                        "text": text,
                        "type": "docx",
                })
            elif ext in {".txt", ".md"}:
                text = read_txt(path)
                docs.append({
                        "source": os.path.relpath(path, directory),
                        "page": None,
                        "text": text,
                        "type": "text",
                })
            else:
                continue
    return docs


def chunk_documents(docs: List[Dict], chunk_size: int, chunk_overlap: int) -> List[Dict]:
    chunks: List[Dict] = []
    for doc in docs:
        text = doc.get("text", "")
        if not text:
            continue
        parts = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for i, part in enumerate(parts):
            md = {
                "source": doc.get("source"),
                "page": doc.get("page"),
                "chunk_id": i,
                "type": doc.get("type"),
            }
            chunks.append({"text": part, "metadata": md})
    return chunks


def process_raw_documents(
    config_path: str = "config.yaml",
    chunk_size_override: Optional[int] = None,
    chunk_overlap_override: Optional[int] = None,
) -> List[Dict]:
    cfg = load_config(config_path)
    raw_dir = cfg["paths"]["raw_documents_dir"]
    ensure_dir(raw_dir)
    docs = load_documents_from_dir(raw_dir)
    chunk_cfg = cfg.get("chunking", {})
    chunk_size = int(chunk_size_override or chunk_cfg.get("chunk_size", 500))
    chunk_overlap = int(chunk_overlap_override or chunk_cfg.get("chunk_overlap", 100))
    return chunk_documents(docs, chunk_size, chunk_overlap)