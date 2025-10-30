import os
import json
from typing import List, Dict, Optional
from pathlib import Path

import yaml


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_config(config_path: Optional[str] = None) -> Dict:
    """Load configuration file.

    Behavior:
    - If `config_path` is provided, use it (absolute or relative).
    - If omitted, try these in order:
      1. ./config.yaml (cwd)
      2. <project_root>/config.yaml where project_root is the repo root inferred from this file (two levels up: repo/src -> repo)
      3. Walk up from cwd searching for config.yaml (up to 5 parents)

    Additionally, normalize configured paths to absolute paths rooted at the directory
    containing the located config file and create any missing directories.
    """
    # Determine config file location
    project_root = Path(__file__).resolve().parents[1]
    if config_path:
        cfg_path = Path(config_path)
        if not cfg_path.is_absolute():
            # Try several candidate locations for a relative config_path:
            # 1) cwd/config_path
            # 2) project_root/config_path
            # 3) walk up from cwd looking for config_path
            candidate_cwd = (Path.cwd() / cfg_path)
            candidate_pr = (project_root / cfg_path)
            if candidate_cwd.exists():
                cfg_path = candidate_cwd.resolve()
            elif candidate_pr.exists():
                cfg_path = candidate_pr.resolve()
            else:
                p = Path.cwd()
                found = None
                for _ in range(6):
                    maybe = p / cfg_path
                    if maybe.exists():
                        found = maybe.resolve()
                        break
                    if p.parent == p:
                        break
                    p = p.parent
                if found:
                    cfg_path = found
                else:
                    # fallback to resolving relative to cwd (will raise later)
                    cfg_path = candidate_cwd.resolve()
    else:
        # 1) cwd/config.yaml
        candidate = Path.cwd() / "config.yaml"
        if candidate.exists():
            cfg_path = candidate.resolve()
        else:
            # 2) project root relative to this file: ../.. (repo root)
            project_root = Path(__file__).resolve().parents[1]
            candidate2 = project_root / "config.yaml"
            if candidate2.exists():
                cfg_path = candidate2.resolve()
            else:
                # 3) walk up from cwd
                p = Path.cwd()
                found = None
                for _ in range(6):
                    maybe = p / "config.yaml"
                    if maybe.exists():
                        found = maybe.resolve()
                        break
                    if p.parent == p:
                        break
                    p = p.parent
                if found:
                    cfg_path = found
                else:
                    raise FileNotFoundError(
                        f"Could not find 'config.yaml'. Looked in cwd ({Path.cwd()}) and project root ({project_root})"
                    )

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # Normalize derived paths relative to the config file's parent directory
    cfg_dir = cfg_path.parent
    paths = cfg.get("paths", {})
    # Provide defaults and make absolute
    raw_dir = Path(paths.get("raw_documents_dir", "data/raw_documents"))
    eval_dir = Path(paths.get("evaluation_dir", "data/evaluation"))
    vs_dir = Path(paths.get("vector_store_dir", "data/vector_store"))

    def _abs(p: Path) -> str:
        return str((cfg_dir / p).resolve()) if not p.is_absolute() else str(p)

    raw_dir_abs = _abs(raw_dir)
    eval_dir_abs = _abs(eval_dir)
    vs_dir_abs = _abs(vs_dir)

    ensure_dir(raw_dir_abs)
    ensure_dir(eval_dir_abs)
    ensure_dir(vs_dir_abs)

    # Update the config dict so callers get absolute paths
    cfg.setdefault("paths", {})["raw_documents_dir"] = raw_dir_abs
    cfg.setdefault("paths", {})["evaluation_dir"] = eval_dir_abs
    cfg.setdefault("paths", {})["vector_store_dir"] = vs_dir_abs
    return cfg


def save_json(data: Dict, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size")
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = end - chunk_overlap
    return chunks


# Lightweight text extraction helpers
from PyPDF2 import PdfReader
from docx import Document


def read_pdf(path: str) -> List[Dict]:
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append({"page": i + 1, "text": text})
    return pages


def read_docx(path: str) -> str:
    doc = Document(path)
    return "\n".join([p.text for p in doc.paragraphs])


def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def file_extension(path: str) -> str:
    return os.path.splitext(path)[1].lower()