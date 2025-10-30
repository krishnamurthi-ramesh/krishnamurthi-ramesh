import os
import time
import json
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from .utils import load_config
from .vector_store import load_vector_store
from .rag_pipeline import answer_query


def dcg(relevances: List[float]) -> float:
    return sum((rel / np.log2(i + 2)) for i, rel in enumerate(relevances))


def ndcg_at_k(relevances: List[float], k: int) -> float:
    rel_k = relevances[:k]
    ideal = sorted(relevances, reverse=True)[:k]
    return (dcg(rel_k) / (dcg(ideal) or 1.0))


def precision_at_k(relevances: List[int], k: int) -> float:
    rel_k = relevances[:k]
    return sum(rel_k) / max(k, 1)


def recall_at_k(relevances: List[int], total_relevant: int, k: int) -> float:
    rel_k = relevances[:k]
    return (sum(rel_k) / max(total_relevant, 1))


def load_eval_dataset(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def evaluate_retrieval(config_path: str = "config.yaml", dataset_path: str = None) -> pd.DataFrame:
    cfg = load_config(config_path)
    vs = load_vector_store(config_path)

    if dataset_path is None:
        dataset_path = os.path.join(cfg["paths"]["evaluation_dir"], "dataset.json")
    data = load_eval_dataset(dataset_path)
    if not data:
        return pd.DataFrame()

    rows = []
    for item in data:
        q = item["question"]
        relevant_sources = set(item.get("relevant_sources", []))
        top_k = int(cfg["retrieval"]["top_k"])

        start = time.time()
        results = vs.search(q, k=top_k)
        latency = time.time() - start

        retrieved_md = [vs.entries[idx]["metadata"] for idx, _ in results]
        retrieved_sources = [md.get("source") for md in retrieved_md]
        relevances_binary = [1 if s in relevant_sources else 0 for s in retrieved_sources]

        # For NDCG, we treat binary relevance as weights
        rows.append({
            "question": q,
            "Recall@1": recall_at_k(relevances_binary, len(relevant_sources), 1),
            "Recall@3": recall_at_k(relevances_binary, len(relevant_sources), 3),
            "Recall@5": recall_at_k(relevances_binary, len(relevant_sources), 5),
            "NDCG@3": ndcg_at_k(relevances_binary, 3),
            "NDCG@5": ndcg_at_k(relevances_binary, 5),
            "Precision@3": precision_at_k(relevances_binary, 3),
            "Precision@5": precision_at_k(relevances_binary, 5),
            "Latency_s": latency,
        })

    df = pd.DataFrame(rows)
    return df


def evaluate_answer_relevance(config_path: str = "config.yaml", dataset_path: str = None) -> pd.DataFrame:
    """Optional relevance scoring using ragas if installed.
    Dataset items should include keys: question, ground_truth.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import answer_relevancy
    except Exception:
        # ragas not available; return empty frame
        return pd.DataFrame()

    cfg = load_config(config_path)
    if dataset_path is None:
        dataset_path = os.path.join(cfg["paths"]["evaluation_dir"], "dataset.json")
    data = load_eval_dataset(dataset_path)
    if not data:
        return pd.DataFrame()

    records = []
    for item in data:
        q = item["question"]
        gt = item.get("ground_truth", "")
        try:
            ans = answer_query(q, config_path=config_path)["answer"]
        except Exception:
            ans = ""
        records.append({"question": q, "answer": ans, "ground_truth": gt})

    # ragas expects a dataset with keys: question, answer, ground_truth
    # We evaluate answer relevancy metric
    try:
        res = evaluate(pd.DataFrame(records), metrics=[answer_relevancy])
        return res
    except Exception:
        return pd.DataFrame(records)