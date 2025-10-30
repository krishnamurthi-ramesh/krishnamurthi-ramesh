import os
import io
import json
import time

import streamlit as st
import pandas as pd
import plotly.express as px

# Ensure project root is on sys.path so `src` imports work
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import load_config, ensure_dir
from src.data_processing import process_raw_documents
from src.vector_store import build_and_persist_vector_store, load_vector_store
from src.rag_pipeline import answer_query
from src.evaluation import evaluate_retrieval, evaluate_answer_relevance

st.set_page_config(page_title="Enterprise RAG System", layout="wide")
# Reduce noisy watcher errors from torch by switching to polling
try:
    st.set_option('server.fileWatcherType', 'poll')
except Exception:
    pass

cfg = load_config()
raw_dir = cfg["paths"]["raw_documents_dir"]
eval_dir = cfg["paths"]["evaluation_dir"]
vs_dir = cfg["paths"]["vector_store_dir"]
ensure_dir(raw_dir)
ensure_dir(eval_dir)
ensure_dir(vs_dir)

st.title("Enterprise RAG System with Comprehensive Evaluation")

with st.sidebar:
    st.header("Configuration")
    st.write("Edit config.yaml for finer control.")
    st.write(f"Embedding model: {cfg['embeddings']['model_name']}")
    st.write(f"LLM: {cfg['llm']['provider']} / {cfg['llm']['model']}")
    top_k = st.number_input("Top-K retrieval", min_value=1, max_value=20, value=int(cfg["retrieval"]["top_k"]))
    chunk_size = st.number_input("Chunk size", min_value=100, max_value=2000, value=int(cfg["chunking"]["chunk_size"]))
    chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=1000, value=int(cfg["chunking"]["chunk_overlap"]))

st.subheader("1) Upload Documents")
uploaded = st.file_uploader("Upload PDF/DOCX/TXT files", type=["pdf", "docx", "txt", "md"], accept_multiple_files=True)
if uploaded:
    for up in uploaded:
        save_path = os.path.join(raw_dir, up.name)
        with open(save_path, "wb") as f:
            f.write(up.getbuffer())
    st.success(f"Saved {len(uploaded)} file(s) to {raw_dir}")

st.subheader("2) Build / Rebuild Vector Index")
if st.button("Process & Build Index"):
    with st.spinner("Processing documents and building index..."):
        start = time.time()
        chunks = process_raw_documents(chunk_size_override=int(chunk_size), chunk_overlap_override=int(chunk_overlap))
        vs = build_and_persist_vector_store(chunks)
        dur = time.time() - start
    st.success(f"Index built with {len(chunks)} chunks in {dur:.2f}s")

st.subheader("3) Ask a Question")
question = st.text_input("Enter your question", value=cfg["ui"]["default_question"])
if st.button("Get Answer"):
    try:
        res = answer_query(question)
        st.markdown("**Answer**")
        st.write(res["answer"]) 

        st.markdown("**Sources**")
        src_df = pd.DataFrame(res["sources"]).fillna("")
        if not src_df.empty:
            st.dataframe(src_df, use_container_width=True)

        st.markdown("**Retrieved Chunks**")
        chunks_df = pd.DataFrame([
            {"score": c["score"], "source": c["metadata"].get("source"), "page": c["metadata"].get("page"), "chunk_id": c["metadata"].get("chunk_id"), "preview": c["text"][:300]}
            for c in res["retrieved"]
        ])
        st.dataframe(chunks_df, use_container_width=True)
    except Exception as e:
        st.error(f"Error answering: {e}")

st.subheader("4) Evaluation")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Retrieval Metrics**")
    if st.button("Run Retrieval Evaluation"):
        try:
            df = evaluate_retrieval()
            if df.empty:
                st.warning("No evaluation dataset found or empty. Place dataset.json under data/evaluation.")
            else:
                st.dataframe(df, use_container_width=True)
                # Plot summary metrics
                summary = df[["Recall@3", "Recall@5", "NDCG@3", "NDCG@5", "Precision@3", "Precision@5"]].mean().reset_index()
                summary.columns = ["metric", "value"]
                fig = px.bar(summary, x="metric", y="value", title="Average Retrieval Metrics")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Evaluation error: {e}")

with col2:
    st.markdown("**Answer Relevance (ragas)**")
    if st.button("Run Answer Relevance Evaluation"):
        try:
            dataset_path = os.path.join(eval_dir, "dataset.json")
            if not os.path.exists(dataset_path):
                st.warning("No evaluation dataset found. Place dataset.json under data/evaluation.")
            else:
                df_rel = evaluate_answer_relevance(dataset_path=dataset_path)
                if df_rel.empty:
                    st.warning("ragas not available or dataset missing. Install ragas and provide dataset.json.")
                else:
                    st.dataframe(df_rel, use_container_width=True)
        except Exception as e:
            st.error(f"Evaluation error: {e}")

st.caption("Tip: Provide evaluation dataset with keys: question, relevant_sources (list of source identifiers), ground_truth (optional).")