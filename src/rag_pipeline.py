import os
from typing import Dict, List

from openai import OpenAI

from .utils import load_config
from .vector_store import load_vector_store


SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using the provided context. "
    "Cite the sources using their source identifiers and page numbers when available. "
    "If the answer cannot be derived from the context, say you don't know."
)


def build_prompt(query: str, contexts: List[Dict]) -> str:
    context_texts = []
    for c in contexts:
        md = c["metadata"]
        src = md.get("source")
        page = md.get("page")
        prefix = f"Source: {src}"
        if page:
            prefix += f", Page: {page}"
        context_texts.append(f"{prefix}\n{c['text']}")
    joined_context = "\n\n".join(context_texts)
    return (
        f"Context:\n{joined_context}\n\n"
        f"Question: {query}\n"
        f"Answer concisely and cite sources."
    )


def answer_query(query: str, config_path: str = "config.yaml") -> Dict:
    cfg = load_config(config_path)
    top_k = int(cfg["retrieval"]["top_k"])
    vs = load_vector_store(config_path)
    results = vs.search(query, k=top_k)

    contexts: List[Dict] = []
    for idx, score in results:
        entry = vs.entries[idx]
        contexts.append({"text": entry["text"], "metadata": entry["metadata"], "score": score})

 
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(query, contexts)},
    ]

    provider = cfg["llm"]["provider"]
    model = cfg["llm"]["model"]
    temperature = float(cfg["llm"].get("temperature", 0.2))
    max_tokens = int(cfg["llm"].get("max_tokens", 512))

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not set. Please set it to use OpenAI.")
        client = OpenAI(api_key=api_key)

        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        answer = completion.choices[0].message.content
    elif provider == "ollama":
        # Use local Ollama server
        try:
            import ollama
        except Exception as e:
            raise ImportError("Ollama python client not installed. Add 'ollama' to requirements and install.") from e

        options = {"temperature": temperature}
        # Map max_tokens to Ollama's num_predict
        if max_tokens:
            options["num_predict"] = int(max_tokens)

        resp = ollama.chat(model=model, messages=messages, options=options)
        answer = resp["message"]["content"]
    else:
        raise NotImplementedError(f"Unsupported LLM provider: {provider}")
    sources = [c["metadata"] for c in contexts]
    return {"answer": answer, "sources": sources, "retrieved": contexts}