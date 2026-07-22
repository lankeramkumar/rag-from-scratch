"""
05_retrieval.py

Author: Ram Kumar Lanke
Applied AI Solutions Architect
lankeramkumar@gmail.com

RAG STEP 5: RETRIEVAL
========================
Retrieval is where a user's question actually meets the index: embed
the query with the same embedder used to build the index, search the
vector store for the most similar chunks, and return them ranked. Every
earlier script exists to make this moment possible.

This script demonstrates two retrieval modes:

  1. Pure vector (semantic) search -- rank purely by embedding cosine
     similarity between the query and each chunk.

  2. Hybrid search -- blend the vector similarity score with a simple
     keyword-overlap score. Pure semantic search can miss chunks that
     share an exact, important keyword (a product name, an error code, a
     SKU) but happen to be phrased very differently; pure keyword search
     misses paraphrases. Blending both is a common, pragmatic way to get
     the benefits of each. Production systems often implement the
     keyword side with BM25 (via a library like rank_bm25 or a search
     engine such as Elasticsearch/OpenSearch); this script uses a
     lightweight term-overlap scorer to keep the dependency footprint at
     zero while demonstrating the same idea.

Run:
    python scripts/05_retrieval.py "your question here"
    python scripts/05_retrieval.py            # runs a few demo queries
"""

import json
import os
import pickle
import re
import sys
from collections import Counter

import numpy as np
from dotenv import load_dotenv

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = HERE
OUTPUT_DIR = os.path.join(HERE, "..", "output")

# If the saved embedder is a VoyageEmbedder, embedding a query below
# requires VOYAGE_API_KEY to be set -- load it from .env if present.
load_dotenv(os.path.join(HERE, "..", ".env"))

sys.path.insert(0, SCRIPTS_DIR)
import importlib

_vecdb_module = importlib.import_module("04_vector_database")
VectorStore = _vecdb_module.VectorStore

EMBEDDER_PATH = os.path.join(OUTPUT_DIR, "03_embedder.pkl")
INDEX_VECTORS_PATH = os.path.join(OUTPUT_DIR, "04_vector_index.npy")
INDEX_META_PATH = os.path.join(OUTPUT_DIR, "04_vector_index_meta.pkl")
RESULTS_PATH = os.path.join(OUTPUT_DIR, "05_retrieval_results.json")

DEFAULT_TOP_K = 4
HYBRID_VECTOR_WEIGHT = 0.75  # how much to trust semantic similarity vs. keyword overlap

DEMO_QUERIES = [
    "What is Sunrise Medical Center's protocol for treating a suspected stroke patient in the emergency room?",
    "How do I request PTO and what happens to unused time off when I leave the hospital?",
    "What should a nurse do if a medication refrigerator's temperature goes out of range?",
    "What is the RACE procedure during a hospital fire emergency?",
]


def load_embedder():
    with open(EMBEDDER_PATH, "rb") as f:
        return pickle.load(f)


def load_store() -> VectorStore:
    return VectorStore.load(INDEX_VECTORS_PATH, INDEX_META_PATH)


_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def keyword_score(query_tokens: list[str], chunk_text: str) -> float:
    """A lightweight stand-in for BM25: fraction of query terms that
    appear in the chunk, weighted a little by how often they appear.
    Returns a value roughly in [0, 1]."""
    if not query_tokens:
        return 0.0
    chunk_counts = Counter(_tokenize(chunk_text))
    if not chunk_counts:
        return 0.0

    matched = 0
    for term in set(query_tokens):
        if chunk_counts.get(term, 0) > 0:
            matched += 1
    return matched / len(set(query_tokens))


# Memoize query -> embedding within a single process. hybrid_search()
# and vector_search() both need the query vector, and 06/07 call one of
# these once per query too -- without this cache, asking the same
# question through vector + hybrid search doubles the embedding calls,
# which matters for a metered/rate-limited backend like VoyageEmbedder.
_query_vector_cache: dict[tuple[int, str], np.ndarray] = {}


def _embed_query(query: str, embedder) -> np.ndarray:
    cache_key = (id(embedder), query)
    if cache_key not in _query_vector_cache:
        # input_type="query" matters for embedders that encode queries and
        # documents differently (e.g. VoyageEmbedder); ignored by embedders
        # that don't support the distinction (TF-IDF, sentence-transformers).
        _query_vector_cache[cache_key] = embedder.transform([query], input_type="query")[0]
    return _query_vector_cache[cache_key]


def vector_search(query: str, embedder, store: VectorStore, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    query_vector = _embed_query(query, embedder)
    return store.search(query_vector, top_k=top_k)


def hybrid_search(
    query: str,
    embedder,
    store: VectorStore,
    top_k: int = DEFAULT_TOP_K,
    vector_weight: float = HYBRID_VECTOR_WEIGHT,
    candidate_pool: int = 15,
) -> list[dict]:
    """Retrieve a larger candidate pool by vector similarity, re-score
    each candidate by blending vector + keyword scores, then return the
    top_k by the blended score. Re-ranking a modest candidate pool
    (rather than the entire index) keeps this fast while still letting
    keyword signal reorder the final ranking."""
    query_vector = _embed_query(query, embedder)
    candidates = store.search(query_vector, top_k=candidate_pool)

    query_tokens = _tokenize(query)
    for candidate in candidates:
        kw_score = keyword_score(query_tokens, candidate["text"])
        candidate["vector_score"] = candidate["score"]
        candidate["keyword_score"] = kw_score
        candidate["score"] = vector_weight * candidate["vector_score"] + (1 - vector_weight) * kw_score

    candidates.sort(key=lambda c: -c["score"])
    for rank, c in enumerate(candidates[:top_k], start=1):
        c["rank"] = rank
    return candidates[:top_k]


def format_results(query: str, results: list[dict], mode: str) -> str:
    lines = [f"Query: {query!r}  (mode={mode})"]
    for r in results:
        preview = r["text"][:140].replace("\n", " ")
        score_detail = ""
        if "vector_score" in r:
            score_detail = f" [vector={r['vector_score']:.3f} keyword={r['keyword_score']:.3f}]"
        lines.append(
            f"  #{r['rank']}  score={r['score']:.3f}{score_detail}  "
            f"{r['filename']} (chunk {r['chunk_index']})\n"
            f"       {preview}..."
        )
    return "\n".join(lines)


def run_query(query: str, embedder, store: VectorStore, top_k: int = DEFAULT_TOP_K) -> dict:
    vec_results = vector_search(query, embedder, store, top_k=top_k)
    hybrid_results = hybrid_search(query, embedder, store, top_k=top_k)
    return {
        "query": query,
        "vector_search": vec_results,
        "hybrid_search": hybrid_results,
    }


def parse_cli_args(argv: list[str]) -> tuple[str | None, int]:
    """Shared CLI parsing for this script and 06_augmentation.py /
    07_generation.py, which all take the same "[--top-k N] [query...]"
    shape. --top-k lets you pull back more than the DEFAULT_TOP_K=4
    chunks in one run -- useful for a query broad enough that its most
    relevant chunks are spread across many different source documents (or
    file *types*: PDF, Word, text, image-OCR, audio- and video-transcript
    chunks all look identical to retrieval, so a high-enough top_k for a
    broad enough query will pull from all of them in one go)."""
    top_k = DEFAULT_TOP_K
    if argv and argv[0] in ("--top-k", "-k"):
        top_k = int(argv[1])
        argv = argv[2:]
    query = " ".join(argv) if argv else None
    return query, top_k


def main():
    embedder = load_embedder()
    store = load_store()
    print(f"Loaded vector index ({len(store)} chunks) and '{embedder.name}' embedder.\n")

    query, top_k = parse_cli_args(sys.argv[1:])
    queries = [query] if query else DEMO_QUERIES

    all_results = []
    for query in queries:
        result = run_query(query, embedder, store, top_k=top_k)
        all_results.append(result)

        print(format_results(query, result["vector_search"], "vector"))
        print()
        print(format_results(query, result["hybrid_search"], "hybrid"))
        print("\n" + "-" * 78 + "\n")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Saved retrieval results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
