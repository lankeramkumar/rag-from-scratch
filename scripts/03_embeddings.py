"""
03_embeddings.py

Author: Ram Kumar Lanke
Applied AI Solutions Architect
lankeramkumar@gmail.com

RAG STEP 3: EMBEDDINGS
========================
An embedding model turns a piece of text into a dense vector of numbers
such that semantically similar texts land close together in vector
space. That property is the entire basis of vector search: instead of
matching exact keywords, we can find chunks whose *meaning* is close to
a query's meaning, even if they don't share any words.

This script embeds every chunk from 02_chunks.json and writes the result
to disk for the vector database step. It supports three embedding
backends, tried in this order (see embedding_backends.build_embedder):

  1. Voyage AI (hosted API)  -- real semantic embeddings, no local model
     if VOYAGE_API_KEY is set  download. Anthropic's recommended
                                 embeddings provider. Uses input_type=
                                 "document" here vs. "query" at retrieval
                                 time (05_retrieval.py) -- Voyage supports
                                 this asymmetric encoding natively, which
                                 improves retrieval quality over encoding
                                 both the same way.

  2. Sentence-Transformers   -- a real local "dense" neural embedding
     (if installed)            model (all-MiniLM-L6-v2). Same semantic
                                 properties as Voyage, but runs fully
                                 offline after a one-time ~90MB download.

  3. TF-IDF (scikit-learn)   -- offline, deterministic, no model download
     (always available)        at all. A classic "sparse" vectorization:
                                 each dimension is a vocabulary word,
                                 weighted by term frequency and inverse
                                 document frequency. Captures keyword
                                 overlap well but has no real understanding
                                 of synonyms or word order.

Whichever tier is unavailable (missing API key, package not installed,
no internet) is skipped automatically, so the pipeline always produces a
usable result.

Run:
    python scripts/03_embeddings.py
    # Optional upgrades from the TF-IDF baseline:
    #   put VOYAGE_API_KEY=... in .env             -- hosted, no download
    #   pip install sentence-transformers            -- local, ~1-2GB download
"""

import json
import os
import pickle
import sys

import numpy as np
from dotenv import load_dotenv

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "..", "output")
CHUNKS_PATH = os.path.join(OUTPUT_DIR, "02_chunks.json")

EMBEDDINGS_PATH = os.path.join(OUTPUT_DIR, "03_embeddings.npy")
META_PATH = os.path.join(OUTPUT_DIR, "03_embeddings_meta.json")
EMBEDDER_PATH = os.path.join(OUTPUT_DIR, "03_embedder.pkl")

load_dotenv(os.path.join(HERE, "..", ".env"))  # picks up VOYAGE_API_KEY if present

# Embedder classes live in embedding_backends.py (a normal, importable
# module) rather than being defined here, so that the embedder object we
# pickle below can be unpickled correctly by 05_retrieval.py later -- see
# the docstring at the top of embedding_backends.py for why that matters.
sys.path.insert(0, HERE)
from embedding_backends import build_embedder  # noqa: E402


def main():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [c["text"] for c in chunks]
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}\n")

    from embedding_backends import TfidfEmbedder

    embedder = build_embedder(texts)
    try:
        vectors = embedder.transform(texts, input_type="document")
    except Exception as exc:
        # The chosen backend was reachable in principle (e.g. VOYAGE_API_KEY
        # was set) but the actual embedding call failed -- bad key, rate
        # limit exhausted after retries, no internet, etc. Don't let the
        # whole pipeline crash; fall back to the always-available TF-IDF
        # backend instead.
        print(f"\n'{embedder.name}' embedding failed ({exc.__class__.__name__}: {exc})")
        print("Falling back to TfidfEmbedder (offline, scikit-learn based).")
        embedder = TfidfEmbedder()
        embedder.fit(texts)
        vectors = embedder.transform(texts, input_type="document")

    print(f"\nEmbedded {vectors.shape[0]} chunks into {vectors.shape[1]}-dimensional vectors "
          f"using '{embedder.name}'.")

    np.save(EMBEDDINGS_PATH, vectors)

    # Metadata rows are positionally aligned with `vectors` -- row i of
    # the embedding matrix corresponds to meta[i]. 04_vector_database.py
    # relies on this alignment.
    #
    # `parent_id`/`parent_text` are carried through when present (i.e. the
    # chunks came from chunk_parent_child() in 02_chunking_strategies.py)
    # so 06_augmentation.py can build context from the larger parent
    # passage instead of the small embedded child -- everything embedded
    # and searched is still just `text`, unaffected either way.
    meta = [
        {
            "chunk_id": c["chunk_id"],
            "doc_id": c["doc_id"],
            "filename": c["filename"],
            "source_type": c["source_type"],
            "chunk_index": c["chunk_index"],
            "text": c["text"],
            **({"parent_id": c["parent_id"], "parent_text": c["parent_text"]} if "parent_text" in c else {}),
        }
        for c in chunks
    ]
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump({"embedder_name": embedder.name, "dimension": vectors.shape[1], "chunks": meta}, f, indent=2, ensure_ascii=False)

    # Persist the embedder itself (fitted TF-IDF vocabulary, or just the
    # model name for sentence-transformers) so 05_retrieval.py can embed
    # a *query* with the exact same vector space as the index.
    with open(EMBEDDER_PATH, "wb") as f:
        pickle.dump(embedder, f)

    print(f"\nSaved:")
    print(f"  {EMBEDDINGS_PATH}  (embedding matrix, shape {vectors.shape})")
    print(f"  {META_PATH}  (chunk metadata, aligned by row index)")
    print(f"  {EMBEDDER_PATH}  (fitted embedder, for embedding queries consistently)")

    # A quick sanity check: two chunks that are actually similar should
    # have a higher cosine similarity (== dot product, since vectors are
    # L2-normalized) than two unrelated chunks.
    if len(vectors) >= 2:
        sims = vectors @ vectors[0]
        best_other = int(np.argsort(-sims)[1])  # skip index 0 (itself, sim=1.0)
        print(
            f"\nSanity check: chunk 0 ('{meta[0]['filename']}') is most similar to "
            f"chunk {best_other} ('{meta[best_other]['filename']}') "
            f"with cosine similarity {sims[best_other]:.3f}"
        )


if __name__ == "__main__":
    main()
