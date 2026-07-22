"""
04_vector_database.py

Author: Ram Kumar Lanke
Applied AI Solutions Architect
lankeramkumar@gmail.com

RAG STEP 4: VECTOR DATABASE
==============================
A vector database stores embeddings alongside their source metadata and
answers "nearest neighbor" queries: given a query vector, find the k
stored vectors that are most similar to it, fast. That similarity search
is the mechanism retrieval is built on.

This script implements a small but complete vector store from scratch
using nothing but numpy, so the underlying mechanics -- what "the
index" actually is, how similarity is scored, how top-k search works --
are fully visible instead of hidden behind a library. It:

  - Loads the embeddings + metadata produced by 03_embeddings.py
  - Builds an in-memory index (just a 2D numpy array + a metadata list)
  - Exposes add() / search() / save() / load(), the same shape as a real
    vector database client
  - Persists the index to disk so 05_retrieval.py can load it without
    re-embedding anything

In production you would typically reach for a purpose-built vector
database instead of hand-rolling this: FAISS (a local library, very
fast, no server) for pure similarity search at scale, or a full vector
database service like Chroma, Qdrant, Weaviate, Pinecone, or pgvector
(Postgres + a vector extension) when you also need persistence,
filtering, metadata queries, and multi-client access. The API shape
(add/search/persist) is essentially the same across all of them -- this
script's VectorStore class is deliberately built to mirror that shape.

Run:
    python scripts/04_vector_database.py
"""

import json
import os
import pickle

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "..", "output")

EMBEDDINGS_PATH = os.path.join(OUTPUT_DIR, "03_embeddings.npy")
META_PATH = os.path.join(OUTPUT_DIR, "03_embeddings_meta.json")

INDEX_VECTORS_PATH = os.path.join(OUTPUT_DIR, "04_vector_index.npy")
INDEX_META_PATH = os.path.join(OUTPUT_DIR, "04_vector_index_meta.pkl")


class VectorStore:
    """A minimal cosine-similarity vector index.

    Vectors are stored L2-normalized (both 03_embeddings.py's TF-IDF and
    sentence-transformers paths already normalize their output), which
    means cosine similarity between two vectors reduces to a plain dot
    product -- no separate norm computation needed at search time. This
    is the same trick real vector databases use internally for speed.
    """

    def __init__(self, dimension: int | None = None):
        self.dimension = dimension
        self._vectors: list[np.ndarray] = []
        self._metadata: list[dict] = []

    def __len__(self) -> int:
        return len(self._metadata)

    def add(self, vector: np.ndarray, metadata: dict) -> None:
        """Add a single (vector, metadata) pair to the index."""
        vector = np.asarray(vector, dtype=np.float32)
        if self.dimension is None:
            self.dimension = vector.shape[0]
        elif vector.shape[0] != self.dimension:
            raise ValueError(
                f"Vector dimension {vector.shape[0]} does not match index dimension {self.dimension}"
            )
        self._vectors.append(vector)
        self._metadata.append(metadata)

    def add_many(self, vectors: np.ndarray, metadata_list: list[dict]) -> None:
        for vector, metadata in zip(vectors, metadata_list):
            self.add(vector, metadata)

    def _matrix(self) -> np.ndarray:
        return np.vstack(self._vectors) if self._vectors else np.empty((0, self.dimension or 0))

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
        """Return the top_k stored items most similar to query_vector,
        each annotated with its similarity score. This is a brute-force
        O(n) scan over every stored vector -- perfectly fine at the
        99-chunk scale of this demo, but it's exactly the step that
        FAISS/HNSW-based approximate nearest neighbor indexes optimize
        for corpora with millions of vectors."""
        if not self._vectors:
            return []

        matrix = self._matrix()
        query_vector = np.asarray(query_vector, dtype=np.float32)

        # Cosine similarity via dot product, since all vectors are
        # L2-normalized. If you ever store un-normalized vectors, divide
        # by (||a|| * ||b||) here instead.
        scores = matrix @ query_vector

        top_k = min(top_k, len(scores))
        top_indices = np.argsort(-scores)[:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            results.append({
                "rank": rank + 1,
                "score": float(scores[idx]),
                **self._metadata[idx],
            })
        return results

    def save(self, vectors_path: str, meta_path: str) -> None:
        matrix = self._matrix()
        np.save(vectors_path, matrix)
        with open(meta_path, "wb") as f:
            pickle.dump({"dimension": self.dimension, "metadata": self._metadata}, f)

    @classmethod
    def load(cls, vectors_path: str, meta_path: str) -> "VectorStore":
        matrix = np.load(vectors_path)
        with open(meta_path, "rb") as f:
            payload = pickle.load(f)

        store = cls(dimension=payload["dimension"])
        store._vectors = [row for row in matrix]
        store._metadata = payload["metadata"]
        return store


def build_index_from_embeddings() -> VectorStore:
    vectors = np.load(EMBEDDINGS_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)

    store = VectorStore(dimension=payload["dimension"])
    store.add_many(vectors, payload["chunks"])
    return store


def main():
    print(f"Loading embeddings from {EMBEDDINGS_PATH} ...")
    store = build_index_from_embeddings()
    print(f"Built an in-memory vector index with {len(store)} vectors "
          f"(dimension={store.dimension}).\n")

    store.save(INDEX_VECTORS_PATH, INDEX_META_PATH)
    print(f"Persisted index to:")
    print(f"  {INDEX_VECTORS_PATH}")
    print(f"  {INDEX_META_PATH}")

    # Demonstrate a round trip: reload from disk and confirm it works.
    reloaded = VectorStore.load(INDEX_VECTORS_PATH, INDEX_META_PATH)
    print(f"\nReloaded index from disk: {len(reloaded)} vectors, dimension={reloaded.dimension}")

    # Sanity-check search using the index's own first vector as the
    # query -- the top result should be that same chunk, with a
    # similarity score of ~1.0.
    demo_query_vector = reloaded._vectors[0]
    results = reloaded.search(demo_query_vector, top_k=3)
    print(f"\nSanity search (query = index's own first vector):")
    for r in results:
        print(f"  rank {r['rank']}  score={r['score']:.3f}  "
              f"{r['filename']} (chunk {r['chunk_index']})")

    print(
        "\nNote: 05_retrieval.py performs the realistic version of this -- "
        "embedding a natural-language question with the same embedder from "
        "03_embeddings.py, then searching this same index."
    )


if __name__ == "__main__":
    main()
