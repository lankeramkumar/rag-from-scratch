"""
embedding_backends.py

Author: Ram Kumar Lanke
Applied AI Solutions Architect
lankeramkumar@gmail.com

Shared embedder classes used by 03_embeddings.py (to build the index) and
05_retrieval.py (to embed queries against that same index).

This lives in its own module -- rather than being defined inline inside
03_embeddings.py -- for a specific, easy-to-miss reason: 03_embeddings.py
pickles a fitted embedder instance to disk so later scripts can reuse the
exact same vector space. Python's pickle format stores a class by
*import path* (module name + class name), not by value. If TfidfEmbedder
were defined inside a script that's run directly (where Python labels it
module "__main__"), unpickling it from a *different* script would fail
with "Can't get attribute 'TfidfEmbedder' on <module '__main__'>", because
that other script's __main__ doesn't contain the class. Defining it here,
in a normally-imported module, gives it a stable import path that works
the same way regardless of which script does the unpickling.

Every embedder exposes the same interface:
    fit(texts)                                   -- one-time setup (no-op for hosted/pretrained models)
    transform(texts, input_type=None) -> ndarray  -- embed a batch of texts
    dimension                                     -- embedding vector size

`input_type` ("document" or "query") lets embedders that support
*asymmetric* embeddings -- a separate encoding for things being indexed
vs. things being searched for -- take advantage of that. Voyage AI's API
supports this natively and it measurably improves retrieval quality.
Embedders that don't support it (TF-IDF, sentence-transformers) simply
ignore the argument, so callers can pass it unconditionally.
"""

import os

import numpy as np

SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
VOYAGE_MODEL = "voyage-3.5"


class TfidfEmbedder:
    """Sparse TF-IDF embeddings, L2-normalized so that cosine similarity
    reduces to a plain dot product."""

    name = "tfidf"

    def __init__(self, max_features: int = 4000):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=(1, 2),
            norm="l2",
        )
        self._fitted = False

    def fit(self, texts: list[str]) -> None:
        self.vectorizer.fit(texts)
        self._fitted = True

    def transform(self, texts: list[str], input_type: str | None = None) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder.fit() must be called before transform()")
        return self.vectorizer.transform(texts).toarray().astype(np.float32)

    @property
    def dimension(self) -> int:
        return len(self.vectorizer.vocabulary_)


class SentenceTransformerEmbedder:
    """Thin wrapper around a real neural sentence-embedding model."""

    name = "sentence-transformers"

    def __init__(self, model_name: str = SENTENCE_TRANSFORMER_MODEL):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def fit(self, texts: list[str]) -> None:
        pass

    def transform(self, texts: list[str], input_type: str | None = None) -> np.ndarray:
        return self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()


class VoyageEmbedder:
    """Hosted embeddings via the Voyage AI API (Anthropic's recommended
    embeddings provider). No local model download -- each transform()
    call is an API request.

    Deliberately does NOT hold a `voyageai.Client` instance as an
    attribute: this object gets pickled to disk (03_embedder.pkl) so
    05_retrieval.py can reuse it, and a live network client typically
    isn't cleanly picklable. Instead, transform() creates a short-lived
    client per call. voyageai.Client() reads VOYAGE_API_KEY from the
    environment on its own, so nothing secret ends up stored in the
    pickle file either.

    Voyage embeddings are supplied pre-normalized (L2 norm == 1), which
    is what 04_vector_database.py's cosine-similarity-via-dot-product
    search relies on.
    """

    name = "voyage"

    def __init__(self, model_name: str = VOYAGE_MODEL):
        self.model_name = model_name
        self._dimension: int | None = None

    def fit(self, texts: list[str]) -> None:
        pass  # hosted, pretrained model -- nothing to fit

    def transform(self, texts: list[str], input_type: str | None = None) -> np.ndarray:
        import voyageai

        # max_retries lets the SDK's built-in tenacity-based backoff absorb
        # 429s from accounts without a billing method on file, which Voyage
        # caps at 3 requests/minute -- see the note in build_embedder().
        client = voyageai.Client(max_retries=5)  # reads VOYAGE_API_KEY from the environment
        # Voyage batches requests server-side; keep client-side batches to a
        # sane size so a large corpus doesn't hit request-size limits.
        batch_size = 128
        all_embeddings: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            result = client.embed(
                batch,
                model=self.model_name,
                input_type=input_type or "document",
            )
            all_embeddings.extend(result.embeddings)

        vectors = np.array(all_embeddings, dtype=np.float32)
        self._dimension = vectors.shape[1]
        return vectors

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            raise RuntimeError("VoyageEmbedder.dimension is only known after calling transform() at least once")
        return self._dimension


def build_embedder(texts: list[str]):
    """Preference order: Voyage AI (hosted, real semantic embeddings, no
    local download -- used if VOYAGE_API_KEY is set) -> sentence-
    transformers (real semantic embeddings, but a ~1-2GB local model
    download) -> TF-IDF (always works, fully offline, keyword-based
    rather than truly semantic). Each tier is only attempted if the one
    before it isn't usable, so the pipeline always produces a working
    embedder.

    Note: this does NOT make a network call to validate the Voyage
    backend -- it just checks that VOYAGE_API_KEY is set and returns the
    embedder unfitted (fit() is a no-op for a hosted, pretrained model
    anyway). Failures (bad key, rate limits, no internet) surface on the
    first real transform() call instead; callers should catch and fall
    back to TfidfEmbedder there if needed. This avoids spending an extra
    API request just to "test" the connection -- worth doing deliberately
    on accounts with low rate limits (e.g. Voyage's free tier, capped at
    3 requests/minute until a payment method is added)."""
    if os.environ.get("VOYAGE_API_KEY"):
        print(f"Using VoyageEmbedder ({VOYAGE_MODEL}, hosted API)")
        return VoyageEmbedder()

    try:
        embedder = SentenceTransformerEmbedder()
        embedder.fit(texts)
        print(f"Using SentenceTransformerEmbedder ({SENTENCE_TRANSFORMER_MODEL})")
        return embedder
    except Exception as exc:
        print(f"sentence-transformers unavailable ({exc.__class__.__name__}: {exc})")

    print("Falling back to TfidfEmbedder (offline, scikit-learn based).")
    embedder = TfidfEmbedder()
    embedder.fit(texts)
    return embedder
