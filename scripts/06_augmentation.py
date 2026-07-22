"""
06_augmentation.py

Author: Ram Kumar Lanke
Applied AI Solutions Architect
lankeramkumar@gmail.com

RAG STEP 6: AUGMENTATION
===========================
Augmentation is the "A" in RAG: taking the chunks retrieval found and
assembling them, together with the user's question, into a single
prompt that gives a language model the grounding it needs to answer
accurately. Retrieval alone just returns ranked text snippets -- it's
augmentation that turns those snippets into something a model can
actually use.

This step matters more than it looks. Concerns handled here:

  - Formatting each chunk so its source is identifiable (for citations)
  - Deduplicating near-identical chunks that would waste context
  - Enforcing a context budget so the prompt doesn't exceed a
    reasonable size, dropping the lowest-ranked chunks first
  - Giving the model explicit instructions on how to use (and cite) the
    provided context, including what to do if the context doesn't
    contain the answer -- this is the main lever for reducing
    hallucination in a RAG system

Run:
    python scripts/06_augmentation.py "your question here"
    python scripts/06_augmentation.py            # runs the same demo queries as step 5
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "..", "output")

sys.path.insert(0, HERE)
import importlib

_retrieval_module = importlib.import_module("05_retrieval")

AUGMENTED_PATH = os.path.join(OUTPUT_DIR, "06_augmented_prompts.json")

# Rough context budget for the assembled prompt, in characters (not
# tokens, to avoid a tokenizer dependency -- see the note in
# 02_chunking_strategies.py about the ~4-chars-per-token rule of thumb).
MAX_CONTEXT_CHARS = 3000

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the \
provided context. Follow these rules:

1. Base your answer strictly on the information in the CONTEXT section below.
2. If the context does not contain enough information to answer the \
question, say so explicitly rather than guessing or using outside knowledge.
3. Cite your sources inline using the bracketed reference numbers, e.g. [1], \
that correspond to the CONTEXT entries.
4. Be concise and directly answer what was asked."""


def _context_text(chunk: dict) -> str:
    """The text actually shown to the LLM for this chunk. Ordinarily
    that's just the chunk's own text -- but parent-child chunks
    (chunk_parent_child() in 02_chunking_strategies.py) were matched by a
    small child embedding that's too narrow to answer from on its own, so
    for those we substitute the larger `parent_text` the child was cut
    from. Retrieval still ran on the precise child; only what the model
    reads changes."""
    return chunk.get("parent_text") or chunk["text"]


def deduplicate_chunks(chunks: list[dict]) -> list[dict]:
    """Drop chunks with near-identical text (e.g. heavy overlap from the
    fixed_size_overlap strategy could otherwise show up twice). Uses a
    simple normalized-text set rather than a fuzzy-match algorithm to
    keep this dependency-free.

    Deduplicates on the *context* text, not the raw chunk text: two
    parent-child children can easily be siblings cut from the same
    parent, which would otherwise expand into the same passage twice."""
    seen = set()
    deduped = []
    for chunk in chunks:
        key = " ".join(_context_text(chunk).split())[:200]  # normalize whitespace, compare a prefix
        if key in seen:
            continue
        seen.add(key)
        deduped.append(chunk)
    return deduped


def fit_to_budget(chunks: list[dict], max_chars: int) -> list[dict]:
    """Keep chunks in rank order until the running character total would
    exceed the budget, then stop. Chunks are already ranked by
    relevance, so this keeps the most relevant material and drops the
    weakest matches first -- exactly what you want under a token/context
    budget. Budgets against the context text (parent text, if any) since
    that's what actually consumes the budget once rendered."""
    kept = []
    total = 0
    for chunk in chunks:
        chunk_len = len(_context_text(chunk))
        if total + chunk_len > max_chars and kept:
            break
        kept.append(chunk)
        total += chunk_len
    return kept


def build_context_block(chunks: list[dict]) -> str:
    """Render chunks as numbered, source-attributed context entries. The
    numbering here is what the model is instructed (in SYSTEM_PROMPT) to
    cite back in its answer, giving the user a way to trace a claim back
    to its source document."""
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        lines.append(
            f"[{i}] (source: {chunk['filename']}, chunk {chunk['chunk_index']})\n{_context_text(chunk)}"
        )
    return "\n\n".join(lines)


def build_augmented_prompt(query: str, chunks: list[dict]) -> dict:
    deduped = deduplicate_chunks(chunks)
    budgeted = fit_to_budget(deduped, MAX_CONTEXT_CHARS)
    context_block = build_context_block(budgeted)

    user_prompt = f"""CONTEXT:
{context_block}

QUESTION:
{query}

Answer the question using only the context above, with inline citations like [1]."""

    return {
        "query": query,
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "num_chunks_retrieved": len(chunks),
        "num_chunks_after_dedup": len(deduped),
        "num_chunks_used": len(budgeted),
        "context_char_count": len(context_block),
        "sources": [
            {"ref": i, "filename": c["filename"], "chunk_index": c["chunk_index"]}
            for i, c in enumerate(budgeted, start=1)
        ],
    }


def augment_query(query: str, embedder, store, top_k: int = 4, mode: str = "hybrid") -> dict:
    """Run retrieval (step 5) then augmentation (this step) for a single
    query. This is the function 07_generation.py calls."""
    if mode == "hybrid":
        chunks = _retrieval_module.hybrid_search(query, embedder, store, top_k=top_k)
    else:
        chunks = _retrieval_module.vector_search(query, embedder, store, top_k=top_k)
    return build_augmented_prompt(query, chunks)


def main():
    embedder = _retrieval_module.load_embedder()
    store = _retrieval_module.load_store()
    print(f"Loaded vector index ({len(store)} chunks) and '{embedder.name}' embedder.\n")

    query, top_k = _retrieval_module.parse_cli_args(sys.argv[1:])
    queries = [query] if query else _retrieval_module.DEMO_QUERIES

    results = []
    for query in queries:
        augmented = augment_query(query, embedder, store, top_k=top_k)
        results.append(augmented)

        print("=" * 78)
        print(f"QUERY: {query}")
        print(
            f"  retrieved={augmented['num_chunks_retrieved']}  "
            f"after_dedup={augmented['num_chunks_after_dedup']}  "
            f"used={augmented['num_chunks_used']}  "
            f"context_chars={augmented['context_char_count']}"
        )
        print("\n--- SYSTEM PROMPT ---")
        print(augmented["system_prompt"])
        print("\n--- USER PROMPT (sent to the LLM) ---")
        print(augmented["user_prompt"])
        print()

    with open(AUGMENTED_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(results)} augmented prompt(s) to {AUGMENTED_PATH}")
    print("Next: python scripts/07_generation.py  -- sends these prompts to Claude.")


if __name__ == "__main__":
    main()
