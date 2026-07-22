"""
07_generation.py

Author: Ram Kumar Lanke
Applied AI Solutions Architect
lankeramkumar@gmail.com

RAG STEP 7: GENERATION
=========================
The final step: send the augmented prompt (retrieved context + question)
from 06_augmentation.py to a large language model and let it produce a
natural-language answer grounded in that context. This is what turns a
list of ranked document snippets into something that reads like a
direct, cited answer to the user's question.

This script calls the Claude API (model: claude-opus-4-8) via the
official `anthropic` Python SDK. If no API key is configured, it falls
back to a "mock" generation mode that assembles an answer directly from
the top retrieved chunk instead of calling the model -- so the full
pipeline (extraction -> chunking -> embedding -> vector DB -> retrieval
-> augmentation -> generation) can still be run and inspected end to end
without an API key.

Setup for real generation:
    pip install anthropic python-dotenv
    Put ANTHROPIC_API_KEY=your-api-key in a .env file at the project root
    (this script loads it automatically), or set it as a real environment
    variable instead:
        $env:ANTHROPIC_API_KEY = "your-api-key"  # PowerShell, current session only

Run:
    python scripts/07_generation.py "your question here"
    python scripts/07_generation.py            # runs the same demo queries as steps 5-6
"""

import json
import os
import sys

from dotenv import load_dotenv

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "..", "output")
PROJECT_ROOT = os.path.join(HERE, "..")

# Load ANTHROPIC_API_KEY (and anything else) from a .env file at the
# project root, if one exists. load_dotenv() does NOT overwrite a
# variable that's already set as a real environment variable -- a real
# env var always wins over .env, which is the expected precedence.
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

sys.path.insert(0, HERE)
import importlib

_augmentation_module = importlib.import_module("06_augmentation")
_retrieval_module = importlib.import_module("05_retrieval")

ANSWERS_PATH = os.path.join(OUTPUT_DIR, "07_generated_answers.json")
MODEL = "claude-opus-4-8"
MAX_TOKENS = 1024


def generate_with_claude(system_prompt: str, user_prompt: str) -> str:
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return next(block.text for block in response.content if block.type == "text")


def generate_mock(augmented: dict) -> str:
    """Fallback used when the Claude API isn't configured. Not a real
    generation step -- just enough to keep the pipeline runnable and
    show what the *shape* of a grounded answer looks like, using only
    the single top-ranked chunk rather than genuine synthesis across
    all retrieved context."""
    if not augmented["sources"]:
        return "No relevant context was found for this question."
    top = augmented["sources"][0]
    return (
        f"[MOCK ANSWER -- no ANTHROPIC_API_KEY configured, so this is not a real "
        f"LLM generation] Based on the top retrieved passage from "
        f"{top['filename']} [1], see the CONTEXT block above for the relevant "
        f"details. Set ANTHROPIC_API_KEY and re-run for a real synthesized, "
        f"cited answer."
    )


def is_api_key_configured() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def run_generation(augmented: dict) -> dict:
    use_real_model = is_api_key_configured()

    if use_real_model:
        try:
            answer = generate_with_claude(augmented["system_prompt"], augmented["user_prompt"])
            mode = "claude"
        except Exception as exc:
            print(f"  Claude API call failed ({exc.__class__.__name__}: {exc}); using mock fallback.")
            answer = generate_mock(augmented)
            mode = "mock (API error)"
    else:
        answer = generate_mock(augmented)
        mode = "mock (no API key)"

    return {
        "query": augmented["query"],
        "answer": answer,
        "mode": mode,
        "sources": augmented["sources"],
    }


def main():
    embedder = _retrieval_module.load_embedder()
    store = _retrieval_module.load_store()

    if not is_api_key_configured():
        print(
            "No ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN found in the environment.\n"
            "Running in MOCK generation mode -- see the module docstring for setup "
            "instructions to enable real Claude generation.\n"
        )
    else:
        print(f"ANTHROPIC_API_KEY detected. Generating with model '{MODEL}'.\n")

    query, top_k = _retrieval_module.parse_cli_args(sys.argv[1:])
    queries = [query] if query else _retrieval_module.DEMO_QUERIES

    results = []
    for query in queries:
        augmented = _augmentation_module.augment_query(query, embedder, store, top_k=top_k)
        result = run_generation(augmented)
        results.append(result)

        print("=" * 78)
        print(f"QUESTION: {result['query']}")
        print(f"MODE: {result['mode']}")
        print(f"\nANSWER:\n{result['answer']}")
        print("\nSOURCES:")
        for s in result["sources"]:
            print(f"  [{s['ref']}] {s['filename']} (chunk {s['chunk_index']})")
        print()

    with open(ANSWERS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(results)} generated answer(s) to {ANSWERS_PATH}")


if __name__ == "__main__":
    main()
