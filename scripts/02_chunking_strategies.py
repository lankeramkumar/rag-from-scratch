"""
02_chunking_strategies.py

Author: Ram Kumar Lanke
Applied AI Solutions Architect
lankeramkumar@gmail.com

RAG STEP 2: CHUNKING STRATEGIES
=================================
Embedding models and LLM context windows both have limits, so a whole
document usually can't be treated as a single retrievable unit. Chunking
splits each document into smaller pieces that (a) fit comfortably in an
embedding model's input size, and (b) are small enough that a retrieved
chunk is *specific* -- if a whole 4,000-word document is "the chunk," a
query about one paragraph still drags in everything else.

This script implements six chunking strategies of increasing
sophistication over the corpus produced by 01_text_extraction.py, and
prints a comparison so you can see how the choice of strategy changes
the shape of the resulting chunks:

  1. Fixed-size character chunking      -- dumbest, fastest, breaks words/sentences
  2. Fixed-size with sliding overlap    -- like #1, but preserves context across chunk edges
  3. Sentence-based chunking            -- never splits a sentence in half
  4. Recursive / paragraph-aware        -- prefers paragraph boundaries, falls back to sentences, then chars
  5. Similarity-based ("semantic")      -- groups adjacent sentences until topic similarity drops
  6. Parent-child (small-to-big)        -- embeds small "child" chunks for precise matching, but each
                                            child carries its larger "parent" chunk's text for generation

Strategy #6 addresses a tension in every strategy above it: small chunks
match narrow queries more precisely (an embedding of one paragraph isn't
diluted by three unrelated neighboring paragraphs), but small chunks often
don't carry *enough* surrounding context for an LLM to answer well from.
Parent-child chunking gets both by splitting each document twice -- once
into large parent chunks, then each parent into small child chunks -- and
keeping the child/parent link. 06_augmentation.py uses that link: it
matches on the child's text but builds the LLM's context from the child's
*parent* text instead, whenever a chunk has one.

Run:
    python scripts/02_chunking_strategies.py                    # writes output/02_chunks.json using DEFAULT_STRATEGY ('recursive')
    python scripts/02_chunking_strategies.py --strategy semantic # ...using 'semantic' instead
    python scripts/02_chunking_strategies.py --strategy parent_child
    python scripts/02_chunking_strategies.py --list-strategies   # print available strategy names and exit

Every strategy is always computed and written to its own
output/02_chunks_<strategy>.json regardless of --strategy, so you can
compare them freely (see the "Comparison across strategies" printout).
--strategy only controls which one gets copied to output/02_chunks.json,
the single file 03_embeddings.py (and everything downstream) actually
reads -- so switching strategies never requires touching any other
script, just re-running this one with a different flag.
"""

import argparse
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "..", "output")
CORPUS_PATH = os.path.join(OUTPUT_DIR, "01_extracted_corpus.json")

# Chunk size targets are in characters, not tokens, to avoid pulling in a
# tokenizer dependency here -- roughly 4 characters per token in English,
# so a 800-character chunk is a decent proxy for ~200 tokens.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")

# Parent-child sizing: parents are several times larger than the CHUNK_SIZE
# used by every other strategy (enough to hold a full section with
# surrounding context), children are smaller (tight enough that their
# embedding is about one specific idea, not a paragraph-spanning blend).
PARENT_CHUNK_SIZE = 2400
CHILD_CHUNK_SIZE = 300


def load_corpus() -> list[dict]:
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    return [doc for doc in corpus if doc.get("extraction_ok")]


def split_sentences(text: str) -> list[str]:
    """Naive but dependency-free sentence splitter. Good enough for the
    prose documents in this corpus; a production system would reach for
    a proper sentence tokenizer (e.g. spaCy or nltk.punkt) to correctly
    handle abbreviations like "Dr." or "e.g."."""
    text = text.replace("\n", " ")
    sentences = SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


def make_chunk(doc: dict, chunk_index: int, text: str, strategy: str) -> dict:
    return {
        "chunk_id": f"{doc['doc_id']}::{strategy}::{chunk_index}",
        "doc_id": doc["doc_id"],
        "filename": doc["filename"],
        "source_type": doc["source_type"],
        "strategy": strategy,
        "chunk_index": chunk_index,
        "text": text,
        "char_count": len(text),
    }


# ---------------------------------------------------------------------------
# Strategy 1: Fixed-size, no overlap
# ---------------------------------------------------------------------------

def chunk_fixed_size(doc: dict, size: int = CHUNK_SIZE) -> list[dict]:
    """Slice the text every `size` characters, ignoring word or sentence
    boundaries entirely. Simple and fast, but a chunk can end (or begin)
    mid-word or mid-sentence, and a fact split across a chunk boundary
    becomes hard to retrieve -- neither half contains the full idea."""
    text = doc["text"]
    chunks = []
    for i, start in enumerate(range(0, len(text), size)):
        piece = text[start:start + size].strip()
        if piece:
            chunks.append(make_chunk(doc, i, piece, "fixed_size"))
    return chunks


# ---------------------------------------------------------------------------
# Strategy 2: Fixed-size with sliding-window overlap
# ---------------------------------------------------------------------------

def chunk_fixed_size_overlap(doc: dict, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """Same idea as fixed-size, but each chunk repeats the last `overlap`
    characters of the previous one. This reduces (but does not eliminate)
    the "fact split across a boundary" problem: if a sentence straddles a
    boundary, there's a good chance the overlap window captures it whole
    in at least one chunk."""
    text = doc["text"]
    chunks = []
    step = size - overlap
    i = 0
    start = 0
    while start < len(text):
        piece = text[start:start + size].strip()
        if piece:
            chunks.append(make_chunk(doc, i, piece, "fixed_size_overlap"))
            i += 1
        start += step
    return chunks


# ---------------------------------------------------------------------------
# Strategy 3: Sentence-based chunking
# ---------------------------------------------------------------------------

def chunk_by_sentences(doc: dict, size: int = CHUNK_SIZE) -> list[dict]:
    """Accumulate whole sentences into a chunk until adding the next one
    would exceed `size`, then start a new chunk. Never splits a sentence
    in half, at the cost of variable chunk sizes and occasionally a very
    long single sentence that alone exceeds the target size."""
    sentences = split_sentences(doc["text"])
    chunks = []
    current: list[str] = []
    current_len = 0
    chunk_index = 0

    for sentence in sentences:
        sentence_len = len(sentence) + 1  # +1 for the joining space
        if current and current_len + sentence_len > size:
            chunks.append(make_chunk(doc, chunk_index, " ".join(current), "sentence"))
            chunk_index += 1
            current, current_len = [], 0
        current.append(sentence)
        current_len += sentence_len

    if current:
        chunks.append(make_chunk(doc, chunk_index, " ".join(current), "sentence"))

    return chunks


# ---------------------------------------------------------------------------
# Strategy 4: Recursive / paragraph-aware chunking
# ---------------------------------------------------------------------------

def _recursive_split(text: str, size: int, separators: list[str]) -> list[str]:
    """Try to split on the highest-priority separator first (paragraph
    breaks); if a resulting piece is still too big, recurse into the
    next separator (sentences, then words, then raw characters as a last
    resort). This is the strategy used by LangChain's
    RecursiveCharacterTextSplitter, reimplemented here from scratch to
    keep the dependency footprint at zero."""
    if len(text) <= size or not separators:
        return [text] if text.strip() else []

    sep, *rest_separators = separators
    if sep == "":
        # Base case: hard character split.
        return [text[i:i + size] for i in range(0, len(text), size)]

    parts = text.split(sep)
    pieces: list[str] = []
    current = ""
    for part in parts:
        candidate = (current + sep + part) if current else part
        if len(candidate) <= size:
            current = candidate
        else:
            if current:
                pieces.append(current)
            if len(part) > size:
                # This single paragraph/sentence is itself too long --
                # recurse into the next-finer separator.
                pieces.extend(_recursive_split(part, size, rest_separators))
                current = ""
            else:
                current = part
    if current:
        pieces.append(current)

    return [p.strip() for p in pieces if p.strip()]


def chunk_recursive(doc: dict, size: int = CHUNK_SIZE) -> list[dict]:
    """Paragraph boundaries first, then sentence boundaries, then raw
    characters as a last resort. This generally produces the most
    "readable" chunks: it respects document structure whenever the
    structure cooperates, and only falls back to a hard split for
    pathological cases (e.g. one giant unbroken paragraph)."""
    separators = ["\n\n", ". ", " ", ""]
    pieces = _recursive_split(doc["text"], size, separators)
    return [make_chunk(doc, i, piece, "recursive") for i, piece in enumerate(pieces)]


# ---------------------------------------------------------------------------
# Strategy 5: Similarity-based ("semantic") chunking
# ---------------------------------------------------------------------------

def _word_set(sentence: str) -> set:
    return set(re.findall(r"[a-z0-9]+", sentence.lower()))


def _jaccard_similarity(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def chunk_semantic(doc: dict, size: int = CHUNK_SIZE, similarity_threshold: float = 0.08) -> list[dict]:
    """Group consecutive sentences together as long as they're "on the
    same topic," and start a new chunk when the topic seems to shift.

    A production semantic chunker measures "on the same topic" with
    embedding cosine similarity between adjacent sentences (see
    03_embeddings.py) -- that's more accurate but means chunking depends
    on having already computed embeddings, and can be slow to run
    sentence-by-sentence over a large corpus. As a fast, dependency-free
    stand-in, this version uses Jaccard word-overlap between consecutive
    sentences as a cheap proxy for topical similarity: a sharp drop in
    shared vocabulary is a reasonable (if noisy) signal of a topic
    change. Swap `_jaccard_similarity` for real embedding similarity to
    upgrade this into a true semantic chunker.
    """
    sentences = split_sentences(doc["text"])
    if not sentences:
        return []

    chunks = []
    current: list[str] = [sentences[0]]
    current_len = len(sentences[0])
    chunk_index = 0
    prev_words = _word_set(sentences[0])

    for sentence in sentences[1:]:
        words = _word_set(sentence)
        similarity = _jaccard_similarity(prev_words, words)
        sentence_len = len(sentence) + 1

        topic_shift = similarity < similarity_threshold
        too_big = current_len + sentence_len > size

        if current and (topic_shift or too_big):
            chunks.append(make_chunk(doc, chunk_index, " ".join(current), "semantic"))
            chunk_index += 1
            current, current_len = [], 0

        current.append(sentence)
        current_len += sentence_len
        prev_words = words

    if current:
        chunks.append(make_chunk(doc, chunk_index, " ".join(current), "semantic"))

    return chunks


# ---------------------------------------------------------------------------
# Strategy 6: Parent-child ("small-to-big") chunking
# ---------------------------------------------------------------------------

def chunk_parent_child(
    doc: dict,
    parent_size: int = PARENT_CHUNK_SIZE,
    child_size: int = CHILD_CHUNK_SIZE,
) -> list[dict]:
    """Split each document twice. First into large "parent" chunks (using
    the same paragraph-aware recursive splitter as strategy #4), then split
    each parent into smaller "child" chunks the same way.

    Children are the retrievable unit: `chunk["text"]` is the child text,
    so 03_embeddings.py embeds and 04/05 search over *small, precise*
    pieces of text, exactly like the other strategies. The difference is
    that every child chunk also carries `parent_id` and `parent_text` --
    the full text of the parent it was cut from. 06_augmentation.py checks
    for `parent_text` and, when present, builds the LLM's context from
    that larger passage instead of the tiny child, so generation gets
    enough surrounding context to actually answer from, without diluting
    the embedding that found it in the first place.

    The tradeoff: children of the same parent share (duplicate) that
    parent's text once expanded, and the index stores each parent's text
    once per child rather than once total -- more disk/memory for a real
    deployment, though irrelevant at this corpus's scale.
    """
    separators = ["\n\n", ". ", " ", ""]
    parent_texts = _recursive_split(doc["text"], parent_size, separators)

    chunks = []
    chunk_index = 0
    for parent_index, parent_text in enumerate(parent_texts):
        parent_id = f"{doc['doc_id']}::parent::{parent_index}"
        child_texts = _recursive_split(parent_text, child_size, separators)
        for child_text in child_texts:
            chunk = make_chunk(doc, chunk_index, child_text, "parent_child")
            chunk["parent_id"] = parent_id
            chunk["parent_index"] = parent_index
            chunk["parent_text"] = parent_text
            chunk["parent_char_count"] = len(parent_text)
            chunks.append(chunk)
            chunk_index += 1

    return chunks


STRATEGIES = {
    "fixed_size": chunk_fixed_size,
    "fixed_size_overlap": chunk_fixed_size_overlap,
    "sentence": chunk_by_sentences,
    "recursive": chunk_recursive,
    "semantic": chunk_semantic,
    "parent_child": chunk_parent_child,
}

# The strategy the rest of the pipeline (03_embeddings.py onward) uses by
# default. Recursive chunking with paragraph-awareness is a strong
# general-purpose default: it respects document structure without
# requiring embeddings to already exist (unlike semantic chunking).
DEFAULT_STRATEGY = "recursive"


def summarize(chunks: list[dict], label: str) -> None:
    if not chunks:
        print(f"  {label:<22} 0 chunks")
        return
    sizes = [c["char_count"] for c in chunks]
    print(
        f"  {label:<22} {len(chunks):>4} chunks   "
        f"avg={sum(sizes) / len(sizes):>6.0f} chars   "
        f"min={min(sizes):>5}   max={max(sizes):>5}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Chunk the extracted corpus and pick which strategy the rest of the pipeline uses."
    )
    parser.add_argument(
        "--strategy", "-s",
        choices=sorted(STRATEGIES.keys()),
        default=DEFAULT_STRATEGY,
        help=f"Which strategy to copy to output/02_chunks.json (default: '{DEFAULT_STRATEGY}').",
    )
    parser.add_argument(
        "--list-strategies", action="store_true",
        help="Print available strategy names and exit, without chunking anything.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.list_strategies:
        print("Available strategies:")
        for name in sorted(STRATEGIES.keys()):
            marker = "  (default)" if name == DEFAULT_STRATEGY else ""
            print(f"  {name}{marker}")
        return

    corpus = load_corpus()
    print(f"Loaded {len(corpus)} extracted documents from {CORPUS_PATH}\n")
    print(f"Target chunk size: {CHUNK_SIZE} chars (overlap for strategy #2: {CHUNK_OVERLAP} chars)\n")

    results: dict[str, list[dict]] = {}
    for strategy_name, strategy_fn in STRATEGIES.items():
        chunks = []
        for doc in corpus:
            chunks.extend(strategy_fn(doc))
        results[strategy_name] = chunks

        out_path = os.path.join(OUTPUT_DIR, f"02_chunks_{strategy_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

    print("Comparison across strategies (all documents combined):")
    for strategy_name, chunks in results.items():
        summarize(chunks, strategy_name)

    print(f"\nPer-strategy chunk files written to {OUTPUT_DIR}/02_chunks_<strategy>.json")

    # Also write the *chosen* strategy's chunk set under a stable filename
    # so downstream scripts (03_embeddings.py onward) don't need to know
    # which strategy was picked -- they just read 02_chunks.json. Which
    # strategy that is comes from --strategy (default: DEFAULT_STRATEGY),
    # not a hardcoded name, so switching strategies is just a flag away.
    chosen_strategy = args.strategy
    default_path = os.path.join(OUTPUT_DIR, "02_chunks.json")
    with open(default_path, "w", encoding="utf-8") as f:
        json.dump(results[chosen_strategy], f, indent=2, ensure_ascii=False)
    print(
        f"\nActive strategy for the rest of the pipeline: '{chosen_strategy}' "
        f"({len(results[chosen_strategy])} chunks) -> {default_path}"
    )
    if chosen_strategy != DEFAULT_STRATEGY:
        print(f"(overriding the default '{DEFAULT_STRATEGY}' via --strategy)")
    print(
        "Re-run 03_embeddings.py and 04_vector_database.py next to rebuild "
        "the index from this chunk set."
    )

    # Show one example chunk from each strategy for the same document so
    # the qualitative difference is easy to see side by side.
    example_doc_id = corpus[0]["doc_id"]
    print(f"\nExample first chunk per strategy, for document '{example_doc_id}':")
    for strategy_name, chunks in results.items():
        first = next((c for c in chunks if c["doc_id"] == example_doc_id), None)
        if first:
            preview = first["text"][:160].replace("\n", " ")
            print(f"\n  [{strategy_name}] ({first['char_count']} chars)\n    {preview}...")


if __name__ == "__main__":
    main()
