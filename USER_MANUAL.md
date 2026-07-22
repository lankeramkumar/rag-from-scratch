# Sunrise Medical Center RAG Pipeline — User Manual

**Ram Kumar Lanke**
Applied AI Solutions Architect
lankeramkumar@gmail.com

---

This project is a hand-built, dependency-light Retrieval-Augmented Generation
(RAG) pipeline that runs end-to-end over 36 realistic hospital documents —
PDFs, Word docs, plain-text notes, diagrams/forms (as images), and
audio/video training recordings. Every stage of RAG is its own numbered
script in `scripts/`, so you can run the whole thing, inspect any
intermediate output, or swap out one stage (e.g. the chunking strategy)
without touching the others.

```
Sunrise_Medical_Center_Documents/   36 source documents (11 department subfolders)
scripts/                           01-09, run in order
output/                            everything each script produces, in one place
.env                                ANTHROPIC_API_KEY / VOYAGE_API_KEY
```

---

## 1. Prerequisites (one-time, already done on this machine)

| Requirement | Purpose | Status |
|---|---|---|
| Python packages (`reportlab`, `python-docx`, `pillow`, `pypdf`, `pytesseract`, `faster-whisper`, `scikit-learn`, `numpy`, `anthropic`, `python-dotenv`, ...) | Document generation + every pipeline stage | Installed |
| Tesseract OCR (`winget install UB-Mannheim.TesseractOCR`) | Extracts text from the 7 diagram/form images | Installed |
| `faster-whisper` + `tiny.en` model | Transcribes the 3 audio + 2 video files | Installed |
| `.env` with `ANTHROPIC_API_KEY` and `VOYAGE_API_KEY` | Real Claude generation + Voyage embeddings | Present in project root |

Nothing here needs to be reinstalled to use the pipeline — this table is just
so you know why extraction "just works" without extra setup.

---

## 2. Quick Start

From the project root, build the index once:

```
python scripts/01_text_extraction.py
python scripts/02_chunking_strategies.py
python scripts/03_embeddings.py
python scripts/04_vector_database.py
```

Then ask questions, as many times as you like, without re-running the above:

```
python scripts/07_generation.py "What is Sunrise Medical Center's protocol for a suspected stroke patient?"
```

You only need to redo the setup steps when the **source documents** or the
**chunking strategy** change (see §7, "When to re-run what").

---

## 3. The pipeline, stage by stage

| # | Script | What it does | Reads | Writes |
|---|---|---|---|---|
| 01 | `01_text_extraction.py` | Extracts plain text from every file — parses PDF/Word/TXT directly, OCRs images (Tesseract), transcribes audio/video (faster-whisper) | `Sunrise_Medical_Center_Documents/` (recursive) | `output/01_extracted_corpus.json` |
| 02 | `02_chunking_strategies.py` | Splits each document into retrievable chunks; computes **6 strategies** for comparison, writes the chosen one as the active set | `01_extracted_corpus.json` | `output/02_chunks_<strategy>.json` (all 6) + `output/02_chunks.json` (active) |
| 03 | `03_embeddings.py` | Embeds every active chunk into a vector (Voyage API → sentence-transformers → TF-IDF, whichever is available) | `02_chunks.json` | `03_embeddings.npy`, `03_embeddings_meta.json`, `03_embedder.pkl` |
| 04 | `04_vector_database.py` | Builds and persists a numpy-based vector index (add/search/save/load) | `03_embeddings.*` | `04_vector_index.npy`, `04_vector_index_meta.pkl` |
| 05 | `05_retrieval.py` | Embeds a question, searches the index — both pure vector search and hybrid (vector + keyword) search | `04_vector_index.*` | `output/05_retrieval_results.json` (+ console output) |
| 06 | `06_augmentation.py` | Dedupes retrieved chunks, fits them into a character budget, builds the cited prompt sent to the LLM | (calls 05 internally) | `output/06_augmented_prompts.json` |
| 07 | `07_generation.py` | Sends the augmented prompt to Claude (`claude-opus-4-8`) and prints a cited answer; falls back to a mock answer if no API key | (calls 06 internally) | `output/07_generated_answers.json` |
| 08 | `08_visualize_embeddings.py` | Projects all chunk embeddings to 2D and plots them (static PNGs + interactive HTML) | `03_embeddings.*` | `output/08_embedding_space_*.png/html` |
| 09 | `09_visualize_search.py` | Same 2D space, plus a dropdown of demo queries with their nearest neighbors drawn in | `03_embeddings.*`, `03_embedder.pkl` | `output/09_vector_search_interactive.html` |

**07 is the command you'll use most** — it silently runs 05 and 06 for you,
so `python scripts/07_generation.py "..."` is a complete question-answering
call in one line.

---

## 4. Worked examples

### Example 1 — First-time setup (run once, or after adding/editing documents)

```
python scripts/01_text_extraction.py
python scripts/02_chunking_strategies.py
python scripts/03_embeddings.py
python scripts/04_vector_database.py
```

**Expected result:** `01` prints one line per file (36 total) ending in `OK`;
`02` prints a size comparison across 6 strategies and confirms which one is
now active; `03` reports the embedding backend used (falls back gracefully if
Voyage isn't reachable — see §6); `04` confirms the index was built and
reloads it as a sanity check.

---

### Example 2 — A simple, single-document question

```
python scripts/07_generation.py "What extension do I call for the IT Helpdesk, and is it available 24/7?"
```

**Expected result:**

```
ANSWER:
The IT Helpdesk phone extension is 6000, available 24/7 for urgent/system-down issues [3].

SOURCES:
  [1] IT_Helpdesk_Ticket_Submission_Notes.txt (chunk 6)
  [2] Password_and_Access_Control_Policy.docx (chunk 5)
  [3] IT_Helpdesk_Ticket_Submission_Notes.txt (chunk 1)
  [4] IT_Helpdesk_Ticket_Submission_Notes.txt (chunk 2)
```

A narrow, factual question mostly resolves to one document, with a
secondary policy doc pulled in for extra context.

---

### Example 3 — A question that spans multiple documents

```
python scripts/07_generation.py "What are Sunrise Medical Center's rules around accepting gifts from vendors, and how does that relate to conflicts of interest and compliance reporting?"
```

**Expected result:** Claude synthesizes a single answer citing **3 different
source documents across 2 formats**:

```
SOURCES:
  [1] Code_of_Conduct_and_Ethics_Policy.docx (chunk 1)
  [2] Employee_Handbook_2026.docx (chunk 0)
  [3] Code_of_Conduct_and_Ethics_Policy.docx (chunk 2)
  [4] New_Employee_Orientation_Welcome_Message.mp3 (chunk 0)
```

---

### Example 4 — A question that spans **all six file formats**

By default only 4 chunks are retrieved (`DEFAULT_TOP_K = 4`), which usually
isn't enough room for every format at once. Use `--top-k` to pull back more:

```
python scripts/05_retrieval.py --top-k 10 "What fire safety and stroke recognition training does Sunrise Medical Center provide to staff, including training videos, audio announcements, written guides, and diagrams?"
```

**Expected result:** one hit from every format, ranked:

```
 1  score=0.263  Employee_Handbook_2026.docx           (.docx)
 2  score=0.256  New_Employee_Orientation_Welcome_Message.mp3  (.mp3, transcribed)
 3  score=0.247  Fire_Evacuation_Quick_Guide.txt        (.txt)
 4  score=0.211  Staff_ID_Badge_Sample.png              (.png, OCR'd)
 5  score=0.195  Sepsis_Recognition_and_Treatment_Protocol.pdf  (.pdf)
10  score=0.174  Stroke_Recognition_FAST_Training_Video.mp4  (.mp4, transcribed)
```

**Note:** if you run this same question through `07_generation.py --top-k 10`
instead, Claude's final answer may only cite 4 of those 10 — `06_augmentation.py`
enforces a 3000-character context budget (`MAX_CONTEXT_CHARS`) and drops the
lowest-ranked chunks first. Retrieval breadth and what actually reaches the
model are two different things; this is deliberate, not a bug.

---

### Example 5 — Retrieval only, no API call

If you just want to see what would be retrieved, without spending a Claude
API call on generation:

```
python scripts/05_retrieval.py "What should a nurse do if a medication refrigerator's temperature goes out of range?"
```

**Expected result:** both a `vector` search block and a `hybrid` search block
printed side by side, each showing rank, score, filename, chunk index, and a
text preview — useful for comparing the two search modes directly.

---

### Example 6 — Switching the chunking strategy

```
python scripts/02_chunking_strategies.py --list-strategies
```
```
Available strategies:
  fixed_size
  fixed_size_overlap
  parent_child
  recursive  (default)
  semantic
  sentence
```

Switch to parent-child chunking (small chunks for matching, larger parent
chunks for context — see the docstring at the top of
`02_chunking_strategies.py`) and rebuild the index:

```
python scripts/02_chunking_strategies.py --strategy parent_child
python scripts/03_embeddings.py
python scripts/04_vector_database.py
python scripts/06_augmentation.py "What should a nurse do if a medication refrigerator's temperature goes out of range?"
```

**Expected result:** `06_augmentation.py` reports
`retrieved=4  after_dedup=3  used=1` — 4 small child chunks were matched, one
was a duplicate sibling (same parent as another), and expanding to full
parent text alone (~2,366 characters) filled the whole context budget with
one richly-detailed passage instead of four disconnected fragments.

Switch back any time:
```
python scripts/02_chunking_strategies.py --strategy recursive
python scripts/03_embeddings.py
python scripts/04_vector_database.py
```

---

### Example 7 — Running with no API key (mock mode)

If `ANTHROPIC_API_KEY` isn't set, the pipeline still runs end-to-end — it
just can't generate a real answer:

```
No ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN found in the environment.
Running in MOCK generation mode -- see the module docstring for setup instructions to enable real Claude generation.

ANSWER:
[MOCK ANSWER -- no ANTHROPIC_API_KEY configured, so this is not a real LLM generation] Based on the top retrieved passage from IT_Helpdesk_Ticket_Submission_Notes.txt [1], see the CONTEXT block above for the relevant details. Set ANTHROPIC_API_KEY and re-run for a real synthesized, cited answer.
```

Useful for demonstrating that retrieval/augmentation work independently of
having an LLM available at all — generation is the only stage that needs one.

---

### Example 8 — Demo mode (no question argument)

Every script from 05 onward runs 4 built-in example questions if you don't
pass one:

```
python scripts/07_generation.py
```

**Expected result:** answers for all 4 demo questions in turn (stroke
protocol, PTO policy, fridge temperature excursion, fire RACE procedure),
each with its own citations — a quick way to sanity-check the whole pipeline
after any change.

---

### Example 9 — Visualizing the vector space

```
python scripts/08_visualize_embeddings.py
python scripts/09_visualize_search.py
```

**Expected result:** both write self-contained HTML files to `output/` —
`08_embedding_space_interactive.html` (hover/click any chunk to see which
document it's from and its actual text) and
`09_vector_search_interactive.html` (a dropdown of demo queries, plotted as a
star with dashed lines to its top-4 nearest neighbors). Double-click either
file to open it in a browser — no server or download needed.

---

## 5. Command-line flags reference

| Flag | Available in | Meaning |
|---|---|---|
| `"your question"` | `05`, `06`, `07` | Ask this question instead of running the 4 demo queries |
| `--top-k N` / `-k N` | `05`, `06`, `07` | Retrieve N chunks instead of the default 4 (put this *before* the question) |
| `--strategy NAME` / `-s NAME` | `02` | Make `NAME` the active chunking strategy (`02_chunks.json`) |
| `--list-strategies` | `02` | Print available strategy names and exit |

Example combining flags: `python scripts/07_generation.py --top-k 8 "your question here"`

---

## 6. Troubleshooting

| Symptom | Cause | What happens |
|---|---|---|
| `'voyage' embedding failed (RateLimitError...)` | Voyage free tier has no payment method on file (3 requests/min cap) | `03_embeddings.py` automatically falls back to `TfidfEmbedder` — no action needed, just slightly less "semantic" search quality |
| Image files show `char_count: 0` / `EMPTY/FAILED` in `01`'s output | Tesseract OCR not installed or not found | Install via `winget install UB-Mannheim.TesseractOCR`, or set `pytesseract.pytesseract.tesseract_cmd` manually |
| Audio/video files show `[Transcription skipped...]` | `faster-whisper` not installed | `pip install faster-whisper` (downloads the `tiny.en` model on first use, ~75MB) |
| `MODE: mock (no API key)` in `07`'s output | `ANTHROPIC_API_KEY` not set | Add it to `.env` at the project root, or set it as a real environment variable |
| A retrieved format doesn't show up even with `--top-k` bumped up | Query too narrow/specific for that format's (short, sparse) OCR/transcript text | Broaden the query wording, or raise `--top-k` further — see Example 4 |

---

## 7. When to re-run what

| You changed... | Re-run starting from |
|---|---|
| Added/edited/removed a source document | `01` → `02` → `03` → `04` |
| Just want to switch chunking strategy | `02 --strategy ...` → `03` → `04` |
| Nothing — just asking a new question | `05` / `06` / `07` only |
| Want to see the embedding space after any of the above | `08` / `09` |

---

## 8. Output files reference

All outputs live in `output/` and are safe to delete — any script regenerates
its own outputs (and only its own) each time it runs.

| File | Produced by |
|---|---|
| `01_extracted_corpus.json` | `01` |
| `02_chunks_<strategy>.json` (×6), `02_chunks.json` (active) | `02` |
| `03_embeddings.npy`, `03_embeddings_meta.json`, `03_embedder.pkl` | `03` |
| `04_vector_index.npy`, `04_vector_index_meta.pkl` | `04` |
| `05_retrieval_results.json` | `05` |
| `06_augmented_prompts.json` | `06` |
| `07_generated_answers.json` | `07` |
| `08_embedding_space_all.png`, `08_embedding_space_no_code.png`, `08_embedding_space_interactive.html` | `08` |
| `09_vector_search_interactive.html` | `09` |
