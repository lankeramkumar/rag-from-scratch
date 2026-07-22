"""
10_visualize_chunking.py

Author: Ram Kumar Lanke
Applied AI Solutions Architect
lankeramkumar@gmail.com

BONUS: VISUALIZING CHUNKING STRATEGIES
=========================================
02_chunking_strategies.py prints size statistics for all 6 chunking
strategies, but a table of averages doesn't show you the thing that
actually matters: *where* each strategy decides to cut the same piece of
text. A recursive splitter that respects paragraph breaks and a fixed-size
splitter that doesn't look like identical numbers in a table, but produce
completely different (and differently useful) chunks in practice.

This script renders the same handful of real documents six times each --
once per strategy -- with chunk boundaries highlighted directly on the
original text, so the difference is visible rather than inferred from
statistics. For the parent-child strategy specifically, it renders a
second, nested view: small child-chunk highlights sitting inside their
larger parent-chunk boundary, which is the whole mechanic
06_augmentation.py relies on (match on the child, answer from the parent).

Output is a single self-contained HTML file: a document picker, and six
stacked panels (one per strategy) for whichever document is selected, each
annotated with its chunk count and average chunk size for that document.

Run:
    python scripts/10_visualize_chunking.py
"""

import html
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "..", "output")

CORPUS_PATH = os.path.join(OUTPUT_DIR, "01_extracted_corpus.json")
HTML_PATH = os.path.join(OUTPUT_DIR, "10_chunking_strategies_interactive.html")

STRATEGIES = ["fixed_size", "fixed_size_overlap", "sentence", "recursive", "semantic", "parent_child"]
STRATEGY_LABELS = {
    "fixed_size": "1. Fixed-size",
    "fixed_size_overlap": "2. Fixed-size + overlap",
    "sentence": "3. Sentence-based",
    "recursive": "4. Recursive (paragraph-aware)",
    "semantic": "5. Semantic (similarity-based)",
    "parent_child": "6. Parent-child (small-to-big)",
}

# A handful of real documents, spanning formats and structure (prose,
# numbered procedure, ASCII headers, a pipe-delimited table), so the
# strategy differences show up in more than one kind of text.
EXAMPLE_DOC_IDS = [
    "Code_of_Conduct_and_Ethics_Policy",
    "Employee_Handbook_2026",
    "Stroke_Protocol_Emergency_Department",
    "Fire_Evacuation_Quick_Guide",
    "Pharmacy_Refrigerator_Temperature_Log_Instructions",
    "MRI_Scanner_Operations_Manual",
]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


_WS_RUN_RE = re.compile(r"\s+")


def _collapse_whitespace(text: str) -> tuple[str, list[int]]:
    """Collapse every run of whitespace (including newlines) to a single
    space, returning the collapsed text plus a list mapping each
    character index in the collapsed text back to its index in the
    original. Needed because split_sentences() (used by the "sentence"
    and "semantic" strategies) replaces newlines with spaces and then
    re-joins sentences with a single space regardless of how much
    whitespace originally separated them -- so a chunk built that way is
    no longer a byte-exact substring of the source document, even though
    it's clearly "the same text" to a human. Matching in whitespace-
    collapsed space sidesteps that (and, as a side benefit, tolerates the
    same kind of drift in ASCII-art/table-formatted documents where
    repeated spacing carries no real meaning)."""
    chars = []
    index_map = []
    i, n = 0, len(text)
    while i < n:
        if text[i].isspace():
            index_map.append(i)
            chars.append(" ")
            while i < n and text[i].isspace():
                i += 1
        else:
            index_map.append(i)
            chars.append(text[i])
            i += 1
    return "".join(chars), index_map


def find_chunk_spans(doc_text: str, chunk_texts: list[str]) -> list[tuple[int, int]]:
    """Locate each chunk's (start, end) character offset in the original
    document text, in order. Matching happens in whitespace-collapsed
    space (see _collapse_whitespace) and offsets are mapped back to the
    original text, so differences in exact whitespace between a chunk and
    the source document don't cause a match to be missed.

    Every strategy in 02_chunking_strategies.py produces chunks whose
    start position is monotonically non-decreasing through the document
    (even the overlapping strategy: each chunk starts later than the
    previous one, just not later than the previous one *ends*) -- so
    searching forward from the previous chunk's own start, rather than
    its end, correctly handles overlap without needing to know each
    strategy's internal arithmetic here."""
    collapsed_doc, index_map = _collapse_whitespace(doc_text)
    spans = []
    cursor = 0
    for text in chunk_texts:
        collapsed_chunk, _ = _collapse_whitespace(text.strip())
        pos = collapsed_doc.find(collapsed_chunk, cursor)
        if pos == -1 or not collapsed_chunk:
            # Shouldn't happen (every chunk is "the same text" as some
            # span of the original by construction) -- skip defensively
            # rather than crash the whole visualization over one document.
            spans.append(None)
            continue
        end_pos = pos + len(collapsed_chunk)
        start = index_map[pos]
        end = index_map[end_pos - 1] + 1
        spans.append((start, end))
        cursor = pos
    return spans


CHUNK_COLORS = [
    "rgba(56, 189, 248, 0.22)",   # cyan
    "rgba(167, 139, 250, 0.22)",  # violet
]
PARENT_BORDER = "rgba(251, 191, 36, 0.65)"   # amber
CHILD_FILL = "rgba(56, 189, 248, 0.28)"      # cyan


def render_flat_highlight(doc_text: str, spans: list) -> str:
    """Render doc_text as HTML with each chunk span wrapped in a <mark>,
    alternating two background colors so adjacent chunks are visually
    distinguishable even when they touch with no gap between them."""
    parts = []
    cursor = 0
    color_i = 0
    for i, span in enumerate(spans):
        if span is None:
            continue
        start, end = span
        start = max(start, cursor)
        if start >= end:
            continue
        if start > cursor:
            parts.append(html.escape(doc_text[cursor:start]))
        color = CHUNK_COLORS[color_i % len(CHUNK_COLORS)]
        color_i += 1
        chunk_text = doc_text[start:end]
        parts.append(
            f'<mark class="chunk" style="background:{color}" '
            f'data-info="Chunk {i} &middot; {len(chunk_text)} chars">'
            f"{html.escape(chunk_text)}</mark>"
        )
        cursor = end
    if cursor < len(doc_text):
        parts.append(html.escape(doc_text[cursor:]))
    return "".join(parts)


def render_parent_child_highlight(doc_text: str, parent_spans: dict, child_spans: list) -> str:
    """Render parent chunks as an outlined block, with each parent's own
    child chunks highlighted (filled) inside it -- the nested small-to-big
    view. `parent_spans` maps parent_index -> (start, end); `child_spans`
    is a list of (parent_index, child_start, child_end) in document order."""
    ordered_parents = sorted(parent_spans.items(), key=lambda kv: kv[1][0])
    children_by_parent: dict[int, list[tuple[int, int]]] = {}
    for parent_index, c_start, c_end in child_spans:
        children_by_parent.setdefault(parent_index, []).append((c_start, c_end))

    parts = []
    cursor = 0
    for parent_index, (p_start, p_end) in ordered_parents:
        p_start = max(p_start, cursor)
        if p_start > cursor:
            parts.append(html.escape(doc_text[cursor:p_start]))
        if p_start >= p_end:
            continue

        parts.append(
            f'<span class="parent-block" '
            f'data-info="Parent {parent_index} &middot; {p_end - p_start} chars">'
        )
        inner_cursor = p_start
        for c_start, c_end in sorted(children_by_parent.get(parent_index, [])):
            c_start = max(c_start, inner_cursor)
            if c_start >= c_end or c_start >= p_end:
                continue
            if c_start > inner_cursor:
                parts.append(html.escape(doc_text[inner_cursor:c_start]))
            c_end = min(c_end, p_end)
            parts.append(
                f'<mark class="chunk child" style="background:{CHILD_FILL}" '
                f'data-info="Child chunk &middot; {c_end - c_start} chars">'
                f"{html.escape(doc_text[c_start:c_end])}</mark>"
            )
            inner_cursor = c_end
        if inner_cursor < p_end:
            parts.append(html.escape(doc_text[inner_cursor:p_end]))
        parts.append("</span>")
        cursor = p_end

    if cursor < len(doc_text):
        parts.append(html.escape(doc_text[cursor:]))
    return "".join(parts)


def build_document_payload(doc_id: str, doc_text: str, chunks_by_strategy: dict) -> dict:
    strategies_out = {}
    for strategy in STRATEGIES:
        all_chunks = [c for c in chunks_by_strategy[strategy] if c["doc_id"] == doc_id]
        all_chunks.sort(key=lambda c: c["chunk_index"])
        if not all_chunks:
            strategies_out[strategy] = {"html": "", "count": 0, "avg": 0, "min": 0, "max": 0}
            continue

        sizes = [c["char_count"] for c in all_chunks]
        stats = {
            "count": len(all_chunks),
            "avg": round(sum(sizes) / len(sizes)),
            "min": min(sizes),
            "max": max(sizes),
        }

        if strategy == "parent_child":
            parent_texts = {}
            for c in all_chunks:
                parent_texts.setdefault(c["parent_id"], c["parent_text"])
            parent_id_to_index = {pid: i for i, pid in enumerate(parent_texts)}
            parent_spans_flat = find_chunk_spans(doc_text, list(parent_texts.values()))
            parent_spans = {
                parent_id_to_index[pid]: span
                for pid, span in zip(parent_texts.keys(), parent_spans_flat)
                if span is not None
            }

            child_spans = []
            for c in all_chunks:
                p_idx = parent_id_to_index[c["parent_id"]]
                p_span = parent_spans.get(p_idx)
                if p_span is None:
                    continue
                p_start, p_end = p_span
                # Search for the child within its own parent's slice of the
                # document, so identical short child snippets in different
                # parents don't get matched to the wrong parent.
                local_pos = doc_text.find(c["text"], p_start, p_end + 1)
                if local_pos == -1:
                    continue
                child_spans.append((p_idx, local_pos, local_pos + len(c["text"])))

            rendered = render_parent_child_highlight(doc_text, parent_spans, child_spans)
            stats["parent_count"] = len(parent_texts)
        else:
            chunk_texts = [c["text"] for c in all_chunks]
            spans = find_chunk_spans(doc_text, chunk_texts)
            rendered = render_flat_highlight(doc_text, spans)

        strategies_out[strategy] = {"html": rendered, **stats}

    return strategies_out


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Chunking strategies -- interactive comparison</title>
<style>
  :root {{
    --bg-top: #0a0e16; --bg-bottom: #101520;
    --accent: #38bdf8; --accent2: #a78bfa;
    --white: #f0f4f8; --gray: #94a3b8; --dim: #334155; --node: #161c28;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; min-height: 100vh; padding: 48px 24px 64px;
    background: linear-gradient(180deg, var(--bg-top), var(--bg-bottom));
    color: var(--white);
    font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  }}
  main {{ max-width: 980px; margin: 0 auto; }}
  .eyebrow {{
    font-family: Consolas, "SF Mono", monospace; color: var(--accent);
    letter-spacing: 0.1em; font-size: 12px; margin-bottom: 14px;
  }}
  h1 {{ font-size: 34px; margin: 0 0 10px; }}
  h1 span {{ color: var(--accent); }}
  p.lede {{ color: var(--gray); font-size: 15px; max-width: 760px; line-height: 1.6; margin: 0 0 28px; }}
  .picker {{ margin-bottom: 30px; }}
  .picker label {{
    font-family: Consolas, "SF Mono", monospace; color: var(--accent2);
    font-size: 12px; letter-spacing: 0.08em; display: block; margin-bottom: 8px;
  }}
  select {{
    width: 100%; max-width: 560px; padding: 12px 14px; border-radius: 8px;
    background: var(--node); color: var(--white); border: 1px solid var(--dim);
    font-size: 15px;
  }}
  .panel {{
    background: var(--node); border: 1px solid var(--dim); border-radius: 14px;
    padding: 20px 22px; margin-bottom: 18px;
  }}
  .panel-head {{
    display: flex; justify-content: space-between; align-items: baseline;
    flex-wrap: wrap; gap: 8px; margin-bottom: 12px;
  }}
  .panel-head h3 {{ margin: 0; font-size: 16px; color: var(--white); }}
  .stats {{ font-family: Consolas, "SF Mono", monospace; font-size: 12px; color: var(--gray); }}
  .stats b {{ color: var(--accent); }}
  .doc-text {{
    font-family: Consolas, "SF Mono", monospace; font-size: 13px; line-height: 1.7;
    white-space: pre-wrap; word-break: break-word;
    max-height: 260px; overflow-y: auto; padding: 4px 2px;
    color: #cbd5e1;
  }}
  mark.chunk {{ border-radius: 3px; padding: 0 1px; color: inherit; }}
  span.parent-block {{
    display: inline; border: 1.5px dashed {parent_border}; border-radius: 6px;
    padding: 2px 3px; margin: 0 1px;
  }}
  #info-bar {{
    position: sticky; top: 0; z-index: 5; background: rgba(10,14,22,0.92);
    border: 1px solid var(--dim); border-radius: 8px; padding: 10px 16px;
    font-family: Consolas, "SF Mono", monospace; font-size: 13px; color: var(--accent);
    margin-bottom: 20px; backdrop-filter: blur(6px);
  }}
  footer {{
    margin-top: 40px; padding-top: 20px; border-top: 1px solid var(--dim);
    display: flex; justify-content: space-between; flex-wrap: wrap; gap: 10px; font-size: 14px;
  }}
  footer .name {{ font-weight: 700; }}
  footer .role {{ color: var(--gray); margin-left: 10px; }}
  footer a {{ color: var(--accent); text-decoration: none; font-family: Consolas, "SF Mono", monospace; font-size: 13px; }}
  a.back {{ color: var(--accent); text-decoration: none; font-size: 13px; }}
</style>
</head>
<body>
<main>
  <div class="eyebrow">RAG FROM SCRATCH &nbsp;&middot;&nbsp; BONUS VISUALIZATION</div>
  <h1>Chunking Strategies, <span>Side by Side</span></h1>
  <p class="lede">
    The same document, split six different ways. Chunk boundaries are highlighted
    directly on the original text so you can see -- not just measure -- how each
    strategy decides where to cut. Strategy 6 (parent-child) is rendered as nested
    boxes: the dashed amber outline is the larger parent chunk used for context,
    the filled cyan highlights inside it are the small child chunks actually
    matched against a query.
  </p>

  <div class="picker">
    <label>CHOOSE A DOCUMENT</label>
    <select id="doc-select"></select>
  </div>

  <div id="info-bar">Hover any highlighted chunk to see its size.</div>
  <div id="panels"></div>

  <footer>
    <div><span class="name">Ram Kumar Lanke</span><span class="role">Applied AI Solutions Architect</span></div>
    <a href="mailto:lankeramkumar@gmail.com">lankeramkumar@gmail.com</a>
  </footer>
</main>

<script>
const DATA = {data_json};
const LABELS = {labels_json};
const STRATEGIES = {strategies_json};

const select = document.getElementById("doc-select");
const panelsEl = document.getElementById("panels");
const infoBar = document.getElementById("info-bar");

for (const docId of Object.keys(DATA)) {{
  const opt = document.createElement("option");
  opt.value = docId;
  opt.textContent = DATA[docId].filename + "  (" + DATA[docId].source_type + ")";
  select.appendChild(opt);
}}

function renderDoc(docId) {{
  const doc = DATA[docId];
  panelsEl.innerHTML = "";
  for (const strategy of STRATEGIES) {{
    const s = doc.strategies[strategy];
    const panel = document.createElement("div");
    panel.className = "panel";

    const head = document.createElement("div");
    head.className = "panel-head";
    const h3 = document.createElement("h3");
    h3.textContent = LABELS[strategy];
    const stats = document.createElement("div");
    stats.className = "stats";
    let statsText = "<b>" + s.count + "</b> chunks &middot; avg <b>" + s.avg + "</b> chars &middot; range " + s.min + "-" + s.max;
    if (strategy === "parent_child") {{
      statsText += " &middot; <b>" + s.parent_count + "</b> parents";
    }}
    stats.innerHTML = statsText;
    head.appendChild(h3);
    head.appendChild(stats);

    const body = document.createElement("div");
    body.className = "doc-text";
    body.innerHTML = s.html;

    panel.appendChild(head);
    panel.appendChild(body);
    panelsEl.appendChild(panel);
  }}

  panelsEl.querySelectorAll("mark.chunk, span.parent-block").forEach((el) => {{
    el.addEventListener("mouseenter", () => {{
      infoBar.textContent = el.getAttribute("data-info").replace("&middot;", "\\u00b7");
    }});
  }});
}}

select.addEventListener("change", () => renderDoc(select.value));
renderDoc(Object.keys(DATA)[0]);
</script>
</body>
</html>
"""


def main():
    corpus = load_json(CORPUS_PATH)
    corpus_by_id = {d["doc_id"]: d for d in corpus}

    chunks_by_strategy = {}
    for strategy in STRATEGIES:
        path = os.path.join(OUTPUT_DIR, f"02_chunks_{strategy}.json")
        chunks_by_strategy[strategy] = load_json(path)

    payload = {}
    for doc_id in EXAMPLE_DOC_IDS:
        doc = corpus_by_id.get(doc_id)
        if not doc:
            print(f"  skipping '{doc_id}': not found in extracted corpus")
            continue
        print(f"  building panels for {doc_id} ({doc['source_type']}, {doc['char_count']} chars)")
        strategies_out = build_document_payload(doc_id, doc["text"], chunks_by_strategy)
        payload[doc_id] = {
            "filename": doc["filename"],
            "source_type": doc["source_type"],
            "strategies": strategies_out,
        }

    html_out = HTML_TEMPLATE.format(
        data_json=json.dumps(payload, ensure_ascii=False),
        labels_json=json.dumps(STRATEGY_LABELS),
        strategies_json=json.dumps(STRATEGIES),
        parent_border=PARENT_BORDER,
    )
    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html_out)

    print(f"\nWrote {HTML_PATH}")
    print("Open it directly in your browser -- no server or download needed.")


if __name__ == "__main__":
    main()
