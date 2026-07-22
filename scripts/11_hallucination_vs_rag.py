"""
11_hallucination_vs_rag.py

Author: Ram Kumar Lanke
Applied AI Solutions Architect
lankeramkumar@gmail.com

BONUS: WITHOUT RAG VS. WITH RAG
==================================
This is the argument the whole pipeline exists to make, run for real: ask
Claude the same question about Sunrise Medical Center's private internal
policies two different ways.

  1. UNGROUNDED -- the question, alone, with no document context. This is
     exactly what asking ChatGPT (or Claude) directly looks like: the model
     has never seen this hospital's actual internal documents (they aren't
     on the public internet), so anything specific it states -- an
     extension number, a dollar threshold, a document ID, a named policy
     owner -- is either a guess, a hallucination, or a hedge. It cannot be
     the hospital's actual policy, because the model was never shown it.

  2. GROUNDED -- the exact same question run through the real pipeline:
     05_retrieval.py finds the actual policy chunk, 06_augmentation.py
     builds a cited prompt from it, 07_generation.py answers strictly from
     that context.

Both calls hit the real Claude API -- nothing here is scripted or faked.
Whatever the ungrounded model actually says is what gets recorded and
shown, hedge or hallucination alike; the point survives either way, since
neither can know the hospital's actual internal specifics.

Requires ANTHROPIC_API_KEY (both sides call Claude; there's no meaningful
"mock" version of this comparison).

Run:
    python scripts/11_hallucination_vs_rag.py
"""

import html as html_lib
import json
import os
import re
import sys

from dotenv import load_dotenv

# Claude's answers can include characters (e.g. "<=") outside the default
# Windows console codepage (cp1252); reconfigure stdout so printing them
# doesn't crash the run -- the JSON/HTML output isn't affected either way.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "..", "output")
PROJECT_ROOT = os.path.join(HERE, "..")

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

sys.path.insert(0, HERE)
import importlib

_augmentation_module = importlib.import_module("06_augmentation")
_retrieval_module = importlib.import_module("05_retrieval")
_generation_module = importlib.import_module("07_generation")

RESULTS_PATH = os.path.join(OUTPUT_DIR, "11_hallucination_vs_rag.json")
HTML_PATH = os.path.join(OUTPUT_DIR, "11_hallucination_vs_rag_interactive.html")

MODEL = _generation_module.MODEL
MAX_TOKENS = _generation_module.MAX_TOKENS

# Deliberately no mention of Sunrise Medical Center, or that any documents
# exist -- this is meant to be exactly what a plain "ask ChatGPT" prompt
# looks like, not a system prompt engineered to provoke a worse answer.
UNGROUNDED_SYSTEM_PROMPT = (
    "You are a helpful, knowledgeable assistant. Answer the user's question "
    "as best you can."
)

# Chosen because each has a concrete, specific answer (an extension number,
# a dollar threshold and document ID, a named escalation step) that only
# exists inside Sunrise Medical Center's private documents -- there is no
# way for a model to know these by chance, so the contrast holds whether
# the ungrounded answer turns out to be a hedge or a confident guess.
QUESTIONS = [
    "What is Sunrise Medical Center's protocol for treating a suspected stroke patient in the emergency room?",
    "What extension do I call for the IT Helpdesk at Sunrise Medical Center, and is it available 24/7?",
    "What is the maximum dollar value of a vendor gift an employee can accept at Sunrise Medical Center before it must be reported, and what is the policy document ID?",
    "What should a nurse do if a medication refrigerator's temperature goes out of range at Sunrise Medical Center?",
]


def generate_ungrounded(question: str) -> str:
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=UNGROUNDED_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": question}],
    )
    return next(block.text for block in response.content if block.type == "text")


def run_comparison(question: str, embedder, store) -> dict:
    print(f"Question: {question}")

    print("  calling Claude with no document context (ungrounded)...")
    ungrounded_answer = generate_ungrounded(question)

    print("  running the full RAG pipeline (retrieval -> augmentation -> generation)...")
    augmented = _augmentation_module.augment_query(question, embedder, store, top_k=4)
    grounded_result = _generation_module.run_generation(augmented)

    return {
        "question": question,
        "ungrounded_answer": ungrounded_answer,
        "grounded_answer": grounded_result["answer"],
        "grounded_mode": grounded_result["mode"],
        "sources": grounded_result["sources"],
    }


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_HEADING_RE = re.compile(r"^#{1,4}\s+(.+)$", re.MULTILINE)


def render_answer_html(text: str) -> str:
    """Minimal, ad hoc rendering of a Claude answer: escape HTML, turn
    "## Heading"-style markdown lines into a small heading, **bold** into
    <strong>, and blank-line-separated paragraphs into <p> tags. Not a
    real markdown parser -- just enough for the plain prose (with
    occasional headings/bold terms/lists) style Claude answers in,
    ungrounded answers especially, which lean on markdown structure more
    than the terser, citation-focused grounded answers do."""
    escaped = html_lib.escape(text)
    escaped = _HEADING_RE.sub(r'<span class="mini-heading">\1</span>', escaped)
    escaped = _BOLD_RE.sub(r"<strong>\1</strong>", escaped)
    paragraphs = [p for p in escaped.split("\n\n") if p.strip()]
    return "".join(f"<p>{p.replace(chr(10), '<br>')}</p>" for p in paragraphs)


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Without RAG vs. With RAG -- a live comparison</title>
<style>
  :root {{
    --bg-top: #0a0e16; --bg-bottom: #101520;
    --accent: #38bdf8; --accent2: #a78bfa;
    --bad: #f87171; --good: #34d399;
    --white: #f0f4f8; --gray: #94a3b8; --dim: #334155; --node: #161c28;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; min-height: 100vh; padding: 48px 24px 64px;
    background: linear-gradient(180deg, var(--bg-top), var(--bg-bottom));
    color: var(--white);
    font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  }}
  main {{ max-width: 1040px; margin: 0 auto; }}
  .eyebrow {{
    font-family: Consolas, "SF Mono", monospace; color: var(--accent);
    letter-spacing: 0.1em; font-size: 12px; margin-bottom: 14px;
  }}
  h1 {{ font-size: 34px; margin: 0 0 10px; }}
  h1 span {{ color: var(--accent); }}
  p.lede {{ color: var(--gray); font-size: 15px; max-width: 780px; line-height: 1.6; margin: 0 0 28px; }}
  .picker {{ margin-bottom: 26px; }}
  .picker label {{
    font-family: Consolas, "SF Mono", monospace; color: var(--accent2);
    font-size: 12px; letter-spacing: 0.08em; display: block; margin-bottom: 8px;
  }}
  select {{
    width: 100%; max-width: 720px; padding: 12px 14px; border-radius: 8px;
    background: var(--node); color: var(--white); border: 1px solid var(--dim);
    font-size: 15px;
  }}
  .question-line {{
    font-size: 18px; margin: 0 0 22px; padding-bottom: 18px;
    border-bottom: 1px solid var(--dim); line-height: 1.5;
  }}
  .question-line b {{ color: var(--accent2); }}
  .columns {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  @media (max-width: 800px) {{ .columns {{ grid-template-columns: 1fr; }} }}
  .col {{
    border-radius: 14px; padding: 20px 22px; background: var(--node);
    border: 1.5px solid var(--dim);
  }}
  .col.bad {{ border-color: rgba(248, 113, 113, 0.45); }}
  .col.good {{ border-color: rgba(52, 211, 153, 0.45); }}
  .col-label {{
    display: inline-flex; align-items: center; gap: 8px;
    font-family: Consolas, "SF Mono", monospace; font-size: 12px;
    letter-spacing: 0.06em; padding: 5px 12px; border-radius: 999px;
    margin-bottom: 14px;
  }}
  .col.bad .col-label {{ color: var(--bad); background: rgba(248, 113, 113, 0.12); }}
  .col.good .col-label {{ color: var(--good); background: rgba(52, 211, 153, 0.12); }}
  .col h3 {{ margin: 0 0 4px; font-size: 16px; }}
  .col .sub {{ color: var(--gray); font-size: 13px; margin: 0 0 16px; }}
  .answer {{ font-size: 14.5px; line-height: 1.65; color: #e2e8f0; }}
  .answer p {{ margin: 0 0 12px; }}
  .answer strong {{ color: var(--white); }}
  .answer .mini-heading {{
    display: block; font-weight: 700; color: var(--accent2);
    font-size: 13px; letter-spacing: 0.03em; margin-top: 4px;
  }}
  .sources {{
    margin-top: 16px; padding-top: 14px; border-top: 1px solid var(--dim);
    font-family: Consolas, "SF Mono", monospace; font-size: 12.5px; color: var(--good);
  }}
  .sources div {{ margin-bottom: 4px; }}
  footer {{
    margin-top: 40px; padding-top: 20px; border-top: 1px solid var(--dim);
    display: flex; justify-content: space-between; flex-wrap: wrap; gap: 10px; font-size: 14px;
  }}
  footer .name {{ font-weight: 700; }}
  footer .role {{ color: var(--gray); margin-left: 10px; }}
  footer a {{ color: var(--accent); text-decoration: none; font-family: Consolas, "SF Mono", monospace; font-size: 13px; }}
</style>
</head>
<body>
<main>
  <div class="eyebrow">RAG FROM SCRATCH &nbsp;&middot;&nbsp; BONUS VISUALIZATION</div>
  <h1>Without RAG vs. <span>With RAG</span></h1>
  <p class="lede">
    The same question, asked two different ways, both against the real Claude API --
    nothing on this page is scripted. On the left, the model answers alone, exactly like
    asking ChatGPT directly: it has never seen Sunrise Medical Center's private internal
    documents, so any specific detail it states cannot be the hospital's actual policy.
    On the right, the same question runs through this repo's real retrieval &rarr;
    augmentation &rarr; generation pipeline, answering strictly from the retrieved
    document and citing its source.
  </p>

  <div class="picker">
    <label>CHOOSE A QUESTION</label>
    <select id="q-select"></select>
  </div>

  <div class="question-line" id="question-line"></div>

  <div class="columns">
    <div class="col bad">
      <div class="col-label">&#9888; WITHOUT RAG</div>
      <h3>Claude, no document access</h3>
      <p class="sub">Same model, same question, zero context -- exactly what asking ChatGPT directly looks like.</p>
      <div class="answer" id="ungrounded-answer"></div>
    </div>
    <div class="col good">
      <div class="col-label">&#10003; WITH RAG</div>
      <h3>Claude, grounded in the real document</h3>
      <p class="sub">Retrieved the actual policy chunk, then answered strictly from it.</p>
      <div class="answer" id="grounded-answer"></div>
      <div class="sources" id="grounded-sources"></div>
    </div>
  </div>

  <footer>
    <div><span class="name">Ram Kumar Lanke</span><span class="role">Applied AI Solutions Architect</span></div>
    <a href="mailto:lankeramkumar@gmail.com">lankeramkumar@gmail.com</a>
  </footer>
</main>

<script>
const DATA = {data_json};

const select = document.getElementById("q-select");
const questionLine = document.getElementById("question-line");
const ungroundedEl = document.getElementById("ungrounded-answer");
const groundedEl = document.getElementById("grounded-answer");
const sourcesEl = document.getElementById("grounded-sources");

DATA.forEach((item, i) => {{
  const opt = document.createElement("option");
  opt.value = i;
  opt.textContent = item.question;
  select.appendChild(opt);
}});

function render(i) {{
  const item = DATA[i];
  questionLine.innerHTML = "<b>Q:</b> " + item.question;
  ungroundedEl.innerHTML = item.ungrounded_html;
  groundedEl.innerHTML = item.grounded_html;
  sourcesEl.innerHTML = item.sources.map(
    s => "<div>[" + s.ref + "] " + s.filename + " (chunk " + s.chunk_index + ")</div>"
  ).join("");
}}

select.addEventListener("change", () => render(select.value));
render(0);
</script>
</body>
</html>
"""


def build_html(results: list[dict]) -> None:
    payload = []
    for r in results:
        payload.append({
            "question": r["question"],
            "ungrounded_html": render_answer_html(r["ungrounded_answer"]),
            "grounded_html": render_answer_html(r["grounded_answer"]),
            "sources": r["sources"],
        })
    html_out = HTML_TEMPLATE.format(data_json=json.dumps(payload, ensure_ascii=False))
    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html_out)


def main():
    if not _generation_module.is_api_key_configured():
        print(
            "ANTHROPIC_API_KEY is required for this comparison -- both sides call "
            "Claude, so there isn't a meaningful mock version of it. See "
            "07_generation.py's docstring for setup instructions."
        )
        return

    print(f"ANTHROPIC_API_KEY detected. Comparing with model '{MODEL}'.\n")

    embedder = _retrieval_module.load_embedder()
    store = _retrieval_module.load_store()

    results = []
    for question in QUESTIONS:
        result = run_comparison(question, embedder, store)
        results.append(result)

        print("=" * 78)
        print(f"QUESTION: {result['question']}")
        print(f"\n--- WITHOUT RAG ---\n{result['ungrounded_answer']}")
        print(f"\n--- WITH RAG ({result['grounded_mode']}) ---\n{result['grounded_answer']}")
        print("\nSOURCES:")
        for s in result["sources"]:
            print(f"  [{s['ref']}] {s['filename']} (chunk {s['chunk_index']})")
        print()

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(results)} comparison(s) to {RESULTS_PATH}")

    build_html(results)
    print(f"Wrote {HTML_PATH}")
    print("Open it directly in your browser -- no server or download needed.")


if __name__ == "__main__":
    main()
