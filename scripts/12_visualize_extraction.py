"""
12_visualize_extraction.py

Author: Ram Kumar Lanke
Applied AI Solutions Architect
lankeramkumar@gmail.com

BONUS: VISUALIZING TEXT EXTRACTION
=====================================
01_text_extraction.py's whole job is collapsing six different file formats
into one shape: {doc_id, text, metadata}. That's easy to state and easy to
miss the weight of, because a JSON file full of extracted strings doesn't
*look* like a scanned form, an audio recording, or a training video anymore
-- it just looks like text.

This script puts the original source back next to what was extracted from
it: the actual image next to its OCR'd text, an actual `<audio>`/`<video>`
player next to its transcript, an embedded PDF next to its parsed text --
one example per format, so the transformation is visible rather than
assumed. Word documents can't be rendered by a browser at all, which is its
own small illustration of why extraction has to happen in the first place.

Output is a single self-contained HTML file that also references the
original source files in Sunrise_Medical_Center_Documents/ (relative
paths), so it must be opened from within the project (or from GitHub
Pages, where those files are published alongside it) for the previews to
load -- the extracted text itself is embedded directly and always works.

Run:
    python scripts/12_visualize_extraction.py
"""

import html
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "..", "output")
DOCS_DIR_NAME = "Sunrise_Medical_Center_Documents"

CORPUS_PATH = os.path.join(OUTPUT_DIR, "01_extracted_corpus.json")
HTML_PATH = os.path.join(OUTPUT_DIR, "12_extraction_interactive.html")

# One example per format actually produced by 01_text_extraction.py's
# extractor registry -- chosen from documents already featured elsewhere
# in this project's demos, so a viewer who's seen those recognizes them.
EXAMPLE_DOC_IDS = [
    "Stroke_Protocol_Emergency_Department",           # pdf
    "Code_of_Conduct_and_Ethics_Policy",               # word (docx)
    "Fire_Evacuation_Quick_Guide",                      # text
    "Staff_ID_Badge_Sample",                            # image_ocr (png)
    "New_Employee_Orientation_Welcome_Message",         # audio_transcript (mp3)
    "Stroke_Recognition_FAST_Training_Video",           # video_transcript (mp4)
]

SOURCE_TYPE_LABELS = {
    "pdf": "PDF",
    "word": "Word (.docx)",
    "text": "Plain text (.txt)",
    "image_ocr": "Image -- OCR (Tesseract)",
    "audio_transcript": "Audio -- transcribed (faster-whisper)",
    "video_transcript": "Video -- transcribed (faster-whisper)",
}

EXTRACTOR_NOTES = {
    "pdf": "Parsed directly with pypdf -- page by page, no OCR involved.",
    "word": "Parsed directly with python-docx -- paragraphs and table cells, in order. "
            "Browsers can't render .docx natively, which is exactly why this step exists: "
            "extraction turns a format nothing else in the pipeline understands into "
            "plain text every later stage can.",
    "text": "Already plain text -- read directly, no parsing needed.",
    "image_ocr": "No text layer exists in a PNG -- Tesseract OCR reads the pixels themselves.",
    "audio_transcript": "No text exists at all until this step -- faster-whisper (Whisper) "
                         "transcribes the spoken narration from the raw audio.",
    "video_transcript": "Same as audio -- faster-whisper decodes the audio track straight out "
                         "of the video container and transcribes it, no separate extraction step.",
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_preview_html(doc: dict, rel_path: str) -> str:
    """The left-panel "original source" preview, format-appropriate: an
    embedded PDF viewer, an actual image, or a real playable audio/video
    element -- not a description of one."""
    source_type = doc["source_type"]
    escaped_path = html.escape(rel_path)
    alt = html.escape(doc["filename"])

    if source_type == "pdf":
        return f'<iframe src="{escaped_path}" class="pdf-frame" title="{alt}"></iframe>'
    if source_type == "image_ocr":
        return f'<img src="{escaped_path}" alt="{alt}" class="image-preview">'
    if source_type == "audio_transcript":
        return f'<audio controls preload="none" src="{escaped_path}" class="audio-preview"></audio>'
    if source_type == "video_transcript":
        return f'<video controls preload="none" src="{escaped_path}" class="video-preview"></video>'
    if source_type == "word":
        return (
            '<div class="unrenderable">'
            '<div class="unrenderable-icon">&#128196;</div>'
            f'<div class="unrenderable-label">{alt}</div>'
            '<div class="unrenderable-note">Browsers cannot render .docx directly.</div>'
            f'<a class="unrenderable-link" href="{escaped_path}" download>Download the source file</a>'
            "</div>"
        )
    # "text": the original *is* the extracted text -- no separate preview.
    return (
        '<div class="unrenderable">'
        '<div class="unrenderable-icon">&#128221;</div>'
        f'<div class="unrenderable-label">{alt}</div>'
        '<div class="unrenderable-note">Plain text -- identical to the extracted result on the right.</div>'
        "</div>"
    )


def build_payload(corpus: list[dict]) -> tuple[list[dict], dict]:
    corpus_by_id = {d["doc_id"]: d for d in corpus}

    examples = []
    for doc_id in EXAMPLE_DOC_IDS:
        doc = corpus_by_id[doc_id]
        rel_path = f"../{DOCS_DIR_NAME}/{doc['folder']}/{doc['filename']}"
        examples.append({
            "doc_id": doc_id,
            "filename": doc["filename"],
            "source_type": doc["source_type"],
            "source_label": SOURCE_TYPE_LABELS[doc["source_type"]],
            "note": EXTRACTOR_NOTES[doc["source_type"]],
            "char_count": doc["char_count"],
            "extraction_ok": doc["extraction_ok"],
            "preview_html": build_preview_html(doc, rel_path),
            "text": doc["text"],
        })

    # Corpus-wide stats: how many of each format, and how many extracted
    # cleanly, across all 36 documents -- not just the 6 shown above.
    counts_by_type: dict[str, int] = {}
    ok_by_type: dict[str, int] = {}
    for d in corpus:
        st = d["source_type"]
        counts_by_type[st] = counts_by_type.get(st, 0) + 1
        if d["extraction_ok"]:
            ok_by_type[st] = ok_by_type.get(st, 0) + 1

    stats = {
        "total": len(corpus),
        "total_ok": sum(1 for d in corpus if d["extraction_ok"]),
        "by_type": [
            {
                "source_type": st,
                "label": SOURCE_TYPE_LABELS.get(st, st),
                "count": counts_by_type[st],
                "ok": ok_by_type.get(st, 0),
            }
            for st in sorted(counts_by_type, key=lambda k: -counts_by_type[k])
        ],
    }
    return examples, stats


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Text extraction -- original source vs. extracted text</title>
<style>
  :root {{
    --bg-top: #0a0e16; --bg-bottom: #101520;
    --accent: #38bdf8; --accent2: #a78bfa;
    --good: #34d399;
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
  p.lede {{ color: var(--gray); font-size: 15px; max-width: 780px; line-height: 1.6; margin: 0 0 22px; }}

  .stats-strip {{
    display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 30px;
  }}
  .stat-chip {{
    font-family: Consolas, "SF Mono", monospace; font-size: 12.5px;
    padding: 8px 14px; border-radius: 999px; border: 1px solid var(--dim);
    background: var(--node); color: var(--gray);
  }}
  .stat-chip b {{ color: var(--good); }}
  .stat-chip.total {{ border-color: rgba(56,189,248,0.5); color: var(--accent); }}
  .stat-chip.total b {{ color: var(--accent); }}

  .picker {{ margin-bottom: 22px; }}
  .picker label {{
    font-family: Consolas, "SF Mono", monospace; color: var(--accent2);
    font-size: 12px; letter-spacing: 0.08em; display: block; margin-bottom: 8px;
  }}
  select {{
    width: 100%; max-width: 560px; padding: 12px 14px; border-radius: 8px;
    background: var(--node); color: var(--white); border: 1px solid var(--dim);
    font-size: 15px;
  }}

  .note-line {{
    color: var(--gray); font-size: 14px; margin: 0 0 20px; line-height: 1.6;
    padding: 12px 16px; border-left: 3px solid var(--accent2); background: rgba(167,139,250,0.06);
    border-radius: 0 8px 8px 0;
  }}

  .columns {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  @media (max-width: 800px) {{ .columns {{ grid-template-columns: 1fr; }} }}
  .col {{ border-radius: 14px; padding: 20px 22px; background: var(--node); border: 1.5px solid var(--dim); }}
  .col-label {{
    display: inline-flex; align-items: center; gap: 8px;
    font-family: Consolas, "SF Mono", monospace; font-size: 12px;
    letter-spacing: 0.06em; padding: 5px 12px; border-radius: 999px;
    margin-bottom: 14px; color: var(--gray); background: rgba(148,163,184,0.1);
  }}
  .col.right .col-label {{ color: var(--good); background: rgba(52,211,153,0.12); }}
  .col h3 {{ margin: 0 0 4px; font-size: 16px; }}
  .col .sub {{ color: var(--gray); font-size: 13px; margin: 0 0 16px; }}

  .pdf-frame {{ width: 100%; height: 360px; border: none; border-radius: 8px; background: #fff; }}
  .image-preview {{ width: 100%; height: auto; border-radius: 8px; display: block; }}
  .audio-preview {{ width: 100%; margin-top: 40px; }}
  .video-preview {{ width: 100%; border-radius: 8px; display: block; }}
  .unrenderable {{
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    text-align: center; padding: 48px 16px; border: 1.5px dashed var(--dim); border-radius: 10px;
  }}
  .unrenderable-icon {{ font-size: 40px; margin-bottom: 10px; }}
  .unrenderable-label {{ font-family: Consolas, "SF Mono", monospace; font-size: 13px; color: var(--white); margin-bottom: 6px; }}
  .unrenderable-note {{ color: var(--gray); font-size: 13px; margin-bottom: 14px; }}
  .unrenderable-link {{ color: var(--accent); text-decoration: none; font-size: 13px; font-family: Consolas, "SF Mono", monospace; }}

  .doc-text {{
    font-family: Consolas, "SF Mono", monospace; font-size: 13px; line-height: 1.7;
    white-space: pre-wrap; word-break: break-word;
    max-height: 360px; overflow-y: auto; padding: 4px 2px; color: #cbd5e1;
  }}
  .meta-row {{
    display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 14px;
    font-family: Consolas, "SF Mono", monospace; font-size: 12px; color: var(--gray);
  }}
  .meta-row b {{ color: var(--good); }}

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
  <h1>Text Extraction, <span>Before and After</span></h1>
  <p class="lede">
    Every RAG pipeline starts by collapsing every input format into the same
    plain-text-plus-metadata shape. That's easy to describe and easy to take for
    granted -- so here's the actual source file next to what
    <code>01_text_extraction.py</code> pulled out of it: a real image next to its
    OCR'd text, a playable audio/video recording next to its transcript, an
    embedded PDF next to its parsed text.
  </p>

  <div class="stats-strip" id="stats-strip"></div>

  <div class="picker">
    <label>CHOOSE A DOCUMENT</label>
    <select id="doc-select"></select>
  </div>

  <div class="note-line" id="note-line"></div>

  <div class="columns">
    <div class="col left">
      <div class="col-label">ORIGINAL SOURCE</div>
      <h3 id="left-filename"></h3>
      <p class="sub" id="left-type"></p>
      <div id="preview-slot"></div>
    </div>
    <div class="col right">
      <div class="col-label">EXTRACTED TEXT</div>
      <h3>What 01_text_extraction.py produced</h3>
      <div class="meta-row" id="meta-row"></div>
      <div class="doc-text" id="extracted-text"></div>
    </div>
  </div>

  <footer>
    <div><span class="name">Ram Kumar Lanke</span><span class="role">Applied AI Solutions Architect</span></div>
    <a href="mailto:lankeramkumar@gmail.com">lankeramkumar@gmail.com</a>
  </footer>
</main>

<script>
const EXAMPLES = {examples_json};
const STATS = {stats_json};

const statsStrip = document.getElementById("stats-strip");
const totalChip = document.createElement("div");
totalChip.className = "stat-chip total";
totalChip.innerHTML = "<b>" + STATS.total_ok + "/" + STATS.total + "</b> documents extracted successfully";
statsStrip.appendChild(totalChip);
STATS.by_type.forEach((s) => {{
  const chip = document.createElement("div");
  chip.className = "stat-chip";
  chip.innerHTML = "<b>" + s.ok + "/" + s.count + "</b> " + s.label;
  statsStrip.appendChild(chip);
}});

const select = document.getElementById("doc-select");
EXAMPLES.forEach((ex, i) => {{
  const opt = document.createElement("option");
  opt.value = i;
  opt.textContent = ex.filename + "  (" + ex.source_label + ")";
  select.appendChild(opt);
}});

const noteLine = document.getElementById("note-line");
const leftFilename = document.getElementById("left-filename");
const leftType = document.getElementById("left-type");
const previewSlot = document.getElementById("preview-slot");
const metaRow = document.getElementById("meta-row");
const extractedText = document.getElementById("extracted-text");

function render(i) {{
  const ex = EXAMPLES[i];
  noteLine.textContent = ex.note;
  leftFilename.textContent = ex.filename;
  leftType.textContent = ex.source_label;
  previewSlot.innerHTML = ex.preview_html;
  metaRow.innerHTML =
    "<span><b>" + ex.char_count + "</b> chars</span>" +
    "<span>status: <b>" + (ex.extraction_ok ? "OK" : "FAILED") + "</b></span>";
  extractedText.textContent = ex.text;
}}

select.addEventListener("change", () => render(select.value));
render(0);
</script>
</body>
</html>
"""


def main():
    corpus = load_json(CORPUS_PATH)
    examples, stats = build_payload(corpus)

    for ex in examples:
        print(f"  {ex['filename']:<48} {ex['source_label']:<32} {ex['char_count']:>6} chars")

    html_out = HTML_TEMPLATE.format(
        examples_json=json.dumps(examples, ensure_ascii=False),
        stats_json=json.dumps(stats),
    )
    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html_out)

    print(f"\nWrote {HTML_PATH}")
    print(
        f"Corpus-wide: {stats['total_ok']}/{stats['total']} documents extracted "
        f"successfully across {len(stats['by_type'])} formats."
    )
    print("Open it directly in your browser -- no server or download needed.")


if __name__ == "__main__":
    main()
