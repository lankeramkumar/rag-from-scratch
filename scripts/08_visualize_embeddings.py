"""
08_visualize_embeddings.py

Author: Ram Kumar Lanke
Applied AI Solutions Architect
lankeramkumar@gmail.com

BONUS: VISUALIZING EMBEDDINGS
================================
The embedding matrix from 03_embeddings.py is 99 chunks x 4000 numbers --
too many dimensions to look at directly. This script projects each
chunk's vector down to 2 dimensions (via TruncatedSVD, the sparse-matrix
equivalent of PCA) purely so it can be plotted, and saves the result in
two forms, both written directly to output/ -- nothing to export or
download, they're already local files:

  1. Static PNGs (matplotlib) -- open in any image viewer.
  2. A single self-contained HTML file (Chart.js via CDN) -- open in any
     browser for the interactive version: hover a dot for its filename,
     click a dot to read the actual chunk text it represents.

Two "views" are produced in each form:
  1. All 99 chunks -- the 4 source-code files (.py/.js/.sql/.sh) sit far
     from every prose document, because TF-IDF vocabulary for code
     barely overlaps with prose vocabulary at all.
  2. The same data with code removed and zoomed in -- at this scale you
     can see chunks from the same document sitting closer to each other
     than chunks from different documents, which is the property
     retrieval (05_retrieval.py) actually relies on.

Run:
    python scripts/08_visualize_embeddings.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")  # write PNGs directly, no display/GUI backend needed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import TruncatedSVD

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "..", "output")

EMBEDDINGS_PATH = os.path.join(OUTPUT_DIR, "03_embeddings.npy")
META_PATH = os.path.join(OUTPUT_DIR, "03_embeddings_meta.json")
HTML_PATH = os.path.join(OUTPUT_DIR, "08_embedding_space_interactive.html")

PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
]

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Embedding vector space -- interactive</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }}
  h1 {{ font-size: 20px; font-weight: 600; }}
  p.sub {{ color: #555; font-size: 14px; }}
  h2 {{ font-size: 15px; font-weight: 600; margin: 2rem 0 6px; }}
  #detail {{ margin-top: 16px; padding: 1rem 1.25rem; background: #f4f4f2; border-radius: 10px; font-size: 13px; color: #444; }}
  #detail b {{ color: #111; }}
  #legend {{ display: flex; flex-wrap: wrap; gap: 6px 14px; margin-top: 12px; font-size: 11px; color: #555; }}
  #legend span.item {{ display: flex; align-items: center; gap: 4px; }}
  #legend span.swatch {{ width: 9px; height: 9px; border-radius: 2px; display: inline-block; }}
</style>
</head>
<body>
<h1>Embedding vector space (TF-IDF, projected to 2D)</h1>
<p class="sub">Each dot is one chunk's 4000-dim vector, projected to 2D. Hover for the filename, click a dot to read the chunk text behind it.</p>
<h2>All {n_all} chunks</h2>
<div style="position:relative;width:100%;height:340px"><canvas id="chartAll"></canvas></div>
<h2>Zoomed in: everything except the code files</h2>
<div style="position:relative;width:100%;height:380px"><canvas id="chartNoCode"></canvas></div>
<div id="detail">Click any dot above to see the chunk text it represents.</div>
<div id="legend"></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
const points = {points_json};
const palette = {palette_json};
const docs = [...new Set(points.map(p => p.doc))];
const colorOf = {{}};
docs.forEach((d, i) => colorOf[d] = palette[i % palette.length]);

function makeDatasets(list) {{
  return docs.filter(d => list.some(p => p.doc === d)).map(d => ({{
    label: d,
    data: list.filter(p => p.doc === d),
    backgroundColor: colorOf[d],
    pointRadius: 5,
    pointHoverRadius: 8
  }}));
}}

const detailEl = document.getElementById('detail');
function showDetail(raw) {{
  detailEl.innerHTML =
    '<b>' + raw.file + '</b> &middot; chunk ' + raw.idx + '<br><br>' + raw.text + '&hellip;';
}}

const tooltipCb = {{ callbacks: {{ label: (ctx) => ctx.raw.file + ' (chunk ' + ctx.raw.idx + ')' }} }};
function onPointClick(evt, elements, chart) {{
  if (!elements.length) return;
  const el = elements[0];
  const raw = chart.data.datasets[el.datasetIndex].data[el.index];
  showDetail(raw);
}}

new Chart(document.getElementById('chartAll'), {{
  type: 'scatter',
  data: {{ datasets: makeDatasets(points) }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    onClick: onPointClick,
    plugins: {{ legend: {{ display: false }}, tooltip: tooltipCb }},
    scales: {{
      x: {{ title: {{ display: true, text: 'dimension 1 (arbitrary units)' }} }},
      y: {{ title: {{ display: true, text: 'dimension 2 (arbitrary units)' }} }}
    }}
  }}
}});

const nonCode = points.filter(p => p.type !== 'code');
new Chart(document.getElementById('chartNoCode'), {{
  type: 'scatter',
  data: {{ datasets: makeDatasets(nonCode) }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    onClick: onPointClick,
    plugins: {{ legend: {{ display: false }}, tooltip: tooltipCb }},
    scales: {{
      x: {{ title: {{ display: true, text: 'dimension 1 (arbitrary units)' }} }},
      y: {{ title: {{ display: true, text: 'dimension 2 (arbitrary units)' }} }}
    }}
  }}
}});

const legend = document.getElementById('legend');
docs.forEach(d => {{
  const item = document.createElement('span');
  item.className = 'item';
  const swatch = document.createElement('span');
  swatch.className = 'swatch';
  swatch.style.background = colorOf[d];
  item.appendChild(swatch);
  item.appendChild(document.createTextNode(d.replace(/^doc\\d+_/, '')));
  legend.appendChild(item);
}});
</script>
</body>
</html>
"""


def write_interactive_html(coords: np.ndarray, chunks: list[dict], out_path: str) -> None:
    points = []
    for (x, y), c in zip(coords, chunks):
        points.append({
            "x": round(float(x), 5), "y": round(float(y), 5),
            "doc": c["doc_id"], "file": c["filename"], "type": c["source_type"],
            "idx": c["chunk_index"], "text": c["text"][:220].replace("\n", " "),
        })

    html = HTML_TEMPLATE.format(
        n_all=len(points),
        points_json=json.dumps(points, separators=(",", ":")),
        palette_json=json.dumps(PALETTE),
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  wrote {out_path}")


def plot_scatter(coords: np.ndarray, chunks: list[dict], title: str, out_path: str) -> None:
    doc_ids = sorted(set(c["doc_id"] for c in chunks))
    color_of = {doc_id: PALETTE[i % len(PALETTE)] for i, doc_id in enumerate(doc_ids)}

    fig, ax = plt.subplots(figsize=(9, 7))
    for doc_id in doc_ids:
        idxs = [i for i, c in enumerate(chunks) if c["doc_id"] == doc_id]
        ax.scatter(
            coords[idxs, 0], coords[idxs, 1],
            label=doc_id.split("_", 1)[1] if "_" in doc_id else doc_id,
            color=color_of[doc_id], s=70, edgecolors="white", linewidths=0.5,
        )

    ax.set_title(title)
    ax.set_xlabel("dimension 1 (arbitrary units)")
    ax.set_ylabel("dimension 2 (arbitrary units)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    vectors = np.load(EMBEDDINGS_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)["chunks"]

    print(f"Loaded {vectors.shape[0]} vectors of dimension {vectors.shape[1]}\n")

    svd = TruncatedSVD(n_components=2, random_state=42)
    coords = svd.fit_transform(vectors)
    print(f"Projected to 2D (explained variance ratio: {svd.explained_variance_ratio_.sum():.1%})\n")

    all_path = os.path.join(OUTPUT_DIR, "08_embedding_space_all.png")
    plot_scatter(coords, chunks, "All 99 chunks (code sits far from prose)", all_path)

    non_code_idxs = [i for i, c in enumerate(chunks) if c["source_type"] != "code"]
    non_code_chunks = [chunks[i] for i in non_code_idxs]
    non_code_coords = coords[non_code_idxs]
    zoom_path = os.path.join(OUTPUT_DIR, "08_embedding_space_no_code.png")
    plot_scatter(non_code_coords, non_code_chunks, "Zoomed in: same-document chunks cluster together", zoom_path)

    write_interactive_html(coords, chunks, HTML_PATH)

    print(f"\nStatic images (open in any image viewer):")
    print(f"  {all_path}")
    print(f"  {zoom_path}")
    print(f"\nInteractive version (double-click to open in your browser -- hover + click-to-inspect):")
    print(f"  {HTML_PATH}")


if __name__ == "__main__":
    main()
