"""
09_visualize_search.py

Author: Ram Kumar Lanke
Applied AI Solutions Architect
lankeramkumar@gmail.com

BONUS: VISUALIZING A VECTOR DATABASE SEARCH
==============================================
08_visualize_embeddings.py shows where the 99 chunk embeddings sit
relative to each other -- but it never actually performs a search. This
script closes that gap: it takes a handful of demo queries, embeds each
one with the exact same embedder used to build the index
(03_embeddings.py / embedding_backends.py), projects it into the same 2D
space as the corpus (via the same fitted TruncatedSVD), and finds its
real top-k nearest neighbors in the full-dimensional space -- i.e. it
runs the same computation 04_vector_database.py's VectorStore.search()
performs, and plots the result.

Output is a single self-contained HTML file with a dropdown to switch
between demo queries. For the selected query you'll see:
  - the query itself, plotted as a star in the same vector space
  - dashed lines to its top-4 nearest neighbors
  - a results panel listing those neighbors with their similarity score
    and actual chunk text -- the same information 05_retrieval.py prints

Run:
    python scripts/09_visualize_search.py
"""

import json
import os
import pickle
import sys

import numpy as np
from sklearn.decomposition import TruncatedSVD

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "..", "output")

EMBEDDINGS_PATH = os.path.join(OUTPUT_DIR, "03_embeddings.npy")
META_PATH = os.path.join(OUTPUT_DIR, "03_embeddings_meta.json")
EMBEDDER_PATH = os.path.join(OUTPUT_DIR, "03_embedder.pkl")
HTML_PATH = os.path.join(OUTPUT_DIR, "09_vector_search_interactive.html")

sys.path.insert(0, HERE)  # required so pickle.load can resolve embedding_backends classes

TOP_K = 4
DEMO_QUERIES = [
    "What is Sunrise Medical Center's protocol for treating a suspected stroke patient in the emergency room?",
    "How do I request PTO and what happens to unused time off when I leave the hospital?",
    "What should a nurse do if a medication refrigerator's temperature goes out of range?",
    "What is the RACE procedure during a hospital fire emergency?",
]

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Vector database search -- interactive</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }}
  h1 {{ font-size: 20px; font-weight: 600; }}
  p.sub {{ color: #555; font-size: 14px; }}
  select {{ width: 100%; padding: 8px; margin-bottom: 12px; font-size: 14px; }}
  #results div {{ padding: 8px 0; border-top: 1px solid #e5e5e0; font-size: 13px; }}
  #results b {{ color: #111; }}
  .muted {{ color: #666; }}
</style>
</head>
<body>
<h1>Vector database search, visualized</h1>
<p class="sub">A live query gets embedded into the same space as the 99 chunks, then the vector database finds its nearest neighbors by cosine similarity. Pick a query below.</p>
<select id="querySelect"></select>
<div style="position:relative;width:100%;height:420px"><canvas id="chartSearch"></canvas></div>
<div id="results" style="margin-top:16px"></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
const points = {points_json};
const queries = {queries_json};

const select = document.getElementById('querySelect');
queries.forEach((q, i) => {{
  const opt = document.createElement('option');
  opt.value = i;
  opt.textContent = q.query;
  select.appendChild(opt);
}});

const linePlugin = {{
  id: 'searchLines',
  afterDatasetsDraw(chart) {{
    const q = chart.$queryPoint;
    if (!q) return;
    const xScale = chart.scales.x, yScale = chart.scales.y;
    const ctx = chart.ctx;
    const qx = xScale.getPixelForValue(q.x), qy = yScale.getPixelForValue(q.y);
    ctx.save();
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1.5;
    q.neighbors.forEach(n => {{
      const nx = xScale.getPixelForValue(n.px), ny = yScale.getPixelForValue(n.py);
      ctx.strokeStyle = 'rgba(216,90,48,0.6)';
      ctx.beginPath();
      ctx.moveTo(qx, qy);
      ctx.lineTo(nx, ny);
      ctx.stroke();
    }});
    ctx.restore();
  }}
}};
Chart.register(linePlugin);

let chart = null;
function render(qIndex) {{
  const q = queries[qIndex];
  const neighborIdx = new Set(q.neighbors.map(n => n.i));
  const background = points.filter((p, i) => !neighborIdx.has(i));
  const matches = q.neighbors.map((n, rank) => ({{ ...points[n.i], score: n.s, rank: rank + 1 }}));

  if (chart) chart.destroy();
  chart = new Chart(document.getElementById('chartSearch'), {{
    type: 'scatter',
    data: {{
      datasets: [
        {{ label: 'other chunks', data: background, backgroundColor: 'rgba(150,150,150,0.35)', pointRadius: 4 }},
        {{ label: 'top matches', data: matches, backgroundColor: '#D85A30', borderColor: '#712B13', borderWidth: 1.5, pointRadius: 8 }},
        {{ label: 'your query', data: [{{ x: q.x, y: q.y }}], backgroundColor: '#26215C', pointStyle: 'star', pointRadius: 12, pointHoverRadius: 14 }}
      ]
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      plugins: {{
        legend: {{ display: true, position: 'top', labels: {{ boxWidth: 10, font: {{ size: 11 }} }} }},
        tooltip: {{
          callbacks: {{
            label: (ctx) => {{
              const raw = ctx.raw;
              if (ctx.datasetIndex === 2) return 'query: ' + q.query;
              if (ctx.datasetIndex === 1) return '#' + raw.rank + ' score=' + raw.score.toFixed(3) + '  ' + raw.file;
              return raw.file + ' (chunk ' + raw.idx + ')';
            }}
          }}
        }}
      }},
      scales: {{
        x: {{ title: {{ display: true, text: 'dimension 1 (arbitrary units)' }} }},
        y: {{ title: {{ display: true, text: 'dimension 2 (arbitrary units)' }} }}
      }}
    }}
  }});
  chart.$queryPoint = {{ x: q.x, y: q.y, neighbors: q.neighbors.map(n => ({{ px: points[n.i].x, py: points[n.i].y }})) }};
  chart.update();

  const results = document.getElementById('results');
  results.innerHTML = '<div style="font-size:13px;font-weight:500;margin-bottom:8px">Top ' + matches.length + ' matches:</div>' +
    matches.map(m =>
      '<div><span class="muted">#' + m.rank + ' score=' + m.score.toFixed(3) + '</span> &middot; ' +
      '<b>' + m.file + '</b> (chunk ' + m.idx + ')<br>' +
      '<span class="muted">' + m.text + '&hellip;</span></div>'
    ).join('');
}}

select.addEventListener('change', () => render(parseInt(select.value, 10)));
render(0);
</script>
</body>
</html>
"""


def main():
    vectors = np.load(EMBEDDINGS_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)["chunks"]
    with open(EMBEDDER_PATH, "rb") as f:
        embedder = pickle.load(f)

    print(f"Loaded {vectors.shape[0]} vectors and the '{embedder.name}' embedder\n")

    # Fit the projection once on the corpus, then reuse it (via .transform,
    # not .fit_transform) to place each query in the SAME 2D space -- this
    # is the same principle as reusing the fitted TF-IDF vectorizer to
    # embed queries in 05_retrieval.py: the query must land in the space
    # the index was built in, not a freshly-fit one.
    svd = TruncatedSVD(n_components=2, random_state=42)
    coords = svd.fit_transform(vectors)

    points = []
    for (x, y), c in zip(coords, chunks):
        points.append({
            "x": round(float(x), 5), "y": round(float(y), 5),
            "doc": c["doc_id"], "file": c["filename"], "type": c["source_type"],
            "idx": c["chunk_index"], "text": c["text"][:220].replace("\n", " "),
        })

    query_data = []
    for q in DEMO_QUERIES:
        qvec = embedder.transform([q], input_type="query")[0]
        sims = vectors @ qvec  # real cosine similarity, full-dimensional space
        top_idx = np.argsort(-sims)[:TOP_K]
        qxy = svd.transform(qvec.reshape(1, -1))[0]
        query_data.append({
            "query": q,
            "x": round(float(qxy[0]), 5),
            "y": round(float(qxy[1]), 5),
            "neighbors": [{"i": int(i), "s": round(float(sims[i]), 4)} for i in top_idx],
        })
        print(f"{q}")
        for i in top_idx:
            print(f"  {sims[i]:.3f}  {chunks[i]['filename']} (chunk {chunks[i]['chunk_index']})")
        print()

    html = HTML_TEMPLATE.format(
        points_json=json.dumps(points, separators=(",", ":")),
        queries_json=json.dumps(query_data, separators=(",", ":")),
    )
    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Wrote {HTML_PATH}")
    print("Open it directly in your browser -- no download step needed.")


if __name__ == "__main__":
    main()
