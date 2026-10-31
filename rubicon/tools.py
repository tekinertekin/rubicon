"""Tools the agent may call.

Each tool is a plain function. The agent decides which to call and when; nothing
here fixes an order. Real, public data only — arXiv's open API, no key required.
"""
from __future__ import annotations

import math
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field

_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV = "{http://arxiv.org/schemas/atom}"

# A small slice of the arXiv taxonomy — enough for the demo, extend freely.
TAXONOMY: dict[str, str] = {
    "cs.LG": "Machine Learning",
    "cs.CL": "Computation and Language (NLP)",
    "cs.CV": "Computer Vision and Pattern Recognition",
    "cs.AI": "Artificial Intelligence",
    "math.NA": "Numerical Analysis",
    "stat.ML": "Machine Learning (Statistics)",
    "cs.DC": "Distributed, Parallel, and Cluster Computing",
    "cs.PL": "Programming Languages",
}


@dataclass
class Paper:
    arxiv_id: str
    title: str
    abstract: str
    primary_category: str
    categories: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Tool 1 — fetch real papers from the arXiv API                               #
# --------------------------------------------------------------------------- #
def fetch_arxiv(category: str, max_results: int = 50, *, start: int = 0) -> list[Paper]:
    """Pull real papers for one category. Public API, polite rate limit."""
    q = urllib.parse.urlencode(
        {
            "search_query": f"cat:{category}",
            "start": start,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
    )
    url = f"http://export.arxiv.org/api/query?{q}"
    req = urllib.request.Request(url, headers={"User-Agent": "rubicon-agent/0.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        root = ET.fromstring(r.read())
    papers: list[Paper] = []
    for e in root.findall(f"{_ATOM}entry"):
        prim = e.find(f"{_ARXIV}primary_category")
        cats = [c.get("term") for c in e.findall(f"{_ATOM}category")]
        papers.append(
            Paper(
                arxiv_id=(e.findtext(f"{_ATOM}id") or "").rsplit("/", 1)[-1],
                title=" ".join((e.findtext(f"{_ATOM}title") or "").split()),
                abstract=" ".join((e.findtext(f"{_ATOM}summary") or "").split()),
                primary_category=prim.get("term") if prim is not None else "",
                categories=cats,
            )
        )
    time.sleep(3)  # arXiv asks for ~1 request / 3s
    return papers


# --------------------------------------------------------------------------- #
# Tool 2 — look up similar papers (the agent's "search" tool)                  #
# A tiny stdlib TF-IDF index so the demo runs with zero heavy dependencies.    #
# --------------------------------------------------------------------------- #
_TOKEN = re.compile(r"[a-z]{3,}")
_STOP = set(
    "the and for are with that this from they have was not our can all any its "
    "which when where while into than then them these those such also may over "
    "using used use based show shows propose proposed method methods results "
    "model models paper approach approaches new via".split()
)


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


@dataclass
class Index:
    labels: list[str]                       # true category per indexed doc
    vectors: list[dict[str, float]]         # tf-idf vector per doc
    idf: dict[str, float]


def build_index(papers: list[Paper]) -> Index:
    docs = [_tokens(p.title + " " + p.abstract) for p in papers]
    df: Counter = Counter()
    for d in docs:
        df.update(set(d))
    n = len(docs)
    idf = {w: math.log((n + 1) / (c + 1)) + 1.0 for w, c in df.items()}
    vectors = []
    for d in docs:
        tf = Counter(d)
        vec = {w: (f / len(d)) * idf[w] for w, f in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        vectors.append({w: v / norm for w, v in vec.items()})
    return Index(labels=[p.primary_category for p in papers], vectors=vectors, idf=idf)


def _vectorize(text: str, idf: dict[str, float]) -> dict[str, float]:
    d = _tokens(text)
    tf = Counter(d)
    vec = {w: (f / len(d)) * idf.get(w, 0.0) for w, f in tf.items() if w in idf}
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    return {w: v / norm for w, v in vec.items()}


def search_similar(abstract: str, index: Index, k: int = 5) -> list[tuple[str, float]]:
    """Return the k most similar indexed papers as (category, similarity)."""
    q = _vectorize(abstract, index.idf)
    sims = []
    for lbl, vec in zip(index.labels, index.vectors):
        s = sum(q[w] * vec.get(w, 0.0) for w in q)
        sims.append((lbl, s))
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:k]


def category_votes(neighbors: list[tuple[str, float]]) -> dict[str, float]:
    """Collapse neighbours into a score per category (sum of similarities)."""
    votes: dict[str, float] = defaultdict(float)
    for cat, sim in neighbors:
        votes[cat] += sim
    return dict(votes)
