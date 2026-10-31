"""End-to-end run on real arXiv papers.

Pulls recent papers from a handful of categories, uses most of them as the lookup
index, holds the rest out as a test set, runs the agent on each held-out abstract,
and prints the money table: accuracy by confidence band, and — the number that
matters — accuracy on the cases the agent decided ON ITS OWN vs. how many it sent
to a human.

    python -m examples.run_demo          # fetches live from arXiv
    python -m examples.run_demo --cached # reuse data/papers.jsonl if present
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from rubicon.agent import classify
from rubicon.tools import Paper, build_index, fetch_arxiv

CATEGORIES = ["cs.LG", "cs.CL", "cs.CV", "math.NA"]
PER_CAT = 60
TEST_PER_CAT = 15
CACHE = Path(__file__).resolve().parent.parent / "data" / "papers.jsonl"


def load_papers(use_cache: bool) -> list[Paper]:
    if use_cache and CACHE.exists():
        return [Paper(**json.loads(line)) for line in CACHE.read_text().splitlines()]
    papers: list[Paper] = []
    for cat in CATEGORIES:
        print(f"fetching {cat} ...", file=sys.stderr)
        papers += fetch_arxiv(cat, max_results=PER_CAT)
    CACHE.parent.mkdir(exist_ok=True)
    with CACHE.open("w") as f:
        for p in papers:
            f.write(json.dumps(p.__dict__) + "\n")
    return papers


def main() -> None:
    use_cache = "--cached" in sys.argv
    papers = load_papers(use_cache)

    by_cat: dict[str, list[Paper]] = defaultdict(list)
    for p in papers:
        if p.primary_category in CATEGORIES:
            by_cat[p.primary_category].append(p)

    train, test = [], []
    for cat in CATEGORIES:
        rows = by_cat[cat]
        test += rows[:TEST_PER_CAT]
        train += rows[TEST_PER_CAT:]
    index = build_index(train)
    print(f"index: {len(train)} papers · test: {len(test)} papers\n")

    band_tot: dict[str, int] = defaultdict(int)
    band_hit: dict[str, int] = defaultdict(int)
    auto_tot = auto_hit = escalated = 0

    for p in test:
        st = classify(p.abstract, index)
        v = st["verdict"]
        truth = p.primary_category
        band_tot[v.band] += 1
        # calibration check uses the retrieval pick (confidence signal), consistently
        if v.category == truth:
            band_hit[v.band] += 1
        if st["escalated"]:
            escalated += 1
        else:
            auto_tot += 1
            auto_hit += st["decision"] == truth
        shown = v.category if st["escalated"] else st["decision"]
        mark = "ESCALATED" if st["escalated"] else ("ok " if st["decision"] == truth else "MISS")
        print(f"  {truth:7} -> {shown:7} {v.confidence:4.2f} [{v.band:6}] {mark}  ({len(st['steps'])} steps)")

    def pct(a: int, b: int) -> str:
        return f"{100*a/b:5.1f}%" if b else "   -- "

    print("\n  retrieval confidence vs accuracy (calibration check)")
    print("  band     n   accuracy")
    for b in ("high", "medium", "low"):
        print(f"  {b:6} {band_tot[b]:3d}   {pct(band_hit[b], band_tot[b])}")
    print("\n  ------------------------------------------")
    print(f"  auto-decided (no human):  {auto_tot:3d}   acc {pct(auto_hit, auto_tot)}")
    print(f"  escalated to human:       {escalated:3d}   ({pct(escalated, len(test))} of all)")
    print(f"  overall if forced:        {len(test):3d}   acc "
          f"{pct(sum(band_hit.values()), len(test))}")


if __name__ == "__main__":
    main()
