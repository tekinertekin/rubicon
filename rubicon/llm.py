"""Optional LLM decision step, served by a local Ollama model.

Without this, the agent decides from the retrieval vote alone (see agent.py).
With it, a real model reads the abstract and the candidate categories and makes
the call — this is what lets the agent genuinely *reason* about a case rather than
only count nearest neighbours.

Local model, no API key, no data leaving the machine. Talks to Ollama's HTTP API
over stdlib urllib, so rubicon stays dependency-light.

    from rubicon.llm import ollama_decider
    from rubicon.agent import set_llm
    set_llm(ollama_decider("qwen2.5:7b"))
"""
from __future__ import annotations

import json
import urllib.request

from .tools import TAXONOMY

_HOST = "http://127.0.0.1:11434"


def _chat(model: str, system: str, user: str, timeout: int = 120) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": 0.0},  # deterministic — we want a decision, not prose
    }
    req = urllib.request.Request(
        f"{_HOST}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["message"]["content"].strip()


def ollama_decider(model: str = "qwen2.5:7b"):
    """Return an LLMDecider: (abstract, candidate_categories) -> chosen category.

    The model is constrained to pick from the candidates the retrieval step already
    surfaced, so it refines rather than invents — and its answer is validated back
    against that list.
    """
    system = (
        "You are a precise arXiv subject classifier. You are given a paper abstract "
        "and a short list of candidate categories. Reply with EXACTLY ONE category "
        "code from the candidate list and nothing else — no words, no punctuation."
    )

    def decide(abstract: str, candidates: list[str]) -> str:
        desc = "\n".join(f"- {c}: {TAXONOMY.get(c, c)}" for c in candidates)
        user = f"Candidates:\n{desc}\n\nAbstract:\n{abstract}\n\nAnswer with one code:"
        try:
            ans = _chat(model, system, user)
        except Exception:
            return ""  # any failure -> fall back to the retrieval vote
        # keep only a candidate the model was allowed to choose
        for c in candidates:
            if c.lower() in ans.lower():
                return c
        return ""

    return decide
