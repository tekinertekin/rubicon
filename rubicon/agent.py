"""The decision loop — a LangGraph state graph.

This is what makes rubicon an *agent* rather than a pipeline: the order of steps
is not fixed. The agent retrieves evidence, checks its own confidence, and then
decides for itself whether to

  * finalize            (confident enough),
  * gather more evidence (uncertain, budget remains), or
  * escalate to a human (uncertain, out of budget).

The loop runs until it commits. A fixed pipeline would do each step once, in order.

An optional LLM step (set OPENAI_API_KEY) lets the model re-rank the top candidates
from the abstract. Without a key the agent falls back to the calibrated retrieval
vote, so the demo runs end to end with no account and no network beyond arXiv.
"""
from __future__ import annotations

import os
from typing import Callable, TypedDict

from langgraph.graph import END, StateGraph

from . import calibration as cal
from .tools import Index, category_votes, search_similar

# How many neighbours to start with, how far the agent may expand, and the step.
K_START = 4
K_MAX = 12
K_STEP = 4


class State(TypedDict, total=False):
    abstract: str
    index: Index
    k: int                       # current neighbour budget
    verdict: cal.Verdict         # latest calibrated assessment
    decision: str                # final category, or "" if escalated
    escalated: bool
    steps: list[str]             # trace of what the agent did, for inspection


# Optional LLM hook: (abstract, [candidate categories]) -> chosen category.
# Left as a seam so the heuristic path stays the default and always runnable.
LLMDecider = Callable[[str, list[str]], str]
_llm: LLMDecider | None = None


def set_llm(decider: LLMDecider) -> None:
    global _llm
    _llm = decider


# --------------------------------------------------------------------------- #
# Nodes                                                                        #
# --------------------------------------------------------------------------- #
def retrieve(state: State) -> State:
    k = state.get("k", K_START)
    neighbours = search_similar(state["abstract"], state["index"], k=k)
    verdict = cal.assess(category_votes(neighbours))
    steps = state.get("steps", []) + [f"retrieve(k={k}) -> {verdict.band} {verdict.confidence:.2f}"]
    return {**state, "k": k, "verdict": verdict, "steps": steps}


def plan(state: State) -> State:
    """Pure routing node; the branch is chosen in `route` below."""
    return state


def route(state: State) -> str:
    v = state["verdict"]
    if v.band == "high":
        return "finalize"
    if state["k"] < K_MAX:
        return "expand"          # uncertain, but we can still gather more
    return "escalate"            # uncertain and out of budget


def expand(state: State) -> State:
    return {**state, "k": state["k"] + K_STEP}


def finalize(state: State) -> State:
    v = state["verdict"]
    category = v.category
    if _llm is not None:
        top = sorted(v.probs, key=v.probs.get, reverse=True)[:3]
        category = _llm(state["abstract"], top) or category
    return {
        **state,
        "decision": category,
        "escalated": False,
        "steps": state["steps"] + [f"finalize -> {category}"],
    }


def escalate(state: State) -> State:
    return {
        **state,
        "decision": "",
        "escalated": True,
        "steps": state["steps"] + ["escalate -> human review"],
    }


# --------------------------------------------------------------------------- #
# Graph                                                                        #
# --------------------------------------------------------------------------- #
def build_agent():
    g = StateGraph(State)
    g.add_node("retrieve", retrieve)
    g.add_node("plan", plan)
    g.add_node("expand", expand)
    g.add_node("finalize", finalize)
    g.add_node("escalate", escalate)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "plan")
    g.add_conditional_edges(
        "plan", route, {"expand": "expand", "finalize": "finalize", "escalate": "escalate"}
    )
    g.add_edge("expand", "retrieve")   # the loop: gather more, re-assess
    g.add_edge("finalize", END)
    g.add_edge("escalate", END)
    return g.compile()


def classify(abstract: str, index: Index) -> State:
    """Run the agent on one abstract and return its final state."""
    agent = build_agent()
    return agent.invoke({"abstract": abstract, "index": index})
