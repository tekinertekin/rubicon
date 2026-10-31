"""Confidence banding + the human-escalation rule.

This is the real point of the project. Most agent demos stop at "the model picked
a label." Here the agent also asks: how sure am I? — and routes the uncertain cases
to a human instead of guessing. The same discipline as a production decision-support
system's calibrated confidence and human review queue, ported into an agent loop.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# Bands tuned for the arXiv demo; see examples/run_demo.py for how they were set.
HIGH = 0.60   # >= HIGH  -> decide automatically
LOW = 0.40    # <  LOW   -> escalate to a human
# between LOW and HIGH -> the agent gathers more evidence, then re-checks


def softmax_temperature(scores: dict[str, float], temperature: float = 0.5) -> dict[str, float]:
    """Turn raw category scores into a probability distribution.

    Temperature < 1 sharpens (more confident), > 1 flattens. Softmax over raw
    similarity sums reports confidence the model does not actually have — the
    temperature is the knob that stops that, exactly as softmax-temperature
    calibration does in production serving.
    """
    if not scores:
        return {}
    mx = max(scores.values())
    exps = {k: math.exp((v - mx) / max(temperature, 1e-6)) for k, v in scores.items()}
    z = sum(exps.values()) or 1.0
    return {k: v / z for k, v in exps.items()}


@dataclass
class Verdict:
    category: str          # the argmax category
    confidence: float      # probability of the top category
    band: str              # "high" | "medium" | "low"
    probs: dict[str, float]


def assess(scores: dict[str, float], temperature: float = 0.5) -> Verdict:
    probs = softmax_temperature(scores, temperature)
    if not probs:
        return Verdict("", 0.0, "low", {})
    cat, conf = max(probs.items(), key=lambda x: x[1])
    band = "high" if conf >= HIGH else "low" if conf < LOW else "medium"
    return Verdict(cat, conf, band, probs)
