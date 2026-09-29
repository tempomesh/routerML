"""Temperature scaling — fit on the CALIBRATION split only (never train/test).

Given raw per-choice logits and the correct index, find a single scalar T>0 that
minimizes negative log-likelihood of softmax(logits / T). T is then applied at
inference so the reported probabilities mean what they say (ECE drops). This is
the standard, honest calibration step; reliability is verified on the untouched
test split, not here.

Pure Python (grid + local refine) so it needs no torch — the harness can calibrate
from logits dumped by the model.
"""
from __future__ import annotations

import math


def _softmax(logits: list[float], T: float) -> list[float]:
    m = max(l / T for l in logits)
    exps = [math.exp(l / T - m) for l in logits]
    s = sum(exps)
    return [e / s for e in exps]


def _nll(logit_sets: list[list[float]], correct_idx: list[int], T: float) -> float:
    total = 0.0
    for logits, k in zip(logit_sets, correct_idx):
        p = _softmax(logits, T)[k]
        total += -math.log(max(p, 1e-12))
    return total / max(1, len(logit_sets))


def fit_temperature(logit_sets: list[list[float]], correct_idx: list[int],
                    lo: float = 0.05, hi: float = 10.0) -> float:
    """Coarse grid then golden-section refine for the NLL-minimizing temperature."""
    if not logit_sets:
        return 1.0
    grid = [lo + (hi - lo) * i / 40 for i in range(41)]
    best_T = min(grid, key=lambda T: _nll(logit_sets, correct_idx, T))
    a, b = max(lo, best_T - (hi - lo) / 40), min(hi, best_T + (hi - lo) / 40)
    gr = (math.sqrt(5) - 1) / 2
    c, d = b - gr * (b - a), a + gr * (b - a)
    for _ in range(40):
        if _nll(logit_sets, correct_idx, c) < _nll(logit_sets, correct_idx, d):
            b = d
        else:
            a = c
        c, d = b - gr * (b - a), a + gr * (b - a)
    return round((a + b) / 2, 4)


def apply_temperature(logits: list[float], T: float) -> list[float]:
    return _softmax(logits, T)
