"""CrossEngine — inference core for the CROSS-encoder Pulse (Pulse Decide 150M).

The original PulseEngine serves the dual-encoder (encode labels once, cache, dot). Pulse
Decide is a cross-encoder: it scores each (state, candidate) pair jointly, which is slower
per decision but is what makes arbitrary-runtime-schema decisions work — the candidate text
is read together with the state, so label sets never seen in training still work.

Same public surface as PulseEngine, so `pulse.serve.server` / `cli` can serve either:

    engine = CrossEngine.load(ckpt, labels=[...])
    engine.labels                      -> default label set
    engine.decide(state, min_confidence=0.55)          -> Decision
    engine.decide(state, choices=[...])                -> Decision  (arbitrary schema)

`escalate` is the governance signal: True when confidence is below the contract threshold,
meaning the caller (RouterML / 8mem Gateway) should route elsewhere instead of acting.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field

DEFAULT_LABELS = ["billing", "fraud", "technical", "account_access", "shipping"]


def _device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _softmax(xs):
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e)
    return [x / s for x in e]


@dataclass
class Decision:
    label: str
    confidence: float
    escalate: bool
    latency_ms: float
    top_k: list = field(default_factory=list)      # [(label, prob), ...]
    margin: float = 0.0
    model: str = "pulse-decide-150m"

    def to_dict(self) -> dict:
        return {"label": self.label, "confidence": round(self.confidence, 4),
                "escalate": self.escalate, "latency_ms": round(self.latency_ms, 2),
                "margin": round(self.margin, 4), "model": self.model,
                "top_k": [[l, round(p, 4)] for l, p in self.top_k]}


class CrossEngine:
    def __init__(self, model, tok, cfg, labels, device, threshold: float = 0.55,
                 ckpt: str = ""):
        self._m, self._tok, self._cfg = model, tok, cfg
        self._labels = list(labels)
        self._device = device
        self.threshold = threshold
        self.ckpt = ckpt

    @classmethod
    def load(cls, ckpt: str, labels: list[str] | None = None,
             threshold: float = 0.55, device: str | None = None) -> "CrossEngine":
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        import torch
        from pulse.model import PulseConfig, PulseUnified, load_tokenizer
        dev = device or _device()
        blob = torch.load(ckpt, map_location=dev, weights_only=True)
        if blob.get("arch") not in (None, "cross"):
            raise ValueError(f"CrossEngine needs a cross-encoder checkpoint, got arch="
                             f"{blob.get('arch')!r}. Use PulseEngine for dual-encoders.")
        cfg = PulseConfig(**blob["cfg"])
        tok = load_tokenizer(cfg)
        model = PulseUnified(cfg).to(dev)
        model.load_state_dict(blob["state_dict"])
        model.eval()
        # label set: explicit > sidecar > built-in default
        if labels is None:
            side = ckpt + ".labels.json"
            if os.path.exists(side):
                labels = json.load(open(side))["choices"]
            else:
                labels = DEFAULT_LABELS
        return cls(model, tok, cfg, labels, dev, threshold, ckpt)

    @property
    def labels(self) -> list[str]:
        return list(self._labels)

    def decide(self, state: str, min_confidence: float | None = None, top_k: int = 3,
               choices: list[str] | None = None, question: str = "Which option applies?",
               context: list[str] | None = None) -> Decision:
        """Score the state against `choices` (or the default label set) and return a governed
        Decision. `context` (e.g. 8mem's compiled governed context) is prepended so the
        decision is made WITH the governed facts."""
        import torch
        opts = list(choices) if choices else self._labels
        if not opts:
            raise ValueError("no choices to decide between")
        thr = self.threshold if min_confidence is None else min_confidence
        ctx = (" | ".join(context) + " || ") if context else ""
        left = f"{ctx}{state} [SEP] {question}"

        t0 = time.time()
        with torch.no_grad():
            enc = self._tok([left] * len(opts), opts, truncation=True,
                            max_length=self._cfg.max_length, padding=True,
                            return_tensors="pt")
            raw = self._m.score_pairs(enc["input_ids"].to(self._device),
                                      enc["attention_mask"].to(self._device)).tolist()
        ms = (time.time() - t0) * 1000

        probs = _softmax(raw)
        order = sorted(range(len(opts)), key=lambda i: -probs[i])
        conf = probs[order[0]]
        runner = probs[order[1]] if len(order) > 1 else 0.0
        return Decision(
            label=opts[order[0]], confidence=conf, escalate=conf < thr, latency_ms=ms,
            top_k=[(opts[i], probs[i]) for i in order[:top_k]], margin=conf - runner,
        )

    def info(self) -> dict:
        n = sum(p.numel() for p in self._m.parameters())
        return {"model": "pulse-decide-150m", "arch": "cross-encoder", "parameters": n,
                "device": self._device, "threshold": self.threshold,
                "generative": False, "offline": True,
                "checkpoint": os.path.basename(self.ckpt)}
