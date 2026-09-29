"""PulseEngine — the reusable inference core for the dual-encoder product.

Loads a trained dual-encoder, encodes the label set ONCE and caches the vectors,
then answers many decisions fast: encode the state once, dot with cached labels,
temperature-calibrate, return a typed decision + calibrated confidence + an
escalate signal when confidence is below the contract threshold.

This is what `pulse decide`, the local server, and RouterML's L1 adapter all call.
Requires torch + transformers. Calibration temperature is supplied explicitly (or
via a `<ckpt>.calib.json` sidecar) so the served probabilities mean what they say.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import torch

from pulse.dual.model import DualConfig, PulseDual, load_tokenizer
from pulse.eval.calibrate import apply_temperature


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Decision:
    label: str
    confidence: float
    escalate: bool
    top_k: list[tuple[str, float]]
    latency_ms: float
    calibrated: bool

    def to_dict(self) -> dict:
        return {"label": self.label, "confidence": round(self.confidence, 4),
                "escalate": self.escalate,
                "top_k": [{"label": l, "p": round(p, 4)} for l, p in self.top_k],
                "latency_ms": round(self.latency_ms, 2), "calibrated": self.calibrated}


class PulseEngine:
    def __init__(self, model, tok, labels, label_vec, cfg, temperature, device):
        self._m, self._tok, self._labels = model, tok, labels
        self._label_vec, self._cfg = label_vec, cfg
        self._T, self._device = temperature, device

    @classmethod
    def load(cls, ckpt: str, temperature: float | None = None) -> "PulseEngine":
        device = _device()
        blob = torch.load(ckpt, map_location=device)
        cfg = DualConfig(**blob["cfg"])
        labels, label_texts = blob["labels"], blob["label_texts"]
        tok = load_tokenizer(cfg)
        model = PulseDual(cfg).to(device)
        model.load_state_dict(blob["state_dict"]); model.eval()
        with torch.no_grad():
            lab = tok(label_texts, truncation=True, max_length=cfg.max_length,
                      padding=True, return_tensors="pt").to(device)
            label_vec = model.encode(lab["input_ids"], lab["attention_mask"])  # cached
        if temperature is None:
            side = ckpt.replace(".pt", ".calib.json")
            temperature = (json.load(open(side)).get("temperature", 1.0)
                           if os.path.exists(side) else 1.0)
        return cls(model, tok, labels, label_vec, cfg, temperature, device)

    @torch.no_grad()
    def decide(self, state: str, min_confidence: float = 0.0, top_k: int = 3) -> Decision:
        t0 = time.perf_counter()
        st = self._tok(state, truncation=True, max_length=self._cfg.max_length,
                       padding=True, return_tensors="pt").to(self._device)
        sv = self._m.encode(st["input_ids"], st["attention_mask"])
        logits = self._m.logits(sv, self._label_vec)[0].tolist()
        probs = apply_temperature(logits, self._T)
        ranked = sorted(zip(self._labels, probs), key=lambda x: x[1], reverse=True)
        label, conf = ranked[0]
        return Decision(label=label, confidence=conf,
                        escalate=conf < min_confidence,
                        top_k=ranked[:top_k],
                        latency_ms=(time.perf_counter() - t0) * 1000,
                        calibrated=self._T != 1.0)

    @property
    def labels(self) -> list[str]:
        return list(self._labels)
