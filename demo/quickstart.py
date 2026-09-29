"""Pulse Decide 150M — smallest possible working example.

Downloads the model from the Hugging Face Hub, makes one decision, and shows the
governance signal: act when confident, escalate when not.

    pip install torch transformers huggingface_hub
    python demo/quickstart.py
"""
from __future__ import annotations

import torch
from huggingface_hub import hf_hub_download

from pulse.model import PulseConfig, PulseUnified, load_tokenizer

REPO = "RouterML/pulse-decide-150m"
GATE = 0.60          # act only at/above this confidence; escalate below it


def load():
    ckpt = hf_hub_download(REPO, "pulse_decide_150m.pt")
    blob = torch.load(ckpt, map_location="cpu", weights_only=True)
    cfg = PulseConfig(**blob["cfg"])
    model = PulseUnified(cfg)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, load_tokenizer(cfg), cfg


def decide(model, tok, cfg, state, question, choices, context=None):
    """Score every choice, return (label, confidence).

    Choices are read at RUNTIME, so label sets the model never saw in training work too.
    """
    ctx = " | ".join(context) + " || " if context else ""
    left = [f"{ctx}{state} [SEP] {question}"] * len(choices)
    enc = tok(left, choices, truncation=True, max_length=cfg.max_length,
              padding=True, return_tensors="pt")
    with torch.no_grad():
        scores = model.score_pairs(enc["input_ids"], enc["attention_mask"])
    probs = torch.softmax(scores, dim=0)
    i = int(probs.argmax())
    return choices[i], float(probs[i])


if __name__ == "__main__":
    model, tok, cfg = load()

    state = "We were invoiced twice this month and the agreed payment terms were not applied."
    question = "Which team should handle this?"
    choices = ["sales", "support", "billing", "success", "compliance"]

    # The same request, decided without and with retrieved context. Context is where a
    # governed memory layer plugs in; any list of strings works.
    for name, context in [("no context", None),
                          ("with context", ["NET-60 payment terms",
                                            "prior billing dispute on file"])]:
        label, confidence = decide(model, tok, cfg, state, question, choices, context)
        verdict = "ACT" if confidence >= GATE else "ESCALATE"
        print(f"{name:14s} -> {label:<11s} {confidence:6.1%}   {verdict}")
