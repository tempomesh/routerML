"""Pulse dual-encoder (encode-once) — the serveable product architecture.

One shared transformer encodes the STATE once and each LABEL once; the decision is
a similarity (dot product of L2-normalized vectors), scaled by a learned
temperature (CLIP-style logit_scale). This is:
  - fast to serve: encode the state once, compare to CACHED label vectors;
  - fast to train: ~B states + L labels per step, not B*L cross-encodings;
  - zero-shot capable: new label sets work by encoding their text.

Cross-encoder Pulse remains available as the high-interaction verifier for the
hybrid. Requires torch + transformers (>=4.48 for ModernBERT). Status:
EXPERIMENTAL until measured on the untouched test split.
"""
from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import nn
    from transformers import AutoModel, AutoTokenizer
    _HAVE = True
except Exception:  # pragma: no cover
    _HAVE = False
    nn = object  # type: ignore


@dataclass
class DualConfig:
    encoder_name: str = "answerdotai/ModernBERT-base"
    max_length: int = 128
    model_version: str = "pulse-dual-0.0.0-experimental"


if _HAVE:

    class PulseDual(nn.Module):
        def __init__(self, cfg: DualConfig) -> None:
            super().__init__()
            self.cfg = cfg
            self.encoder = AutoModel.from_pretrained(cfg.encoder_name)
            import math
            # learned temperature, initialized like CLIP (1/0.07)
            self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

        def _pool(self, input_ids, attention_mask):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
            return torch.nn.functional.normalize(pooled, dim=-1)

        def encode(self, input_ids, attention_mask):
            """L2-normalized representation (state or label). Cache label outputs."""
            return self._pool(input_ids, attention_mask)

        def logits(self, state_vec, label_vec):
            """(B,H)·(L,H) -> (B,L) scaled similarities."""
            scale = self.logit_scale.exp().clamp(max=100.0)
            return scale * state_vec @ label_vec.t()

    def load_tokenizer(cfg: DualConfig):
        return AutoTokenizer.from_pretrained(cfg.encoder_name)

else:  # pragma: no cover

    class PulseDual:  # type: ignore
        def __init__(self, *_, **__):
            raise RuntimeError("Install torch+transformers (>=4.48) in a venv.")

    def load_tokenizer(_cfg):  # type: ignore
        raise RuntimeError("torch/transformers not installed.")


def humanize_labels(labels: list[str]) -> list[str]:
    """banking77 labels like 'card_arrival' -> a natural phrase the encoder reads."""
    return [f"banking intent: {l.replace('_', ' ')}" for l in labels]


def schema_label_text(label: str) -> str:
    """Canonical label rendering used EVERYWHERE (train_schema, eval_schema,
    fast_decisions_dual) so the label prompt at eval matches training exactly.

    train_schema encodes the dataset's raw `choices` strings, so the canonical is
    the raw label (identity). Any prefix here MUST also be applied in training, or
    the eval prompt won't match what the model learned (that mismatch produced the
    28.8% Fast Decisions dip). Kept as one function so a future change stays in sync.
    """
    return label
