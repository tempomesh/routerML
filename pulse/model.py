"""Pulse Unified DDM — the model (question-conditioned cross-encoder).

CORRECTION (2026-09-24): the model now READS each typed question. It does not use
fixed positional heads. For a CHOICE/BOOLEAN question it scores every declared
choice by encoding `state [SEP] question [SEP] choice`; for a SCORE question it
encodes `state [SEP] question`. So the same model answers arbitrary typed
questions with arbitrary choice sets — this is the general Pulse contract, not a
triage classifier.

"One unified model invocation" = one logical call. Internally it encodes several
short sequences (one per (question,choice) pair, one per (state,plan) pair) and
batches them; it is non-autoregressive and generates no text.

Compute: default encoder is SMALL, to validate the full pipeline first on the
64 GB MLX Mac via torch's MPS backend. The production encoder is chosen by
measured quality/calibration/latency/memory/license in DownshiftEval — NOT copied
from any other project.

Requires torch + transformers (use a supported Python environment; the system
Python 3.14 environment does not currently provide the required torch wheel).
Status: EXPERIMENTAL BASELINE. Instantiating this class creates pretrained
encoder weights plus newly initialized Pulse heads; a trained Banking77
checkpoint exists at ``checkpoints/pulse_banking77.pt`` and must be loaded
explicitly. That checkpoint is not an SCPO or general OutcomeLM model.
"""
from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import nn
    from transformers import AutoModel, AutoTokenizer
    _HAVE_TORCH = True
except Exception:  # pragma: no cover
    _HAVE_TORCH = False
    nn = object  # type: ignore


@dataclass
class PulseConfig:
    # SMALL default for first end-to-end pipeline validation (~22M params, 384 hid).
    # Swap up (and re-measure) via DownshiftEval; do not hardcode a "winner".
    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    hidden_size: int = 384
    max_length: int = 256
    dropout: float = 0.1
    model_version: str = "pulse-1-unified-0.0.0-experimental"


if _HAVE_TORCH:

    class PulseUnified(nn.Module):
        """Shared cross-encoder + scalar scorers.

        Public scoring API (used by trainer, evaluator, planner):
          score_pairs(seq_ids, seq_mask) -> (N,) raw scalar scores
        The collator builds the sequences; grouping (below) turns per-choice
        scores into per-question distributions.
        """

        def __init__(self, cfg: PulseConfig) -> None:
            super().__init__()
            self.cfg = cfg
            self.encoder = AutoModel.from_pretrained(cfg.encoder_name)
            h = cfg.hidden_size
            self.drop = nn.Dropout(cfg.dropout)
            # one scalar scorer reused for choice-scoring AND score-questions;
            # a separate abstain scorer; a separate policy scorer.
            self.value_head = nn.Linear(h, 1)     # choice score / SCORE logit
            self.abstain_head = nn.Linear(h, 1)   # per (state,question) abstain logit
            self.policy_head = nn.Sequential(
                nn.Linear(h, h // 2), nn.GELU(), nn.Linear(h // 2, 1)
            )

        def _pool(self, input_ids, attention_mask):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
            return self.drop(pooled)

        def score_pairs(self, seq_ids, seq_mask):
            """Raw scalar per input sequence (state+question+choice, or state+question)."""
            return self.value_head(self._pool(seq_ids, seq_mask)).squeeze(-1)

        def abstain_logits(self, seq_ids, seq_mask):
            return self.abstain_head(self._pool(seq_ids, seq_mask)).squeeze(-1)

        def policy_scores(self, seq_ids, seq_mask):
            """Raw scalar per (state, plan) sequence; planner ranks by this."""
            return self.policy_head(self._pool(seq_ids, seq_mask)).squeeze(-1)

    @staticmethod
    def grouped_softmax(scores, groups):  # type: ignore
        """Turn flat per-choice `scores` into per-question distributions.
        `groups` is a list of (start, end) slices, one per CHOICE/BOOLEAN
        question, with variable choice counts. Returns list of prob tensors."""
        return [torch.softmax(scores[s:e], dim=0) for (s, e) in groups]

    def load_tokenizer(cfg: PulseConfig):
        return AutoTokenizer.from_pretrained(cfg.encoder_name)

else:  # pragma: no cover

    class PulseUnified:  # type: ignore
        def __init__(self, *_: object, **__: object) -> None:
            raise RuntimeError(
                "torch/transformers not installed. Create a Python 3.12 venv and "
                "`pip install -r pulse/requirements.txt` (MPS works on the MLX Mac)."
            )

    def load_tokenizer(_cfg):  # type: ignore
        raise RuntimeError("torch/transformers not installed.")
