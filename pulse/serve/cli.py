"""PulseKit CLI — the developer surface.

NOTE: --ckpt is a TOP-LEVEL flag, so it comes BEFORE the subcommand:
  python -m pulse.serve.cli --ckpt <ckpt> decide "I was charged twice"
  python -m pulse.serve.cli --ckpt <ckpt> serve --port 8088
  python -m pulse.serve.cli --ckpt <ckpt> bench -n 200

Keeps the "5-minute local try" promise: load a checkpoint, get a typed decision
with calibrated confidence and an escalate signal, or start a local HTTP endpoint.
"""
from __future__ import annotations

import argparse
import json
import time


def _decide(args):
    from pulse.serve.engine import PulseEngine
    eng = PulseEngine.load(args.ckpt, temperature=args.temperature)
    d = eng.decide(args.state, min_confidence=args.min_confidence, top_k=args.top_k)
    print(json.dumps(d.to_dict(), indent=2))


def _bench(args):
    from pulse.serve.engine import PulseEngine
    eng = PulseEngine.load(args.ckpt, temperature=args.temperature)
    sample = args.state or "I was charged twice and need a refund"
    # warmup
    eng.decide(sample)
    lat = []
    for _ in range(args.n):
        t0 = time.perf_counter(); eng.decide(sample); lat.append((time.perf_counter()-t0)*1000)
    lat.sort()
    print(json.dumps({"n": args.n, "p50_ms": round(lat[len(lat)//2], 2),
                      "p95_ms": round(lat[int(len(lat)*0.95)], 2),
                      "min_ms": round(lat[0], 2)}, indent=2))


def _serve(args):
    from pulse.serve.server import serve
    serve(args.ckpt, host=args.host, port=args.port, temperature=args.temperature,
          min_confidence=args.min_confidence)


def main() -> None:
    ap = argparse.ArgumentParser(prog="pulse")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--temperature", type=float, default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("decide"); d.add_argument("state")
    d.add_argument("--min-confidence", type=float, default=0.0)
    d.add_argument("--top-k", type=int, default=3); d.set_defaults(fn=_decide)

    b = sub.add_parser("bench"); b.add_argument("--n", type=int, default=200)
    b.add_argument("--state", default=None); b.set_defaults(fn=_bench)

    s = sub.add_parser("serve"); s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8088)
    s.add_argument("--min-confidence", type=float, default=0.0); s.set_defaults(fn=_serve)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
