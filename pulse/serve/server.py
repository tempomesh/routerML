"""Zero-dependency local HTTP server for Pulse (stdlib only).

Endpoints:
  GET  /health           -> {"ok": true, "labels": N}
  GET  /v1/labels        -> {"labels": [...]}
  POST /v1/decide        -> body {"state": "...", "min_confidence": 0.9}
                            returns the typed decision + calibrated confidence

Deliberately stdlib (http.server) so `pulse serve` runs with no extra install.
A production deployment can swap in FastAPI/uvicorn; the engine is unchanged.
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def serve(ckpt: str, host: str = "127.0.0.1", port: int = 8088,
          temperature: float | None = None, min_confidence: float = 0.0) -> None:
    # auto-select the engine by checkpoint architecture: cross-encoder (Pulse Decide)
    # or the older dual-encoder. Same interface either way.
    import torch as _t
    _arch = _t.load(ckpt, map_location="cpu", weights_only=True).get("arch")
    if _arch == "cross":
        from pulse.serve.cross_engine import CrossEngine
        engine = CrossEngine.load(ckpt)
        print(f"engine: CrossEngine (Pulse Decide, arch=cross)")
    else:
        from pulse.serve.engine import PulseEngine
        engine = PulseEngine.load(ckpt, temperature=temperature)
        print(f"engine: PulseEngine (dual-encoder, arch={_arch})")
    default_min = min_confidence

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, payload):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"ok": True, "labels": len(engine.labels)})
            elif self.path == "/v1/labels":
                self._send(200, {"labels": engine.labels})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/v1/decide":
                return self._send(404, {"error": "not found"})
            try:
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(n) or b"{}")
            except Exception:
                return self._send(400, {"error": "invalid JSON"})
            state = req.get("state")
            if not isinstance(state, str) or not state:
                return self._send(400, {"error": "missing 'state' string"})
            mc = float(req.get("min_confidence", default_min))
            kw = {}
            if req.get("choices"):    kw["choices"] = req["choices"]
            if req.get("context"):    kw["context"] = req["context"]
            if req.get("question"):   kw["question"] = req["question"]
            d = engine.decide(state, min_confidence=mc, top_k=int(req.get("top_k", 3)), **kw)
            self._send(200, d.to_dict())

        def log_message(self, *_):  # quiet
            return

    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"Pulse serving on http://{host}:{port}  ({len(engine.labels)} labels)")
    print("  POST /v1/decide  {\"state\": \"...\"}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()
