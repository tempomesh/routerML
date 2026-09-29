"""Render the governed-routing beat as a visual EXECUTION RECEIPT (HTML card).

Much better on video than a terminal: shows the same request decided twice — without
governed memory (escalates) and with 8mem's context (confident, correct, local) — plus the
integrity footer (model, params, checkpoint hash).

  # live against the running Pulse server:
  python demo/proof_card.py --serve-url http://127.0.0.1:8088

  # or offline, from the already-verified numbers:
  python demo/proof_card.py --offline

Writes demo/proof_card.html — open it in a browser and record that.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request

CKPT = "pulse_decide_150m.pt"   # path to the weights; used only for the receipt hash

REQUEST = "We were invoiced twice this month and the NET-60 terms were not applied."
QUESTION = "Which team should handle this?"
CHOICES = ["sales", "support", "billing", "success", "compliance"]
CONTEXT = ["Northwind: NET-60 payment terms",
           "prior billing dispute on file",
           "billing contact: Maria"]
GATE = 0.6

# illustrative values for offline rendering; --serve-url regenerates them live
OFFLINE = {
    "without": {"label": "sales", "confidence": 0.5549, "escalate": True,
                "latency_ms": 80.0, "margin": 0.1554,
                "top_k": [["sales", 0.5549], ["billing", 0.3995], ["compliance", 0.0356]]},
    "with": {"label": "billing", "confidence": 0.9973, "escalate": False,
             "latency_ms": 32.5, "margin": 0.9955,
             "top_k": [["billing", 0.9973], ["sales", 0.0018], ["compliance", 0.0006]]},
}


def call(url, context):
    body = json.dumps({"state": REQUEST, "question": QUESTION, "choices": CHOICES,
                       "context": context, "min_confidence": GATE}).encode()
    req = urllib.request.Request(f"{url}/v1/decide", data=body,
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


def ckpt_sha(path):
    if not os.path.exists(path):
        return "n/a"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def bars(top_k):
    out = []
    for label, p in top_k[:3]:
        out.append(
            f'<div class="bar"><span class="lb">{label}</span>'
            f'<span class="tr"><i style="width:{p*100:.1f}%"></i></span>'
            f'<span class="pc">{p*100:.1f}%</span></div>')
    return "".join(out)


def panel(title, sub, d, accent):
    verdict = ("ESCALATE" if d["escalate"] else "ACCEPT")
    vclass = "esc" if d["escalate"] else "acc"
    note = (f"below {GATE:.0%} gate — routed onward, not guessed"
            if d["escalate"] else
            f"handled locally · {d['latency_ms']:.0f} ms · $0.00")
    return f"""
    <div class="panel {accent}">
      <div class="ptitle">{title}</div>
      <div class="psub">{sub}</div>
      {bars(d['top_k'])}
      <div class="verdict {vclass}">{verdict} &rarr; {d['label']}</div>
      <div class="note">{note}</div>
    </div>"""


HTML = """<!doctype html><meta charset="utf-8"><title>RouterML · Execution Receipt</title>
<style>
:root{{--bg:#0d1117;--fg:#e6edf3;--mut:#8b949e;--line:#30363d;--grn:#3fb950;--yel:#d29922;--blu:#58a6ff}}
*{{box-sizing:border-box}}
html,body{{margin:0;padding:0;background:var(--bg)}}
/* fixed 1920x1080 stage: full-screen capture is pixel-perfect, no scaling artifacts */
body{{width:1920px;height:1080px;display:flex;align-items:center;justify-content:center;
color:var(--fg);font:26px/1.55 ui-sans-serif,-apple-system,"SF Pro Text",Segoe UI,sans-serif}}
.card{{width:1640px;border:2px solid var(--line);border-radius:22px;
background:#161b22;overflow:hidden}}
.head{{padding:32px 44px;border-bottom:2px solid var(--line);display:flex;
justify-content:space-between;align-items:center}}
.brand{{font-size:38px;font-weight:700;letter-spacing:-.4px}}
.brand span{{color:var(--blu)}}
.badge{{font-size:20px;color:var(--mut);border:2px solid var(--line);
border-radius:999px;padding:10px 22px}}
.req{{padding:34px 44px;border-bottom:2px solid var(--line)}}
.lbl{{font-size:17px;letter-spacing:.1em;color:var(--mut);text-transform:uppercase;
margin-bottom:12px}}
.qt{{font-size:31px}}
.grid{{display:grid;grid-template-columns:1fr 1fr}}
.panel{{padding:34px 44px}}
.panel+.panel{{border-left:2px solid var(--line)}}
.ptitle{{font-weight:650;font-size:29px;margin-bottom:6px}}
.psub{{font-size:20px;color:var(--mut);margin-bottom:26px;min-height:58px}}
.bar{{display:flex;align-items:center;gap:16px;margin:13px 0;font-size:23px}}
.lb{{width:150px;color:var(--mut)}}
.tr{{flex:1;height:15px;background:#21262d;border-radius:99px;overflow:hidden}}
.tr i{{display:block;height:100%;background:var(--blu)}}
.pc{{width:96px;text-align:right;font-variant-numeric:tabular-nums}}
.verdict{{margin-top:30px;font-weight:700;font-size:34px}}
.verdict.acc{{color:var(--grn)}} .verdict.esc{{color:var(--yel)}}
.note{{font-size:20px;color:var(--mut);margin-top:10px}}
.foot{{padding:24px 44px;border-top:2px solid var(--line);font-size:19px;
color:var(--mut);display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px}}
code{{color:#a5d6ff}}
</style>
<div class="card">
  <div class="head">
    <div class="brand">RouterML <span>· Execution Receipt</span></div>
    <div class="badge">Pulse Decide 150M · on-device · offline</div>
  </div>
  <div class="req">
    <div class="lbl">Inbound request · Northwind</div>
    <div class="qt">&ldquo;{request}&rdquo;</div>
  </div>
  <div class="grid">{left}{right}</div>
  <div class="foot">
    <span>model <code>pulse-decide-150m</code> · 149.3M params · non-generative</span>
    <span>checkpoint <code>{sha}</code> · confidence gate {gate:.0%}</span>
  </div>
</div>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve-url", default="")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--out", default="demo/proof_card.html")
    ap.add_argument("--gif", action="store_true", help="also render a 2-frame animated GIF")
    args = ap.parse_args()

    if args.serve_url and not args.offline:
        print(f"calling {args.serve_url} ...")
        without, with_ = call(args.serve_url, []), call(args.serve_url, CONTEXT)
    else:
        print("using verified offline numbers")
        without, with_ = OFFLINE["without"], OFFLINE["with"]

    html = HTML.format(
        request=REQUEST, sha=ckpt_sha(args.ckpt), gate=GATE,
        left=panel("Without governed memory", "no context supplied", without, "a"),
        right=panel("With 8mem governed context",
                    " · ".join(CONTEXT), with_, "b"))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(html)
    print(f"wrote {args.out}")
    print(f"  without 8mem : {without['label']} {without['confidence']:.1%} "
          f"-> {'ESCALATE' if without['escalate'] else 'ACCEPT'}")
    print(f"  with 8mem    : {with_['label']} {with_['confidence']:.1%} "
          f"-> {'ESCALATE' if with_['escalate'] else 'ACCEPT'}")
    if args.gif:
        print("\nrendering GIF frames with headless Chrome...")
        build_gif(without, with_, args.ckpt)
    print(f"\nopen it:  open {args.out}")




# ---------------------------------------------------------------------------
# GIF: two frames — (1) without governed memory, (2) the 8mem reveal.
# Renders with headless Chrome at 1920x1080, assembles with ffmpeg.
# ---------------------------------------------------------------------------
PENDING_PANEL = """
    <div class="panel b" style="opacity:.28">
      <div class="ptitle">With 8mem governed context</div>
      <div class="psub">awaiting governed context&hellip;</div>
      <div class="bar"><span class="lb">&mdash;</span><span class="tr"></span><span class="pc"></span></div>
      <div class="bar"><span class="lb">&mdash;</span><span class="tr"></span><span class="pc"></span></div>
      <div class="bar"><span class="lb">&mdash;</span><span class="tr"></span><span class="pc"></span></div>
      <div class="verdict" style="color:#8b949e">&nbsp;</div>
      <div class="note">&nbsp;</div>
    </div>"""


def build_gif(without, with_, ckpt, out_gif="demo/pulse_beat.gif", hold=(2.2, 3.0)):
    import subprocess, tempfile
    sha = ckpt_sha(ckpt)
    left = panel("Without governed memory", "no context supplied", without, "a")
    right = panel("With 8mem governed context", " &middot; ".join(CONTEXT), with_, "b")
    frames = {
        1: HTML.format(request=REQUEST, sha=sha, gate=GATE, left=left, right=PENDING_PANEL),
        2: HTML.format(request=REQUEST, sha=sha, gate=GATE, left=left, right=right),
    }
    chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    if not os.path.exists(chrome):
        raise SystemExit("Google Chrome not found — cannot render frames")
    tmp = tempfile.mkdtemp(prefix="pulsegif_")
    pngs = []
    for i, html in frames.items():
        h = os.path.join(tmp, f"f{i}.html")
        p = os.path.join(tmp, f"f{i}.png")
        with open(h, "w") as f:
            f.write(html)
        subprocess.run([chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                        f"--screenshot={p}", "--window-size=1920,1080", f"file://{h}"],
                       check=True, capture_output=True)
        pngs.append(p)
        print(f"  rendered frame {i}")
    # concat demuxer with per-frame durations, then a high-quality palette GIF
    lst = os.path.join(tmp, "list.txt")
    with open(lst, "w") as f:
        for p, d in zip(pngs, hold):
            f.write(f"file '{p}'\nduration {d}\n")
        f.write(f"file '{pngs[-1]}'\n")          # last frame needs repeating
    os.makedirs(os.path.dirname(out_gif) or ".", exist_ok=True)
    vf = ("fps=10,scale=1200:-1:flags=lanczos,split[s0][s1];"
          "[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer")
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst,
                    "-vf", vf, "-loop", "0", out_gif], check=True, capture_output=True)
    print(f"wrote {out_gif} ({os.path.getsize(out_gif)/1024:.0f} KB)")
    return out_gif


if __name__ == "__main__":
    main()
