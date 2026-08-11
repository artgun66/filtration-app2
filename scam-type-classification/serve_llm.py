"""A local HTTP endpoint for the 3B teacher, so the web app can show it beside the head.

    python serve_llm.py                 # loads Qwen once, listens on 8000
    python serve_llm.py --port 9000 --fp16

This is a **research tool**, and the distinction matters more here than usual. The phone
app and the web app both run entirely on the device; that is their central claim and the
reason they ask for no permissions. Calling this endpoint breaks that -- the message
leaves the device and travels to whatever machine is running Qwen.

So it is built to be impossible to enable by accident:

  - the web app only calls it when VITE_LLM_URL is set, and never in the phone build;
  - a response is always labelled so the UI can say the message was sent away;
  - nothing here is imported by app-backend/, core/, app/ or the production web build.

It binds to 127.0.0.1 unless told otherwise, because a 3B text generator reachable from
the open internet is someone else's compute budget.
"""
import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import scam_type_prompt as S          # noqa: E402

MAX_CHARS = 2000        # the prompt truncates at 600; this just bounds what we accept
STATE = {}


def classify(text):
    t0 = time.time()
    (kind, conf), = S.classify_choice(STATE["tok"], STATE["model"], [text])
    detail, = S.explain(STATE["tok"], STATE["model"], [text], [kind])
    return {
        "type": kind,
        "confidence": round(float(conf), 4),
        "warning_signs": (detail or {}).get("warning_signs", []),
        "explanation": (detail or {}).get("explanation", ""),
        "model": S.MODEL,
        "seconds": round(time.time() - t0, 2),
        # The web UI shows this verbatim. It is not decoration.
        "left_the_device": True,
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        # The page is served by Vite on another port, so this is cross-origin.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(204, {})

    def do_GET(self):
        # A cheap probe, so the page can hide the panel when this is not running.
        self._send(200, {"ready": "model" in STATE, "model": S.MODEL})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
            text = json.loads(self.rfile.read(n) or b"{}").get("text", "").strip()
        except Exception as exc:
            self._send(400, {"error": f"bad request: {exc}"})
            return
        if not text:
            self._send(400, {"error": "no text"})
            return
        try:
            self._send(200, classify(text[:MAX_CHARS]))
        except Exception as exc:                       # keep the server up
            self._send(500, {"error": repr(exc)})

    def log_message(self, fmt, *args):
        print(f"  {self.address_string()} {fmt % args}")


def main(port=8000, host="127.0.0.1", fp16=False):
    print(f"loading {S.MODEL} ({'fp16' if fp16 else '4-bit nf4'})...")
    STATE["tok"], STATE["model"] = S.load_model(four_bit=not fp16)
    print(f"\nlistening on http://{host}:{port}")
    print("  GET  /   -> readiness probe")
    print("  POST /   -> {\"text\": \"...\"}")
    if host != "127.0.0.1":
        print("\n  NOTE: bound beyond localhost. Anything that can reach this port can "
              "spend\n        your GPU, and every message posted to it leaves the "
              "sender's device.")
    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="127.0.0.1",
                   help="0.0.0.0 to accept from the LAN -- read the note above first")
    p.add_argument("--fp16", action="store_true", help="skip 4-bit (needs >6 GB free)")
    main(**vars(p.parse_args()))
