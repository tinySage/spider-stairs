"""The spider-stairs dashboard.

    python scripts/dashboard.py                    # serve http://localhost:8000, curves refresh while training
    python scripts/dashboard.py --record           # first replay the latest checkpoint on every stair level
    python scripts/dashboard.py --build run.html   # write one self-contained file you can share
"""

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spider_stairs import uidata  # noqa: E402

PAGE = ROOT / "ui" / "index.html"
SLOT = "/*__DATA__*/null"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-e", "--exp-name", default="spider-stairs")
    p.add_argument("--record", action="store_true", help="record replays from a checkpoint before serving")
    p.add_argument("--ckpt", type=int, default=None)
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--build", default=None, help="write a standalone HTML file instead of serving")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    log_dir = f"logs/{args.exp_name}"

    if args.record:
        replay = uidata.record_rollouts(log_dir, args.ckpt, args.gpu)
        with open(f"{log_dir}/rollouts.json", "w") as f:
            json.dump(replay, f, separators=(",", ":"))
        print(f"recorded {len(replay['rollouts'])} replays from model_{replay['ckpt']}.pt")

    if args.build:
        data = json.dumps(uidata.assemble(log_dir, args.exp_name), separators=(",", ":"))
        html = PAGE.read_text().replace(SLOT, data.replace("</", "<\\/"))
        Path(args.build).write_text(html)
        print(f"wrote {args.build} ({len(html) / 1e6:.1f} MB)")
        return

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/data.json"):
                body, ctype = json.dumps(uidata.assemble(log_dir, args.exp_name)).encode(), "application/json"
            else:
                body, ctype = PAGE.read_bytes(), "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    print(f"dashboard for {log_dir} at http://localhost:{args.port}")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
