"""splat2mesh's local app: a wizard for one object or a batch, a job queue with progress, the components' status and
installation, settings. Standard library only; serves http://127.0.0.1:<port> and opens it.

    python -m splat2mesh.app [--port 8765] [--no-browser] [--work <folder>]

Jobs run one at a time (one GPU) as subprocesses: splat2mesh.app.pipeline (an object) or splat2mesh.winsetup
(components). Their PROGRESS lines drive the progress bars; everything is kept under <work>/jobs/<id>/.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import queue
import re
import subprocess
import sys
import threading
import time
import urllib.parse
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ..config import ROOT, WINDOWS

STATIC = Path(__file__).with_name("static")
WORK = Path(os.environ.get("SPLAT2MESH_WORK") or ROOT / "work")
PROG = re.compile(r"^PROGRESS (\d+) ([^:]*): ?(.*)$")


class Jobs:
    def __init__(self, work: Path):
        self.dir = work / "jobs"; self.dir.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, dict] = {}
        self.q: queue.Queue = queue.Queue()
        self.proc: dict[str, subprocess.Popen] = {}
        self.env: dict[str, dict] = {}       # per-job secrets (a download token): memory only, never in state.json
        self.lock = threading.Lock()
        for f in sorted(self.dir.glob("*/state.json")):           # earlier sessions; unfinished ones are stopped
            try:
                j = json.loads(f.read_text())
            except Exception:
                continue
            if j["status"] in ("queued", "running"):
                j["status"] = "interrupted"
            self.jobs[j["id"]] = j
        threading.Thread(target=self._worker, daemon=True).start()

    def _save(self, j):
        (self.dir / j["id"]).mkdir(parents=True, exist_ok=True)
        (self.dir / j["id"] / "state.json").write_text(json.dumps(j, indent=1))

    def add(self, kind: str, params: dict, title: str, env: dict | None = None) -> dict:
        jid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        d = self.dir / jid; d.mkdir(parents=True)
        j = {"id": jid, "kind": kind, "title": title, "params": params, "status": "queued", "progress": 0,
             "stage": "queued", "message": "", "created": time.time(), "results": None, "error": None}
        if kind == "create":
            params["out"] = str(d / "out")
        with self.lock:
            self.jobs[jid] = j
            if env:
                self.env[jid] = env
        self._save(j); self.q.put(jid)
        return j

    def cancel(self, jid: str):
        j = self.jobs.get(jid)
        if not j:
            return
        if j["status"] == "queued":
            j["status"] = "cancelled"; self._save(j)
        p = self.proc.get(jid)
        if p and p.poll() is None:
            j["status"] = "cancelled"
            if WINDOWS:
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output=True)
            else:
                p.terminate()

    def _cmd(self, j) -> list[str]:
        d = self.dir / j["id"]
        if j["kind"] == "create":
            (d / "job.json").write_text(json.dumps(j["params"], indent=1))
            return [sys.executable, "-u", "-m", "splat2mesh.app.pipeline", str(d / "job.json")]
        p = j["params"]
        return [sys.executable, "-u", "-m", "splat2mesh.winsetup", "install", *p["components"],
                "--vae", p.get("vae", "klein"), "--iris-mode", p.get("iris_mode", "cpu-encoder")] + (["--optix"] if p.get("optix") else [])

    def _worker(self):
        while True:
            jid = self.q.get()
            j = self.jobs.get(jid)
            if not j or j["status"] != "queued":
                continue
            j["status"] = "running"; j["started"] = time.time(); self._save(j)
            log = open(self.dir / jid / "console.txt", "w", encoding="utf-8")
            env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
            env.update(self.env.pop(jid, {}))
            p = subprocess.Popen(self._cmd(j), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                 encoding="utf-8", errors="replace", cwd=ROOT, env=env)
            self.proc[jid] = p
            for line in p.stdout:
                log.write(line); log.flush()
                m = PROG.match(line.strip())
                if m:
                    j["progress"], j["stage"], j["message"] = int(m[1]), m[2], m[3]
                elif line.startswith(("FAILED", "Traceback")):
                    j["error"] = (j.get("error") or "") + line
            rc = p.wait(); log.close()
            j["finished"] = time.time()
            if j["status"] != "cancelled":
                j["status"] = "done" if rc == 0 else "failed"
            if j["kind"] == "create":
                rj = Path(j["params"]["out"]) / "results.json"
                j["results"] = json.loads(rj.read_text()) if rj.exists() else None
            if rc == 0:
                j["progress"] = 100
            self._save(j)

    def view(self, j, log_tail: int = 0) -> dict:
        v = dict(j)
        if log_tail:
            f = self.dir / j["id"] / "console.txt"
            txt = f.read_text(encoding="utf-8", errors="replace") if f.exists() else ""
            lg = Path(j["params"].get("out", "")) / "log.txt" if j["kind"] == "create" else None
            if lg and lg.exists():
                txt += "\n--- pipeline log ---\n" + lg.read_text(encoding="utf-8", errors="replace")
            v["log"] = txt[-log_tail:]
        return v


def _allowed(path: Path) -> bool:
    p = path.resolve()
    return any(str(p).startswith(str(r.resolve())) for r in (WORK, ROOT / "out", ROOT / "assets"))


class Handler(BaseHTTPRequestHandler):
    jobs: Jobs = None

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj, default=str).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def _file(self, f: Path):
        if not f.exists() or not f.is_file():
            return self._json({"error": "not found"}, 404)
        ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
        size = f.stat().st_size
        rng = self.headers.get("Range")                       # videos seek with ranges
        if rng and rng.startswith("bytes="):
            a, _, b = rng[6:].partition("-"); a = int(a or 0); b = int(b) if b else size - 1
            self.send_response(206); self.send_header("Content-Range", f"bytes {a}-{b}/{size}")
            n = b - a + 1
        else:
            a, n = 0, size; self.send_response(200)
        self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(n))
        self.send_header("Accept-Ranges", "bytes"); self.end_headers()
        with open(f, "rb") as fh:
            fh.seek(a); left = n
            while left > 0:
                chunk = fh.read(min(1 << 20, left))
                if not chunk:
                    break
                self.wfile.write(chunk); left -= len(chunk)

    def _body(self) -> bytes:
        return self.rfile.read(int(self.headers.get("Content-Length") or 0))

    def do_GET(self):
        u = urllib.parse.urlparse(self.path); q = urllib.parse.parse_qs(u.query)
        if u.path in ("/", "/index.html"):
            return self._file(STATIC / "index.html")
        if u.path.startswith("/static/"):
            f = (STATIC / u.path[len("/static/"):]).resolve()
            return self._file(f) if str(f).startswith(str(STATIC.resolve())) else self._json({}, 403)
        if u.path == "/api/status":
            from .. import winsetup
            from ..config import load
            c = load()
            return self._json({"components": winsetup.status(), "windows": WINDOWS, "root": str(ROOT),
                               "work": str(WORK), "iris_mode": c["iris"].get("mode", "gpu"),
                               "hdris": sorted(p.name for p in (ROOT / "assets/hdri").glob("*.hdr"))})
        if u.path == "/api/jobs":
            js = sorted(self.jobs.jobs.values(), key=lambda j: -j["created"])
            return self._json([self.jobs.view(j) for j in js])
        m = re.match(r"^/api/jobs/([\w-]+)$", u.path)
        if m and m[1] in self.jobs.jobs:
            return self._json(self.jobs.view(self.jobs.jobs[m[1]], log_tail=int(q.get("log", ["20000"])[0])))
        if u.path == "/api/file":
            f = Path(q.get("path", [""])[0])
            return self._file(f) if _allowed(f) else self._json({"error": "outside the work folders"}, 403)
        return self._json({"error": "not found"}, 404)

    def do_POST(self):
        u = urllib.parse.urlparse(self.path); q = urllib.parse.parse_qs(u.query)
        if u.path == "/api/upload":
            name = re.sub(r"[^\w.\-]", "_", q.get("name", ["file"])[0])[-80:]
            d = WORK / "uploads" / uuid.uuid4().hex[:8]; d.mkdir(parents=True)
            (d / name).write_bytes(self._body())
            return self._json({"path": str(d / name)})
        body = json.loads(self._body() or b"{}")
        if u.path == "/api/jobs":
            kind = body.get("kind", "create")
            if kind == "batch":                               # one job per item, same options
                out = [self.jobs.add("create", {**body["options"], **it}, it.get("name") or f"batch {i + 1}")
                       for i, it in enumerate(body["items"])]
                return self._json(out)
            if kind == "install":
                env = {}
                if body.get("hf_token"):                      # for this download only: not written anywhere
                    env["HF_TOKEN"] = body.pop("hf_token")
                return self._json(self.jobs.add("install", body, "install " + ", ".join(body["components"]), env))
            return self._json(self.jobs.add("create", body["params"], body.get("title") or "object"))
        m = re.match(r"^/api/jobs/([\w-]+)/cancel$", u.path)
        if m:
            self.jobs.cancel(m[1]); return self._json({"ok": True})
        if u.path == "/api/settings":
            from ..winsetup import USER_CONFIG, write_user_config
            up = {}
            if body.get("deepseek_key"):
                f = USER_CONFIG.parent / "deepseek.env"; f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(f"DEEPSEEK_API_KEY={body['deepseek_key'].strip()}\n")
                up["deepseek"] = {"env_file": str(f)}
            if body.get("iris_mode"):
                up["iris"] = {"mode": body["iris_mode"]}
            if up:
                write_user_config(up)
                from .. import config
                config.load.cache_clear()
            return self._json({"ok": True})
        if u.path == "/api/open":                              # show a result in the file manager
            f = Path(body.get("path", ""))
            if _allowed(f) and f.exists():
                if WINDOWS:
                    os.startfile(f if f.is_dir() else f.parent)   # noqa
                else:
                    subprocess.run(["open" if sys.platform == "darwin" else "xdg-open", str(f if f.is_dir() else f.parent)])
                return self._json({"ok": True})
            return self._json({"error": "not allowed"}, 403)
        return self._json({"error": "not found"}, 404)


def main():
    global WORK
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765); ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--work", default=None)
    a = ap.parse_args()
    if a.work:
        WORK = Path(a.work)
    WORK.mkdir(parents=True, exist_ok=True)
    Handler.jobs = Jobs(WORK)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    url = f"http://127.0.0.1:{a.port}/"
    print("splat2mesh app:", url, flush=True)
    if not a.no_browser:
        threading.Timer(0.8, lambda: _open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


def _open(url: str):
    """Its own window where possible (Edge or Chrome in app mode), else the default browser."""
    if WINDOWS:
        for exe in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
                    r"C:\Program Files\Google\Chrome\Application\chrome.exe"):
            if Path(exe).exists():
                subprocess.Popen([exe, f"--app={url}", "--window-size=1400,900"]); return
    webbrowser.open(url)


if __name__ == "__main__":
    main()
