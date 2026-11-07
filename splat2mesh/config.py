"""Where splat2mesh finds the tools around it: athenea (renderer), the GPU hosts for Idol and ENEAS, Iris-3B, the
DeepSeek API and a notifier. One place, so scripts carry no machine-specific paths.

Sources, later ones win:
  1. defaults (everything local; athenea from third_party/athenea or $PATH)
  2. ~/.config/splat2mesh/config.toml
  3. ./splat2mesh.toml in the repository root (git-ignored; see splat2mesh.example.toml)
  4. environment variables (SPLAT2MESH_* and the few legacy names listed in ENV)

    from splat2mesh.config import cfg
    cfg("idol.host")                  # "local" or an ssh host
    python -m splat2mesh.config       # print the resolved configuration (secrets hidden)
"""
from __future__ import annotations

import os
import shutil
import tomllib
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = os.name == "nt"
TOOLS = ROOT / "tools"          # where the Windows installer puts athenea, uv and the model checkouts


def venv_python(venv) -> str:
    """The Python of a virtual environment, on Windows (Scripts/python.exe) and elsewhere (bin/python)."""
    v = Path(os.path.expanduser(str(venv)))
    return str(v / ("Scripts/python.exe" if WINDOWS else "bin/python"))

DEFAULTS = {
    "athenea": {
        "bin": "",               # athenea executable; empty: third_party/athenea build, then $PATH
        "shaders": "",           # ATHENEA_SHADER_DIR; empty: <build>/shaders next to the binary's bin/
    },
    "gpu": {"lock": ""},         # command prefix that serialises GPU jobs on this machine (e.g. a lock script)
    "idol": {
        "host": "local",         # "local" or an ssh host with CUDA
        "os": "linux",           # remote host OS: "linux" or "windows"
        "sdk": "~/idol-sdk",     # Idol SDK checkout on that host
        "python": "",            # its Python; empty: <sdk>/.venv/bin/python (linux) or python (windows)
        "work": "~/splat2mesh_idol",   # scratch folder on that host
        "weights": "",           # Idol's weights: a Hugging Face repo or a local folder; empty: speridlabs/idol
    },
    "eneas": {
        "host": "local",
        "os": "linux",           # "linux", "windows" (native, this PC) or "wsl" (a Windows host, ENEAS in WSL)
        "python": "",            # ENEAS's Python for "windows"; empty: <dir>/.venv
        "dir": "~/eneas",        # ENEAS checkout (inside WSL for "wsl")
        "work": "~/splat2mesh_eneas/jobs",
        "distro": "Ubuntu-24.04",
    },
    "iris": {"dir": str(ROOT / "third_party/iris-3b"), "python": "",
             "mode": "gpu"},           # gpu | cpu-encoder (8 GB GPUs) | space (the online Space)
    "deepseek": {
        "url": "https://api.deepseek.com/chat/completions",
        "model": "deepseek-flash",
        "env_file": "",          # file with a DEEPSEEK_API_KEY=... line, used when the variable is not set
    },
    "notify": {"command": ""},   # called as: COMMAND [-f IMAGE] TEXT
    "render": {"hdri": str(ROOT / "assets/hdri/brown_photostudio_02.hdr")},   # studio light for stills/turntables
}

ENV = {   # environment variable → key
    "ATHENEA_BIN": "athenea.bin", "ATHENEA_SHADER_DIR": "athenea.shaders",
    "SPLAT2MESH_GPU_LOCK": "gpu.lock",
    "SPLAT2MESH_IDOL_HOST": "idol.host", "SPLAT2MESH_IDOL_OS": "idol.os", "SPLAT2MESH_IDOL_SDK": "idol.sdk",
    "SPLAT2MESH_IDOL_PYTHON": "idol.python", "SPLAT2MESH_IDOL_WORK": "idol.work",
    "SPLAT2MESH_ENEAS_HOST": "eneas.host", "SPLAT2MESH_ENEAS_OS": "eneas.os", "SPLAT2MESH_ENEAS_DIR": "eneas.dir",
    "SPLAT2MESH_IRIS_DIR": "iris.dir", "SPLAT2MESH_IRIS_PYTHON": "iris.python",
    "DEEPSEEK_ENV": "deepseek.env_file", "SPLAT2MESH_DEEPSEEK_MODEL": "deepseek.model",
    "SPLAT2MESH_NOTIFY": "notify.command", "SPLAT2MESH_HDRI": "render.hdri",
}


def _merge(a, b):
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _merge(a[k], v)
        else:
            a[k] = v
    return a


@lru_cache(maxsize=1)
def load() -> dict:
    import copy
    c = copy.deepcopy(DEFAULTS)
    user = [Path.home() / ".config/splat2mesh/config.toml"]
    if os.environ.get("APPDATA"):                       # Windows: what the installer and the app write
        user.append(Path(os.environ["APPDATA"]) / "splat2mesh" / "config.toml")
    for f in (*user, ROOT / "splat2mesh.toml"):
        if f.exists():
            _merge(c, tomllib.loads(f.read_text()))
    for var, key in ENV.items():
        if os.environ.get(var):
            sec, name = key.split(".")
            c[sec][name] = os.environ[var]
    # legacy: ATHENEA_BUILD=<build dir>
    if not c["athenea"]["bin"] and os.environ.get("ATHENEA_BUILD"):
        c["athenea"]["bin"] = str(Path(os.environ["ATHENEA_BUILD"]) / "bin/athenea")
    return c


def cfg(key: str):
    sec, name = key.split(".")
    return load()[sec][name]


def athenea_bin() -> Path | None:
    """The athenea executable: configured, else the third_party/athenea build, else $PATH."""
    b = cfg("athenea.bin")
    cands = [Path(os.path.expanduser(b))] if b else []
    cands += sorted((ROOT / "third_party/athenea/build").glob("*/bin/athenea" + (".exe" if WINDOWS else "")))
    cands += sorted(TOOLS.glob("athenea*/**/athenea.exe")) if WINDOWS else []
    w = shutil.which("athenea")
    if w:
        cands.append(Path(w))
    return next((c for c in cands if c.exists()), None)


def athenea_env() -> dict:
    """Environment for running athenea (ATHENEA_SHADER_DIR next to the build when not configured)."""
    env = dict(os.environ)
    sh = cfg("athenea.shaders")
    b = athenea_bin()
    if not sh and b is not None:
        for d in (b.parent.parent / "shaders", b.parent / "shaders"):     # a build tree, or a release folder
            if d.exists():
                sh = str(d); break
    if sh:
        env["ATHENEA_SHADER_DIR"] = os.path.expanduser(sh)
    return env


def gpu_prefix() -> list[str]:
    """['bash', LOCK] when a GPU lock is configured, else [] (the job runs directly)."""
    lock = cfg("gpu.lock")
    return ["bash", os.path.expanduser(lock)] if lock else []


def iris_python() -> str:
    p = cfg("iris.python")
    return os.path.expanduser(p) if p else venv_python(Path(cfg("iris.dir")).expanduser() / ".venv")


class gpu_lock:
    """One GPU job at a time on this machine, between processes (a lock file in the user's cache). Used where no
    lock script is configured (Windows); re-entrant within a process."""
    _depth = 0

    def __init__(self, timeout: float = 6 * 3600):
        self.timeout = timeout
        self.path = Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".cache") / "splat2mesh" / "gpu.lock"

    def __enter__(self):
        import time
        if gpu_lock._depth == 0:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            while True:
                try:
                    fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.write(fd, str(os.getpid()).encode()); os.close(fd); break
                except FileExistsError:
                    try:            # a lock left by a process that is gone, or older than the timeout
                        pid = int(self.path.read_text() or 0)
                        stale = time.time() - self.path.stat().st_mtime > self.timeout or not _alive(pid)
                    except (OSError, ValueError):
                        stale = False
                    if stale:
                        self.path.unlink(missing_ok=True)
                    elif time.time() - t0 > self.timeout:
                        raise TimeoutError("GPU lock " + str(self.path))
                    else:
                        time.sleep(2)
        gpu_lock._depth += 1
        return self

    def __exit__(self, *a):
        gpu_lock._depth -= 1
        if gpu_lock._depth == 0:
            self.path.unlink(missing_ok=True)


def _alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if WINDOWS:
        import ctypes
        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)     # PROCESS_QUERY_LIMITED_INFORMATION
        if h:
            ctypes.windll.kernel32.CloseHandle(h); return True
        return False
    try:
        os.kill(pid, 0); return True
    except OSError:
        return False


def deepseek_key() -> str | None:
    k = os.environ.get("DEEPSEEK_API_KEY")
    if k:
        return k
    f = cfg("deepseek.env_file")
    if f and Path(os.path.expanduser(f)).exists():
        for line in Path(os.path.expanduser(f)).read_text().splitlines():
            name, _, val = line.partition("=")
            if name.strip() == "DEEPSEEK_API_KEY" and val.strip():
                return val.strip().strip("\"'")
    return None


if __name__ == "__main__":
    import json
    c = load()
    print(json.dumps(c, indent=1))
    print("athenea binary:", athenea_bin())
    print("DeepSeek key:", "set" if deepseek_key() else "missing")
