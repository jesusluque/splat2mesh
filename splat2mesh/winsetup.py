"""Installs and checks everything splat2mesh runs on one Windows PC with an NVIDIA GPU, all local.

Used by the installer wizard (installer/splat2mesh.iss) and by the app's Components page; also by hand:

    python -m splat2mesh.winsetup status                    # JSON: what is installed, the GPU, the options
    python -m splat2mesh.winsetup install athenea idol eneas iris [--hf-token-env HF_TOKEN] [--vae klein|dev]
                                          [--iris-mode cpu-encoder|gpu|space]
    python -m splat2mesh.winsetup options --deepseek-key-env DEEPSEEK_API_KEY --iris-mode cpu-encoder

Progress goes to stdout as lines the callers parse:   PROGRESS <0-100> <component>: <message>
Every component installs under <splat2mesh>/tools, natively (no WSL). Hugging Face tokens are used for the downloads only (passed through the environment, never
written); afterwards the models run offline from the Hugging Face cache.

Components
  athenea   the renderer: the published Windows release (zip: exe, DLLs, shaders)
  idol      picture → splat: the Idol SDK, PyTorch (CUDA 12.8), its weights; DINOv3 is gated on Hugging Face
            (request access first); the FLUX.2 VAE is taken from FLUX.2-klein-4B (Apache-2.0, identical to the
            non-commercial FLUX.2-dev one in bf16, docs/licenses.md) unless --vae dev
  eneas     part masks: the ENEAS checkout in its own Python environment (without deepspeed, which it does not
            need for inference); its weights (~9 GB) download on the first segmentation
  iris      text → picture: Iris-3B Turbo, its weights and Qwen3-VL-4B; on an 8 GB GPU the text encoder runs on
            the CPU (--iris-mode cpu-encoder), or the online Space is used instead (space)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

from .config import ROOT, TOOLS, WINDOWS, venv_python

ATHENEA_URL = "https://github.com/jesusluque/athenea/releases/download/v0.3.0/athenea-0.3.0-windows-x64.zip"
ATHENEA_SHA256 = "a281d3537637ec97e7a4af94c14a4b962429a573130717bf1cd3de224e82dfce"
UV_URL = "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip"
IDOL_GIT = "https://github.com/speridlabs/idol-sdk.git"
ENEAS_GIT = "https://github.com/speridlabs/eneas.git"
IRIS_GIT = "https://github.com/speridlabs/iris-3b.git"
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
DINOV3 = "facebook/dinov3-vith16plus-pretrain-lvd1689m"
VAE = {"klein": "black-forest-labs/FLUX.2-klein-4B", "dev": "black-forest-labs/FLUX.2-dev"}
USER_CONFIG = Path(os.environ.get("APPDATA") or Path.home() / ".config") / "splat2mesh" / "config.toml"
COMPONENTS = ("athenea", "idol", "eneas", "iris")


def progress(pct: float, comp: str, msg: str):
    print(f"PROGRESS {int(max(0, min(100, pct)))} {comp}: {msg}", flush=True)


def run(cmd, comp: str, env=None, cwd=None, check=True) -> str:
    """Runs a command, echoing its output (the app shows it as the log)."""
    print("$ " + " ".join(map(str, cmd)), flush=True)
    p = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         env={**os.environ, **(env or {})}, cwd=cwd, encoding="utf-8", errors="replace")
    out = []
    for line in p.stdout:
        out.append(line); print(line.rstrip(), flush=True)
    if p.wait() and check:
        raise RuntimeError(f"{comp}: {Path(str(cmd[0])).name} failed ({p.returncode})")
    return "".join(out)


def download(url: str, dst: Path, comp: str, lo: float, hi: float, sha256: str | None = None,
             attempts: int = 6) -> Path:
    """Downloads with retries; a broken transfer resumes where it stopped (HTTP Range)."""
    import time
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and (not sha256 or _sha(dst) == sha256):
        progress(hi, comp, f"{dst.name} already downloaded"); return dst
    tmp = dst.with_suffix(dst.suffix + ".part")
    for k in range(attempts):
        try:
            have = tmp.stat().st_size if tmp.exists() else 0
            req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
            with urllib.request.urlopen(req, timeout=60) as r:
                if have and r.status != 206:                 # the server ignored the range: start again
                    have = 0
                total = have + int(r.headers.get("Content-Length") or 0); got = have; last = -1
                with open(tmp, "ab" if have else "wb") as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk); got += len(chunk)
                        if total:
                            pct = lo + (hi - lo) * got / total
                            if int(pct) != last:
                                last = int(pct); progress(pct, comp, f"downloading {dst.name} {got >> 20}/{total >> 20} MB")
            break
        except Exception as e:
            if k == attempts - 1:
                raise RuntimeError(f"{comp}: download of {dst.name} failed: {e}")
            progress(lo, comp, f"download interrupted ({str(e)[:80]}), retrying in {5 * (k + 1)} s")
            time.sleep(5 * (k + 1))
    if sha256 and _sha(tmp) != sha256:
        tmp.unlink(); raise RuntimeError(f"{comp}: {dst.name} checksum mismatch")
    tmp.replace(dst)
    return dst


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def uv() -> str:
    """uv: the installer's own copy under tools/uv, else one on PATH (downloaded when neither)."""
    own = TOOLS / "uv" / ("uv.exe" if WINDOWS else "uv")
    if own.exists():
        return str(own)
    w = shutil.which("uv")
    if w:
        return w
    z = download(UV_URL, TOOLS / "uv" / "uv.zip", "uv", 0, 100)
    zipfile.ZipFile(z).extractall(own.parent)
    return str(own)


def get_repo(url: str, dst: Path, comp: str, lo: float = 0, hi: float = 5):
    """A GitHub repository's main branch: git when available and already a checkout, else its zip (no git
    needed on a clean Windows)."""
    if (dst / ".git").exists() and shutil.which("git"):
        run(["git", "-C", dst, "pull", "--ff-only"], comp, check=False); return
    if (dst / "pyproject.toml").exists():
        progress(hi, comp, f"{dst.name} already here"); return
    z = download(url.removesuffix(".git") + "/archive/refs/heads/main.zip", TOOLS / "downloads" / f"{dst.name}.zip",
                 comp, lo, hi)
    tmp = dst.with_name(dst.name + ".unpack"); shutil.rmtree(tmp, ignore_errors=True)
    zipfile.ZipFile(z).extractall(tmp)
    inner = next(tmp.iterdir())                       # <repo>-main/
    if dst.exists():
        for item in inner.iterdir():                  # keep an existing .venv / weights
            t = dst / item.name
            shutil.rmtree(t, ignore_errors=True) if t.is_dir() else t.unlink(missing_ok=True)
            shutil.move(str(item), str(t))
    else:
        shutil.move(str(inner), str(dst))
    shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------------------------------------------------- status
def gpu() -> dict:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
        name, mem, drv = [x.strip() for x in out.splitlines()[0].split(",")]
        return {"name": name, "vram_gb": round(int(mem) / 1024, 1), "driver": drv}
    except Exception:
        return {"name": None, "vram_gb": 0, "driver": None}


def status() -> dict:
    from .config import athenea_bin, deepseek_key, cfg
    idol = TOOLS / "idol-sdk"
    iris = TOOLS / "iris-3b"
    s = {
        "gpu": gpu(),
        "athenea": {"ok": athenea_bin() is not None, "path": str(athenea_bin() or "")},
        "idol": {"ok": Path(venv_python(idol / ".venv")).exists() and (idol / "weights" / "config.json").exists(),
                 "path": str(idol)},
        "eneas": {"ok": Path(venv_python(TOOLS / "eneas" / ".venv")).exists(), "path": str(TOOLS / "eneas")},
        "iris": {"ok": Path(venv_python(iris / ".venv")).exists() and (iris / "weights" / "turbo").exists(),
                 "mode": cfg("iris.mode") if "mode" in _cfg_section("iris") else "cpu-encoder"},
        "deepseek": {"ok": bool(deepseek_key())},
    }
    return s


def _cfg_section(sec: str) -> dict:
    from .config import load
    return load().get(sec, {})


# ------------------------------------------------------------------------------------------------- config
def write_user_config(updates: dict):
    """Merges {section: {key: value}} into the user's config file (%APPDATA%/splat2mesh/config.toml)."""
    import tomllib
    c = tomllib.loads(USER_CONFIG.read_text()) if USER_CONFIG.exists() else {}
    for sec, kv in updates.items():
        c.setdefault(sec, {}).update(kv)
    lines = []
    for sec, kv in c.items():
        lines.append(f"[{sec}]")
        for k, v in kv.items():
            lines.append(f"{k} = {json.dumps(v)}")
        lines.append("")
    USER_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    USER_CONFIG.write_text("\n".join(lines))


# ------------------------------------------------------------------------------------------------- components
def install_athenea():
    c = "athenea"
    z = download(ATHENEA_URL, TOOLS / "downloads" / Path(ATHENEA_URL).name, c, 0, 85, ATHENEA_SHA256)
    dst = TOOLS / "athenea"
    progress(88, c, "unpacking")
    shutil.rmtree(dst, ignore_errors=True)
    zipfile.ZipFile(z).extractall(dst)
    exe = next(dst.rglob("athenea.exe"), None)
    if exe is None:
        raise RuntimeError("athenea: athenea.exe not found in the release")
    progress(95, c, "checking")
    run([exe, "info"], c, check=False)
    write_user_config({"athenea": {"bin": str(exe)}})
    progress(100, c, f"ready: {exe}")


def install_idol(vae: str = "klein"):
    c = "idol"
    d = TOOLS / "idol-sdk"
    progress(2, c, "getting the Idol SDK")
    get_repo(IDOL_GIT, d, c, 2, 7)
    progress(8, c, "Python 3.12 environment")
    run([uv(), "venv", "--python", "3.12", d / ".venv"], c, check=False)
    py = venv_python(d / ".venv")
    progress(12, c, "PyTorch with CUDA 12.8 (~3 GB)")
    run([uv(), "pip", "install", "--python", py, "torch", "torchvision", "--index-url", TORCH_INDEX], c)
    progress(55, c, "Idol and its dependencies")
    run([uv(), "pip", "install", "--python", py, "-e", d], c)
    run([uv(), "pip", "install", "--python", py, "huggingface_hub[hf_xet]"], c, check=False)
    # weights: Idol's own (public), DINOv3 (gated) and the VAE; the token, if any, only through the environment
    progress(62, c, "weights: Idol, DINOv3, FLUX.2 VAE")
    w = d / "weights"
    script = f"""
import json, os
from huggingface_hub import snapshot_download, hf_hub_download
w = snapshot_download("speridlabs/idol", local_dir=r"{w}")
cfg = json.load(open(os.path.join(w, "config.json")))
cfg["image_encoders"]["flux2_vae"] = "{VAE[vae]}"
json.dump(cfg, open(os.path.join(w, "config.json"), "w"), indent=1)
snapshot_download("{DINOV3}")
snapshot_download("{VAE[vae]}", allow_patterns=["vae/*"])
print("weights ok")
"""
    run([py, "-c", script], c)
    progress(92, c, "checking CUDA")
    out = run([py, "-c", "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"], c)
    if "cuda True" not in out:
        raise RuntimeError("idol: PyTorch does not see the GPU (update the NVIDIA driver)")
    write_user_config({"idol": {"host": "local", "os": "windows", "sdk": str(d), "python": py,
                                "work": str(ROOT / "work" / "idol"), "weights": str(w)}})
    progress(100, c, "ready")


ENEAS_DEPS = [   # ENEAS's dependencies on native Windows: all but deepspeed (training only; it does not build there)
    "transformers>=4.53.0,<4.54.0", "numpy>=1.24.0,<2.0.0", "opencv-python>=4.8.0", "pillow>=9.0.0", "tqdm>=4.65.0",
    "hydra-core>=1.3.0,<2.0.0", "iopath>=0.1.10", "timm>=0.9.0,<2.0.0", "einops>=0.6.0", "accelerate>=0.20.0",
    "peft>=0.5.0", "typer>=0.9.0", "click>=8.1.0", "spacy==3.8.11", "thinc==8.3.10", "inflect>=6.0.0",
    "sentencepiece>=0.2.0", "protobuf>=3.20.0", "sam2", "ollama", "pydantic", "decord", "huggingface_hub[hf_xet]",
    "en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/"
    "en_core_web_sm-3.8.0-py3-none-any.whl",
]


def install_eneas():
    """ENEAS natively on Windows (no WSL): its own Python 3.12 environment under tools/eneas."""
    c = "eneas"
    d = TOOLS / "eneas"
    progress(2, c, "getting ENEAS")
    get_repo(ENEAS_GIT, d, c, 2, 5)
    progress(6, c, "Python 3.12 environment")
    run([uv(), "venv", "--python", "3.12", d / ".venv"], c, check=False)
    py = venv_python(d / ".venv")
    progress(10, c, "PyTorch with CUDA 12.8 (~3 GB)")
    run([uv(), "pip", "install", "--python", py, "torch", "torchvision", "--index-url", TORCH_INDEX], c)
    progress(50, c, "ENEAS dependencies (without deepspeed; SAM 2 without its CUDA extension)")
    run([uv(), "pip", "install", "--python", py, *ENEAS_DEPS], c, env={"SAM2_BUILD_CUDA": "0"})
    run([uv(), "pip", "install", "--python", py, "--no-deps", "-e", d], c)
    progress(80, c, "checking")
    out = run([py, "-c", "import torch, eneas, sam2, decord; print('cuda', torch.cuda.is_available())"], c)
    if "cuda True" not in out:
        raise RuntimeError("eneas: PyTorch does not see the GPU (update the NVIDIA driver)")
    write_user_config({"eneas": {"host": "local", "os": "windows", "dir": str(d), "python": py,
                                 "work": str(ROOT / "work" / "eneas")}})
    progress(100, c, "ready (its weights, ~9 GB, download on the first segmentation)")


def install_iris(mode: str = "cpu-encoder"):
    c = "iris"
    if mode == "space":
        write_user_config({"iris": {"mode": "space"}})
        progress(100, c, "using the online Iris-3B Turbo Space (nothing to download)"); return
    d = TOOLS / "iris-3b"
    progress(2, c, "getting Iris-3B")
    get_repo(IRIS_GIT, d, c, 2, 8)
    run([uv(), "venv", "--python", "3.12", d / ".venv"], c, check=False)
    py = venv_python(d / ".venv")
    progress(10, c, "PyTorch with CUDA 12.8")
    run([uv(), "pip", "install", "--python", py, "torch", "torchvision", "--index-url", TORCH_INDEX], c)
    progress(45, c, "Iris and its dependencies")
    run([uv(), "pip", "install", "--python", py, "-e", d], c)
    progress(55, c, "weights: Iris-3B Turbo and Qwen3-VL-4B (~14 GB)")
    script = f"""
from huggingface_hub import snapshot_download
snapshot_download("speridlabs/iris-3b", allow_patterns=["turbo/*"], local_dir=r"{d / 'weights'}")
snapshot_download("Qwen/Qwen3-VL-4B-Instruct")
print("weights ok")
"""
    run([py, "-c", script], c)
    write_user_config({"iris": {"dir": str(d), "python": py, "mode": mode}})
    progress(100, c, f"ready ({mode})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("status")
    i = sp.add_parser("install"); i.add_argument("components", nargs="+", choices=COMPONENTS + ("all",))
    i.add_argument("--vae", default="klein", choices=list(VAE))
    i.add_argument("--iris-mode", default="cpu-encoder", choices=["cpu-encoder", "gpu", "space"])
    i.add_argument("--hf-token-file", help="a file with a Hugging Face token: read, deleted, used for the downloads")
    o = sp.add_parser("options")
    o.add_argument("--deepseek-key-env", help="name of an environment variable holding the DeepSeek key")
    o.add_argument("--iris-mode", choices=["cpu-encoder", "gpu", "space"])
    o.add_argument("--deepseek-key-file", help="a file with the DeepSeek key (read and deleted)")
    a = ap.parse_args()
    if a.cmd == "status":
        print(json.dumps(status(), indent=1)); return
    if a.cmd == "options":
        up = {}
        if a.deepseek_key_file and Path(a.deepseek_key_file).exists():
            os.environ["_S2M_DS"] = Path(a.deepseek_key_file).read_text().strip()
            Path(a.deepseek_key_file).unlink(); a.deepseek_key_env = "_S2M_DS"
        if a.deepseek_key_env and os.environ.get(a.deepseek_key_env):
            # the user's own key, in the user's own config folder (the DeepSeek checks need it at run time)
            f = USER_CONFIG.parent / "deepseek.env"; f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(f"DEEPSEEK_API_KEY={os.environ[a.deepseek_key_env]}\n")
            up["deepseek"] = {"env_file": str(f)}
        if a.iris_mode:
            up["iris"] = {"mode": a.iris_mode}
        if up:
            write_user_config(up)
        print("options saved"); return
    if a.hf_token_file and Path(a.hf_token_file).exists():        # this process and its children only
        tok = Path(a.hf_token_file).read_text().strip(); Path(a.hf_token_file).unlink()
        if tok:
            os.environ["HF_TOKEN"] = tok
    comps = COMPONENTS if "all" in a.components else a.components
    failed = []
    for comp in comps:
        try:
            {"athenea": install_athenea, "idol": lambda: install_idol(a.vae), "eneas": install_eneas,
             "iris": lambda: install_iris(a.iris_mode)}[comp]()
        except Exception as e:
            print(f"FAILED {comp}: {e}", flush=True); failed.append(comp)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
