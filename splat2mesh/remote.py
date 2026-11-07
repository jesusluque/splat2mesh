"""Idol and ENEAS where they are installed: on this machine, on a Linux host over ssh, or on a Windows host (Idol
natively, ENEAS inside WSL). Hosts and paths come from splat2mesh.config ([idol], [eneas]).

    idol_sample(image, work_dir, gaussians=262144, seed=42)  →  work_dir/<stem>.ply, idol_input.png, idol_camera.json
    Eneas()  →  .upload(files), .run(frames_dir, out_npz, ann_frames, phrases), .download(name, local)
"""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path

from .config import cfg


def _run(cmd, **kw) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed ({r.returncode}):\n{(r.stdout + r.stderr)[-2000:]}")
    return r.stdout


def _x(p: str) -> str:
    return os.path.expanduser(p)


# ------------------------------------------------------------------------------------------------------- Idol
def idol_mode(image: Path) -> str:
    """Idol's crop mode: photos (.jpg) 'real' (margin 1.7), renders and generated pictures 'synthetic' (1.2)."""
    return "real" if Path(image).suffix.lower() in {".jpg", ".jpeg"} else "synthetic"


def idol_sample(image, work, gaussians: int = 262144, seed: int = 42) -> Path:
    """Picture → 3DGS .ply with Idol; also fetches Idol's prepared input and its estimate of the input camera."""
    image, work = Path(image), Path(work); work.mkdir(parents=True, exist_ok=True)
    host, osname, sdk = cfg("idol.host"), cfg("idol.os"), cfg("idol.sdk")
    args = f"--mode {idol_mode(image)} --num_gaussians {gaussians} --seed {seed}"
    stem = image.stem
    if host == "local":
        py = cfg("idol.python") or str(Path(_x(sdk)) / ".venv/bin/python")
        out = work / "idol"; out.mkdir(exist_ok=True)
        _run([py, "-m", "idol.sample", "--image", str(image), "--out", str(out)] + args.split(),
             cwd=_x(sdk), env={**os.environ, "ATTN_BACKEND": "sdpa"})
        get = lambda name, dst: shutil.copy(out / name, dst) if (out / name).exists() else None
    elif osname == "windows":
        d = cfg("idol.work"); py = cfg("idol.python") or "python"
        _run(["ssh", host, f'mkdir "{d}" 2>nul & exit 0'])
        _run(["scp", "-q", str(image), f"{host}:{d}/{image.name}"])
        # eager mode on Windows: torch.compile is not available there
        _run(["ssh", host, rf'cd /d {sdk} && set ATTN_BACKEND=sdpa&& {py} -m idol.sample --image {d}/{image.name} '
                           rf'--out {d} --no_optimize {args}'])
        get = lambda name, dst: subprocess.run(["scp", "-q", f"{host}:{d}/{name}", str(dst)])
    else:
        d = cfg("idol.work"); py = cfg("idol.python") or f"{sdk}/.venv/bin/python"
        _run(["ssh", host, f"mkdir -p {d}"])
        _run(["scp", "-q", str(image), f"{host}:{d}/{image.name}"])
        _run(["ssh", host, f"cd {sdk} && ATTN_BACKEND=sdpa {py} -m idol.sample --image {d}/{image.name} --out {d} {args}"])
        get = lambda name, dst: subprocess.run(["scp", "-q", f"{host}:{d}/{name}", str(dst)])
    ply = work / f"{stem}.ply"
    get(f"{stem}.ply", ply); get(f"{stem}_input.png", work / "idol_input.png")
    get(f"{stem}_camera.json", work / "idol_camera.json")
    if not ply.exists():
        raise RuntimeError(f"Idol wrote no {stem}.ply")
    return ply


# ------------------------------------------------------------------------------------------------------- ENEAS
class Eneas:
    """One ENEAS job folder (frames in, masks.npz out) on the configured host."""

    def __init__(self, tag: str):
        self.host, self.os, self.dir = cfg("eneas.host"), cfg("eneas.os"), cfg("eneas.dir")
        work = cfg("eneas.work")
        if self.os == "wsl":                       # Windows path for scp, the same folder seen from WSL
            self.job = f"{work}/{tag}"
            self.inside = "/mnt/" + self.job[0].lower() + self.job[2:] if self.job[1] == ":" else self.job
        else:
            self.job = f"{work}/{tag}"; self.inside = self.job

    def _ssh(self, cmd: str, **kw) -> str:
        if self.host == "local":
            return _run(["bash", "-lc", cmd], **kw)
        return _run(["ssh", self.host, cmd], **kw)

    def reset(self):
        if self.os == "wsl":
            w = self.job.replace("/", "\\")
            self._ssh(f'rmdir /s /q "{w}" 2>nul & mkdir "{w}\\frames"')
        else:
            self._ssh(f"rm -rf {self.job}; mkdir -p {self.job}/frames")

    def upload(self, files, sub: str = ""):
        dst = f"{self.job}/{sub}".rstrip("/") + "/"
        if self.host == "local":
            for f in files:
                shutil.copy(f, _x(dst))
        else:
            _run(["scp", "-q", *map(str, files), f"{self.host}:{dst}"])

    def download(self, name: str, local):
        if self.host == "local":
            shutil.copy(_x(f"{self.job}/{name}"), local)
        else:
            _run(["scp", "-q", f"{self.host}:{self.job}/{name}", str(local)])

    def run(self, script_name: str, files: list[str], words: list[str], timeout: int = 5400) -> str:
        """Run a script already uploaded to the job folder, inside the ENEAS environment (uv). files: paths inside
        the job folder (not quoted, so ~ expands); words: quoted arguments (phrases)."""
        q = " ".join([f"{self.inside}/{f}" for f in files] + [shlex.quote(w) for w in words])
        inner = f"cd {self.dir} && ~/.local/bin/uv run python {self.inside}/{script_name} {q}"
        if self.os == "wsl":
            inner = inner.replace('"', '\\"')
            return self._ssh(f'wsl -d {cfg("eneas.distro")} -- bash -lc "{inner}"', timeout=timeout)
        return self._ssh(inner, timeout=timeout)
