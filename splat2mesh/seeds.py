"""Scores one Idol sample from four sides (the Python version of scripts/seed_views.sh, also for Windows): the raw
splat rendered by athenea from Idol's own predicted camera and 90°, 180° and 270° around it, then the DeepSeek
validator (scripts/mv_check.py) against the photo.

    python -m splat2mesh.seeds SPLAT.ply CAMERA.json PHOTO OUT_DIR      → OUT_DIR.check.json (score, …)
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

from .athenea import render_splats
from .config import ROOT
from .ply import read_vertex

SIDES = {"front": 0.0, "left": 90.0, "back": 180.0, "right": 270.0}


def views(ply, cam_json):
    """[(name, eye, target, up)] in the USD frame of a Y-down splat turned 180° about X (the pipeline's)."""
    v = read_vertex(ply)
    xyz = np.stack([v["x"], v["y"], v["z"]], 1) * np.array([1, -1, -1])
    op = 1 / (1 + np.exp(-np.asarray(v["opacity"])))
    p = xyz[op > 0.05]; lo, hi = np.percentile(p, 0.5, 0), np.percentile(p, 99.5, 0)
    C = (lo + hi) / 2; r = float(np.linalg.norm(hi - lo) / 2)
    D = r / math.sin(math.atan(24.576 / 2 / 50.0)) * 1.08
    c = json.loads(Path(cam_json).read_text())
    out = []
    for side, dy in SIDES.items():
        y, pt = c["yaw"] + math.radians(dy), c["pitch"]
        d = np.array([math.cos(pt) * math.sin(y), math.sin(pt), -math.cos(pt) * math.cos(y)])   # Idol → USD
        out.append((side, (C + D * d).tolist(), C.tolist(), [0, 1, 0]))
    return out


def score(ply, cam_json, photo, out_dir) -> dict:
    out = Path(out_dir)
    render_splats(ply, views(ply, cam_json), out, rotate_x180=True, background=(1.0, 1.0, 1.0),
                  log=out.with_suffix(".log"))
    p = out / "views"
    js = out.with_name(out.name + ".check.json")
    subprocess.run([sys.executable, str(ROOT / "scripts/mv_check.py"), str(photo),
                    *[f"{int(dy)}={p / (s + '.png')}" for s, dy in SIDES.items()], "--json", str(js)],
                   check=True, stdout=subprocess.DEVNULL)
    return json.loads(js.read_text())


if __name__ == "__main__":
    a = sys.argv[1:]
    r = score(*a[:4])
    print(json.dumps({"score": r["score"], "coherent": r["coherent"], "worst": r["worst"][0]}, ensure_ascii=False))
