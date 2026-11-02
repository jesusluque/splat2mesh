"""Multi-view dataset: RGB + depth + cameras.

Expected layout (one folder per asset):
    views/cameras.json   {"frames": [{"file": "000.png", "depth": "000.exr",
                                       "K": [[fx,0,cx],[0,fy,cy],[0,0,1]],
                                       "w2c": 4x4 (OpenCV: +x right, +y down, +z forward),
                                       "width": W, "height": H}, ...],
                          "depth_kind": "z" | "range"}
    views/*.png / *.exr

Depth is metric z (or ray range) in the stage units; 0 or inf = background.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class View:
    rgb: np.ndarray        # HxWx3 float32 linear-ish [0,1] (sRGB as stored)
    alpha: np.ndarray | None
    depth: np.ndarray      # HxW float32 z-depth, 0 = invalid
    K: np.ndarray          # 3x3
    w2c: np.ndarray        # 4x4 OpenCV convention

    @property
    def c2w(self) -> np.ndarray:
        return np.linalg.inv(self.w2c)

    @property
    def center(self) -> np.ndarray:
        return self.c2w[:3, 3]


def _read_depth(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".exr":
        import OpenEXR  # optional dependency
        import Imath
        f = OpenEXR.InputFile(str(path))
        dw = f.header()["dataWindow"]
        w, h = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
        chans = f.header()["channels"].keys()
        ch = "Z" if "Z" in chans else ("R" if "R" in chans else next(iter(chans)))
        buf = f.channel(ch, Imath.PixelType(Imath.PixelType.FLOAT))
        return np.frombuffer(buf, dtype=np.float32).reshape(h, w).copy()
    if path.suffix.lower() == ".npy":
        return np.load(path).astype(np.float32)
    raise ValueError(f"unsupported depth format: {path}")


def _range_to_z(rng: np.ndarray, K: np.ndarray) -> np.ndarray:
    h, w = rng.shape
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    x = (u - K[0, 2]) / K[0, 0]
    y = (v - K[1, 2]) / K[1, 1]
    return rng / np.sqrt(1.0 + x * x + y * y)


def load_views(folder: str | Path) -> list[View]:
    folder = Path(folder)
    meta = json.loads((folder / "cameras.json").read_text())
    kind = meta.get("depth_kind", "z")
    views = []
    for fr in meta["frames"]:
        img = np.asarray(Image.open(folder / fr["file"]), dtype=np.float32) / 255.0
        alpha = img[..., 3] if img.shape[-1] == 4 else None
        K = np.asarray(fr["K"], dtype=np.float64)
        d = _read_depth(folder / fr["depth"])
        d = np.where(np.isfinite(d), d, 0.0).astype(np.float32)
        if kind == "range":
            d = _range_to_z(d, K).astype(np.float32)
        if alpha is not None:
            d[alpha < 0.5] = 0.0
        views.append(View(img[..., :3], alpha, d, K, np.asarray(fr["w2c"], dtype=np.float64)))
    return views


def load_athenea(folder: str | Path, erode_px: int = 1) -> list[View]:
    """athenea's export: cameras.json {intrinsics, views[{name, R_world_to_cam, t_world_to_cam}]},
    views/vNNN.png (sRGB), vNNN.exr (Z = camera-axis depth, 0 = background), vNNN_mask.png."""
    import OpenEXR
    from scipy.ndimage import binary_erosion
    folder = Path(folder)
    meta = json.loads((folder / "cameras.json").read_text())
    it = meta["intrinsics"]
    fx = it.get("fx", it.get("fx_px")); fy = it.get("fy", it.get("fy_px", fx))
    cx = it.get("cx", it.get("cx_px")); cy = it.get("cy", it.get("cy_px", cx))
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    views = []
    for fr in meta["views"]:
        n = fr["name"]
        rgb = np.asarray(Image.open(folder / "views" / f"{n}.png").convert("RGB"), np.float32) / 255.0
        mask = np.asarray(Image.open(folder / "views" / f"{n}_mask.png").convert("L")) > 127
        npy = folder / "views" / f"{n}_depth.npy"
        if npy.exists():
            z = np.load(npy).astype(np.float32)
        else:
            z = OpenEXR.File(str(folder / "views" / f"{n}.exr")).channels()["Z"].pixels.astype(np.float32)
        if erode_px:
            mask = binary_erosion(mask, iterations=erode_px)   # silhouette pixels mix sky colour
        z = np.where(mask & np.isfinite(z), z, 0.0).astype(np.float32)
        w2c = np.eye(4)
        w2c[:3, :3] = np.asarray(fr["R_world_to_cam"])
        w2c[:3, 3] = np.asarray(fr["t_world_to_cam"])
        views.append(View(rgb, mask.astype(np.float32), z, K, w2c))
    return views
