"""An object as an assembly of components, with no geometry: like an IFC model, a list of components, each made on
its own and fitted in its place, all as gaussian splats.

    python scripts/gs_components.py REFERENCE.png OUT_DIR [--whole WHOLE_FLOW_DIR] [--step bom|whole|parts|place|all]

 1. bom: deepseek-flash reads the reference picture and writes the bill of components: each KIND of component once,
    with its material, how many identical copies (repetitions: buckles, pouches, tools) and whether the copies are
    mirror images (a left and a right boot), and which component is the base the others are mounted on (the body).
    OUT_DIR/bom.json.
 2. whole: the whole object as one splat with materials (scripts/splat_flow.py) whose part list is the BOM, so every
    gaussian of the whole votes for a component: that is where each component goes, in 3D, seen from all sides.
    Reused from --whole when it is given.
 3. parts: each component other than the base is cut out of the reference picture (its ENEAS mask; one copy for a
    repeated part), completed by Iris-3B (scripts/components.py: recompose + iris_picture) and made by Idol on its own:
    a splat with more gaussians on it than it had in the whole.
 4. place: the whole's gaussians of each component are split into its copies (3D clusters); each copy gets the
    component's splat by a similarity transform (rotations around Idol's camera, scale from the extents, then scaled
    ICP on the gaussian centres), mirrored for a mirror copy. The whole's gaussians of placed components are
    replaced by the components' own. OUT_DIR/assembly.ply (+ mat attributes), components.json (the list: component,
    material, copy, transform, mirrored, fit), assembly.usdc (athenea).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))

MAX_COMPONENTS = 16
BOM_PROMPT = (
    "This picture shows one object (a figure, a product, a piece of furniture). Write its bill of components for a 3D "
    "artist who builds it like an assembly: each KIND of separate physical component once (a piece of a different "
    "material, or a piece mounted on another: an accessory, a buckle, a strap, a tool, a boot), each large enough to be "
    "seen clearly. At most {n} lines, the base first: the component the others are mounted on (for a figure, the body "
    "with its skin; a figure's hair and its clothes are separate components). For each line: 'phrase', a short English "
    "noun phrase starting with 'the' that a segmentation model can find in the picture (plural for a repeated part, "
    "e.g. 'the corset buckles', so all visible copies are found); 'material', chosen ONLY from: {mats}; 'count', how "
    "many IDENTICAL copies the whole object has, counting hidden ones (a figure has 2 boots); 'mirrored', true when the "
    "copies are mirror images of each other (a left and a right boot or glove), false when they are the same piece "
    "repeated (three buckles); 'revolution', true for a body of revolution (a bottle, a flask, a wheel); 'standalone', true when the component is "
    "a piece that exists on its own and could be photographed alone on a table (a boot, a wrench, a buckle, a belt, "
    "goggles), false for a layer that only has its shape on the base (hair, a blouse, a skirt, paint). "
    "Answer JSON: {{\"bom\": [{{\"phrase\": ..., \"material\": ..., \"count\": n, \"mirrored\": true|false, "
    "\"revolution\": true|false, \"standalone\": true|false}}]}}")


def bom(ref: Path, out: Path) -> list:
    f = out / "bom.json"
    if f.exists():
        return json.loads(f.read_text())
    from splat2mesh.deepseek import ask_image
    mats = list(json.loads((ROOT / "splat2mesh/material_library.json").read_text())["materials"])
    j = ask_image(Image.open(ref).convert("RGB"), BOM_PROMPT.format(n=MAX_COMPONENTS, mats=", ".join(mats)))
    rows, seen = [], set()
    for p in (j.get("bom") or [])[:MAX_COMPONENTS]:
        ph = str(p.get("phrase", "")).strip().lower(); m = str(p.get("material", "")).strip().lower()
        if not ph or m not in mats:
            continue
        ph = (ph if ph.startswith("the ") else "the " + ph).replace("'", "").replace('"', "")   # as ENEAS gets it
        if ph in seen:
            continue
        seen.add(ph)
        n = int(p["count"]) if str(p.get("count", 1)).isdigit() else 1
        rows.append({"phrase": ph, "material": m, "count": max(n, 1), "mirrored": bool(p.get("mirrored")) and n > 1,
                     "revolution": p.get("revolution") is True, "standalone": p.get("standalone") is True and bool(rows),
                     "base": not rows})
    f.write_text(json.dumps(rows, indent=1))
    return rows


def description(rows) -> str:
    """The BOM as a materials description for splat_flow: 'material: phrase, phrase; ...' in BOM order."""
    by = {}
    for r in rows:
        by.setdefault(r["material"], []).append(r["phrase"])
    return "; ".join(f"{m}: {', '.join(ph)}" for m, ph in by.items())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref"); ap.add_argument("out")
    ap.add_argument("--whole", default=None)
    ap.add_argument("--step", default="all", choices=["bom", "whole", "parts", "place", "all"])
    a = ap.parse_args()
    ref, out = Path(a.ref), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = bom(ref, out)
    for r in rows:
        print(f"{'BASE ' if r['base'] else '     '}{r['phrase']:<32} {r['material']:<22} x{r['count']}"
              f"{' mirrored' if r['mirrored'] else ''}{' revolution' if r['revolution'] else ''}"
              f"{' standalone' if r.get('standalone') else ''}", flush=True)
    print("description:", description(rows), flush=True)
    if a.step == "bom":
        return
    whole = Path(a.whole) if a.whole else out / "whole"
    if a.step in ("whole", "all") and not (whole / "mat" / "splat_materials.ply").exists():
        whole_flow(ref, out, whole, rows)
    if a.step == "whole":
        return
    if a.step in ("parts", "all"):
        make_parts(ref, out, rows)
    if a.step == "parts":
        return
    place_all(out, whole, rows)


def whole_flow(ref: Path, out: Path, whole: Path, rows):
    """The whole object's splat with the BOM as its part list: Idol's splat from OUT/whole_src (input.ply,
    idol_input.png, idol_camera.json) when it is there, else Idol on the reference picture."""
    import subprocess
    whole.mkdir(parents=True, exist_ok=True)
    (whole / "materials.txt").write_text(description(rows))
    src = out / "whole_src"
    cmd = [sys.executable, "-X", "utf8", str(ROOT / "scripts/splat_flow.py")]
    if (src / "input.ply").exists():
        cmd += [str(src / "input.ply"), str(whole), "--input", str(src / "idol_input.png"),
                "--camera", str(src / "idol_camera.json")]
    else:
        cmd += [str(ref), str(whole), "--gaussians", "1048576"]
    subprocess.run(cmd, check=True)



# ---------------------------------------------------------------- 3. parts: each standalone component on its own
PART_GAUSSIANS = 262144


def slug(t):
    import re
    return re.sub(r"[^a-z0-9]+", "_", re.sub(r"^the\s+", "", t.lower())).strip("_")


def _components_module():
    sys.path.insert(0, str(ROOT / "scripts"))
    import components as C
    return C


def make_parts(ref: Path, out: Path, rows):
    """Reference mask (ENEAS on the picture itself) → one copy → its whole extent (recomposed where something
    covered it) → Iris-3B's clean picture of it alone (deepseek picks; drawn from its description if none is complete) → Idol → the
    component's material on every gaussian, finished like any splat (normals, albedo, glass tint)."""
    C = _components_module()
    comps = [r for r in rows if r.get("standalone")]
    masks_f = out / "ref_masks.npz"
    C.eneas_on_reference(ref, [r["phrase"] for r in comps], masks_f)
    z = np.load(masks_f)
    H, W = np.asarray(Image.open(ref)).shape[:2]
    rs = lambda m: np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((W, H))) > 127
    allm = {r["phrase"]: rs(z[r["phrase"]][0] if z[r["phrase"]].ndim == 3 else z[r["phrase"]])
            for r in comps if r["phrase"] in z.files}
    for r in comps:
        ph = r["phrase"]; d = out / "parts" / slug(ph); d.mkdir(parents=True, exist_ok=True)
        if (d / "mat" / "splat_materials.ply").exists():
            continue
        m = allm.get(ph)
        if m is None or m.mean() < C.MIN_MASK / 4:
            print(f"{ph}: no mask on the reference, left in the whole", flush=True)
            continue
        copy = pick_copy(C.instance_masks(m), {k: o for k, o in allm.items() if k != ph}) if r["count"] > 1 else m
        others = np.zeros_like(m)
        for k, o in allm.items():
            if k != ph:
                others |= o
        ys, xs = np.nonzero(copy)
        side = max(np.ptp(ys), np.ptp(xs)) + 1
        if side < SMALL_PX:
            # too few pixels to cut out (a buckle of 25 px): the piece is drawn from its description instead
            pic = described_picture(ref, ph, r["material"], (xs.min(), ys.min(), xs.max(), ys.max()), d)
        else:
            region = C.recompose_region(copy, others, r["revolution"])
            cut = d / "cut.png"
            C.cut_out(ref, copy, cut, region if region.sum() > copy.sum() * 1.05 else None)
            pic = None
            try:
                pic = C.iris_picture(ref, ph, r["material"], cut, others, d)
            except Exception as e:
                print(f"{ph}: Iris failed ({e})", flush=True)
            # no complete recomposition: a cut-out with straight edges makes a cut piece, so it is drawn whole
            # from its description instead
            pic = Path(pic) if pic else described_picture(ref, ph, r["material"],
                                                          (xs.min(), ys.min(), xs.max(), ys.max()), d)
        print(f"{ph}: picture {pic.name}", flush=True)
        from splat2mesh.remote import idol_sample
        ply = idol_sample(pic, d / "idol", PART_GAUSSIANS, 42)
        if flatness(ply) < CARD_MAX and pic.name != "described.png":
            # Idol made a flat card (a glass flask seen as its picture): drawn whole from its description instead
            print(f"{ph}: flat card ({flatness(ply):.3f}), drawn from its description", flush=True)
            pic = described_picture(ref, ph, r["material"], (xs.min(), ys.min(), xs.max(), ys.max()), d)
            ply = idol_sample(pic, d / "idol_described", PART_GAUSSIANS, 42)
        one_material(ply, d, f"{r['material']}|{ph}")


CARD_MAX = 0.03          # thinnest / widest spread of a part's gaussians: a flask card 0.005, a flat buckle 0.14


def flatness(ply) -> float:
    """Thinnest over widest principal spread of the opaque gaussians."""
    v, xyz, al = _gauss(ply)
    x = xyz[al > 0.3]
    e = np.sort(np.linalg.svd(x - x.mean(0), compute_uv=False))
    return float(e[0] / max(e[2], 1e-9))


SMALL_PX = 80            # a copy under this many pixels across in the reference is drawn from its description
DESCRIBE = ("The first picture shows {phrase} of the object in the second picture, enlarged. Describe that single "
            "piece for a product photographer, in one English sentence of at most 30 words: its shape, its colour, "
            "its material ({material}) and its visible details, as one separate piece (not worn, not attached). "
            "Answer JSON: {{\"description\": \"...\"}}")


CLAIM_SHARE, CLAIM_SIZE = 0.5, 20    # a blob over half inside another part's mask under 20x its size is that part


def pick_copy(blobs, others: dict):
    """The copy of a repeated part to make: its largest blob that no other part of similar size claims ("the corset
    buckles" also catch the belt buckle, which "the belt buckle" claims; the corset covering every buckle is far
    bigger and does not count)."""
    for b in blobs:
        n = b.sum()
        if not any((b & o).sum() > CLAIM_SHARE * n and o.sum() < CLAIM_SIZE * n for o in others.values()):
            return b
    return blobs[0]


def described_picture(ref: Path, phrase: str, material: str, box, d: Path) -> Path:
    """A small component, drawn whole: deepseek-flash describes it from an enlarged crop (with the whole picture
    for context), Iris-3B Turbo draws that description as a product photo on white."""
    import subprocess
    pic = d / "described.png"
    if pic.exists():
        return pic
    from splat2mesh.deepseek import ask
    from splat2mesh.deepseek import img_url
    from splat2mesh.config import cfg, gpu_lock, iris_python
    im = Image.open(ref).convert("RGB")
    x0, y0, x1, y1 = box; m = max(x1 - x0, y1 - y0) * 0.6
    crop = im.crop((int(max(0, x0 - m)), int(max(0, y0 - m)), int(min(im.width, x1 + m)), int(min(im.height, y1 + m))))
    crop = crop.resize((512, int(512 * crop.height / max(crop.width, 1))), Image.LANCZOS)
    j = ask([{"type": "image_url", "image_url": {"url": img_url(crop)}},
             {"type": "image_url", "image_url": {"url": img_url(im)}},
             {"type": "text", "text": DESCRIBE.format(phrase=phrase, material=material)}])
    desc = str(j.get("description", "")).strip() or f"{phrase} made of {material}"
    prompt = (f"product photo of one single piece: {desc}. Isolated on a plain white background, centred, the whole "
              f"piece in the frame, sharp focus, studio light")
    (d / "described.txt").write_text(prompt)
    print(f"{phrase}: drawn from its description: {desc}", flush=True)
    with gpu_lock():
        subprocess.run([iris_python(), str(ROOT / "scripts/iris_local.py"), prompt, str(pic), "--seed", "7",
                        "--size", "1024x1024", "--checkpoint", str(Path(cfg("iris.dir")) / "weights" / "turbo")],
                       check=True)
    return pic


def one_material(ply, d: Path, name: str):
    """A component made of one library material: every gaussian takes it (no segmentation), then the same finish
    pass as any splat (gs_finish: normals from its own athenea depth renders, floaters, albedo, glass tint)."""
    from splat2mesh.gs_materials import render_orbit_gs, write_ply
    from splat2mesh.gs_finish import finish
    mat = d / "mat"; mat.mkdir(exist_ok=True)
    render_orbit_gs(ply, mat / "gs_orbit")
    from splat2mesh.ply import read_vertex
    n = len(read_vertex(ply))
    write_ply(ply, mat / "splat_materials.ply", np.ones((1, n), np.float32), [name])
    print(name, finish(mat / "splat_materials.ply", mat / "gs_orbit"), flush=True)


# ---------------------------------------------------------------- 4. place: each copy fitted where the whole has it
CLUSTER_R = 0.012       # × the whole's size: gaussians of one part closer than this are one copy
COPY_MIN = 0.15         # a 3D cluster is a copy if it has at least this share of the largest one's gaussians
YAW_SPAN, YAW_STEP = 60, 15     # degrees tried around the rotation the two Idol cameras give
ICP_SCALE_TOL = 0.3


FIT_MIN = 0.2           # two-way share of points within reach after ICP; under it the copy stays as the whole has it
SIZE_TOL = 1.6          # the placed copy's extent may be at most this × (and at least 1/this ×) the whole's copy


def _extent(P):
    return float(np.linalg.norm(np.percentile(P, 98, 0) - np.percentile(P, 2, 0)))


def accept(T, f, Pp, Pw):
    """A fitted copy is placed only if its transform is finite, its fit is fair and its size matches the copy in
    the whole (a scaled ICP on a poor fit can grow a buckle to the size of the corset)."""
    if not np.all(np.isfinite(T)):
        return False, "no fit"
    if f < FIT_MIN:
        return False, f"fit {f:.2f}"
    r = _extent(Pp @ T[:3, :3].T) / max(_extent(Pw), 1e-9)
    if not 1 / SIZE_TOL <= r <= SIZE_TOL:
        return False, f"size ×{r:.2f}"
    return True, ""


def _gauss(ply):
    from splat2mesh.ply import read_vertex
    v = read_vertex(ply)
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    al = 1 / (1 + np.exp(-np.asarray(v["opacity"], np.float64)))
    return v, xyz, al


def cam_basis(cam_json):
    """Idol's camera for its input picture as an orthonormal basis [right, up, forward] (columns) in the splat's own
    file frame (Y down): the eye direction is (cos p sin y, -sin p, cos p cos y) (components.idol_view, in file axes)."""
    c = json.loads(Path(cam_json).read_text()); y, p = c["yaw"], c["pitch"]
    e = np.array([np.cos(p) * np.sin(y), -np.sin(p), np.cos(p) * np.cos(y)])
    f = -e / np.linalg.norm(e); up0 = np.array([0.0, -1.0, 0.0])
    r = np.cross(f, up0); r /= np.linalg.norm(r); u = np.cross(r, f)
    return np.stack([r, u, f], 1)


def copies(xyz, size, count):
    """The copies of a part in the whole: connected 3D clusters of its gaussians (radius CLUSTER_R × size), largest
    first, at most count."""
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    if len(xyz) < 30:
        return []
    sub = np.arange(len(xyz))
    if len(sub) > 60000:
        sub = np.random.default_rng(0).choice(len(xyz), 60000, replace=False)
    pairs = cKDTree(xyz[sub]).query_pairs(CLUSTER_R * size, output_type="ndarray")
    from scipy.sparse import coo_matrix
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(sub), len(sub)))
    n, lab = connected_components(g, directed=False)
    sizes = np.bincount(lab, minlength=n); order = np.argsort(-sizes)
    keep = [i for i in order if sizes[i] >= COPY_MIN * sizes[order[0]]][:max(count, 1)]
    # every gaussian of the part joins its nearest kept cluster
    sel = np.isin(lab, keep)
    nn = cKDTree(xyz[sub][sel]).query(xyz)[1]
    lab_all = lab[sel][nn]
    return [np.nonzero(lab_all == k)[0] for k in keep]


def _rotm(axis, deg):
    a = np.radians(deg); axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def fit(Pp, Pw, R0, mirror=None):
    """Similarity transform of the part's centres Pp onto the whole's copy Pw: rotations R0 then yaw steps about the
    vertical (after an optional mirror), scale from the RMS radii, centroids matched, then scaled ICP. Best by the
    two-way inlier share. Returns (T 4x4, fitness)."""
    import open3d as o3d
    rng = np.random.default_rng(0)
    Pp = Pp[rng.permutation(len(Pp))[:20000]]; Pw = Pw[rng.permutation(len(Pw))[:20000]]
    cw = Pw.mean(0); rw = np.sqrt(((Pw - cw) ** 2).sum(1).mean())
    diag = float(np.linalg.norm(np.percentile(Pw, 98, 0) - np.percentile(Pw, 2, 0)))
    dst = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Pw))
    M = np.eye(3) if mirror is None else mirror
    best = None
    for dy in range(-YAW_SPAN, YAW_SPAN + 1, YAW_STEP):
        R = _rotm([0, -1, 0], dy) @ M @ R0
        Q = Pp @ R.T; cq = Q.mean(0); rq = np.sqrt(((Q - cq) ** 2).sum(1).mean())
        s = rw / max(rq, 1e-9)
        T0 = np.eye(4); T0[:3, :3] = s * R; T0[:3, 3] = cw - s * cq
        src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Pp))
        reg = o3d.pipelines.registration.registration_icp(
            src, dst, 0.1 * diag, T0, o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
        T = reg.transformation
        if abs(np.cbrt(abs(np.linalg.det(T[:3, :3]))) / s - 1) > ICP_SCALE_TOL:
            T = T0
        a = o3d.pipelines.registration.evaluate_registration(src, dst, 0.05 * diag, T).fitness
        b = o3d.pipelines.registration.evaluate_registration(dst, src.transform(T), 0.05 * diag, np.eye(4)).fitness
        if best is None or a * b > best[1]:
            best = (np.asarray(T), a * b)
    return best


def _quat_mat(q):
    w, x, y, z = q.T
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
                     np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
                     np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], 1)


def _mat_quat(R):
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    w = np.sqrt(np.clip(1 + tr, 1e-12, None)) / 2
    q = np.stack([w, (R[:, 2, 1] - R[:, 1, 2]) / (4 * w), (R[:, 0, 2] - R[:, 2, 0]) / (4 * w),
                  (R[:, 1, 0] - R[:, 0, 1]) / (4 * w)], 1)
    bad = w < 1e-3                       # near 180°: the robust branch
    if bad.any():
        from scipy.spatial.transform import Rotation
        xyzw = Rotation.from_matrix(R[bad]).as_quat()
        q[bad] = xyzw[:, [3, 0, 1, 2]]
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def transform_gaussians(v, T):
    """The part's gaussians moved by a similarity T (4x4, possibly with a mirror): positions, rotations (a mirror
    flips one of each gaussian's own axes so the rotation stays proper: Σ' = (H R D) S² (H R D)^T), log scales,
    normals."""
    A = T[:3, :3]; s = np.cbrt(abs(np.linalg.det(A))); Rm = A / s
    out = v.copy()
    p = np.stack([v["x"], v["y"], v["z"]], 1) @ A.T + T[:3, 3]
    out["x"], out["y"], out["z"] = p[:, 0], p[:, 1], p[:, 2]
    q = np.stack([v[f"rot_{i}"] for i in range(4)], 1).astype(np.float64)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    G = np.einsum("ij,njk->nik", Rm, _quat_mat(q))
    if np.linalg.det(Rm) < 0:
        G = G @ np.diag([-1.0, 1, 1])
    qn = _mat_quat(G)
    for i in range(4):
        out[f"rot_{i}"] = qn[:, i]
    for i in range(3):
        out[f"scale_{i}"] = np.asarray(v[f"scale_{i}"]) + np.log(s)
    if "nx" in v.dtype.names:
        n = np.stack([v["nx"], v["ny"], v["nz"]], 1) @ Rm.T
        out["nx"], out["ny"], out["nz"] = n[:, 0], n[:, 1], n[:, 2]
    return out


def place_all(out: Path, whole: Path, rows):
    from splat2mesh.ply import write_vertex
    C = _components_module()
    wply = whole / "mat" / "splat_materials.ply"
    names = json.loads((whole / "mat" / "mat_names.json").read_text())["mat_id"]
    v, xyz, al = _gauss(wply)
    mid = np.asarray(v["mat_id"]).astype(int)
    loose = np.asarray(v["loose"]) > 0 if "loose" in v.dtype.names else np.zeros(len(v), bool)
    size = float(np.linalg.norm(np.percentile(xyz, 99, 0) - np.percentile(xyz, 1, 0)))
    Bw = cam_basis(whole / "src" / "idol_camera.json")
    remove = np.zeros(len(v), bool); pieces = []; listing = []
    sym = C.symmetry_plane(xyz[(al > 0.3) & ~loose])
    for r in rows:
        if not r.get("standalone"):
            continue
        key = f"{r['material']}|{r['phrase']}"
        d = out / "parts" / slug(r["phrase"])
        pply = d / "mat" / "splat_materials.ply"
        if key not in names or not pply.exists():
            listing.append({"component": r["phrase"], "material": r["material"], "status": "kept in the whole"})
            continue
        k = names.index(key)
        sel = np.nonzero((mid == k) & ~loose)[0]
        cps = copies(xyz[sel], size, r["count"])
        pv, pxyz, pal = _gauss(pply)
        pmask = (pal > 0.3) & ~(np.asarray(pv["loose"]) > 0)
        cam = next((d / "idol").glob("*camera*.json"), None)
        R0 = Bw @ cam_basis(cam).T if cam else np.eye(3)
        placed = []; done = []
        for ci, idx in enumerate(cps):
            P = xyz[sel[idx]][al[sel[idx]] > 0.3]
            if len(P) < 50:
                continue
            Hm = None
            if r["mirrored"] and ci % 2 == 1 and sym is not None:
                Hm = np.eye(3); Hm[sym[0], sym[0]] = -1        # the other side's copy: the part mirrored
            T, f = fit(pxyz[pmask], P, R0, Hm)
            ok, why = accept(T, f, pxyz[pmask], P)
            if not ok:
                print(f"{r['phrase']} copy {ci + 1}/{r['count']}: not placed ({why}), the whole keeps it", flush=True)
                listing.append({"component": r["phrase"], "material": r["material"], "copy": ci + 1,
                                "of": r["count"], "status": f"kept in the whole ({why})"})
                continue
            placed.append((T, f, ci, Hm is not None))
            done.append(idx)
        # copies the whole does not show apart (hidden, or fused with the other one): mirrored through the plane
        if len(placed) < r["count"] and r["mirrored"] and sym is not None and placed:
            ax, c0 = sym
            Mw = np.eye(4); Mw[ax, ax] = -1; Mw[ax, 3] = 2 * c0
            T2 = Mw @ placed[0][0]
            if np.linalg.norm(T2[:3, 3] - placed[0][0][:3, 3]) > 0.02 * size:
                placed.append((T2, placed[0][1], len(placed), True))
        for T, f, ci, mir in placed:
            g = transform_gaussians(pv, T)
            g["mat_id"] = k
            pieces.append(g)
            listing.append({"component": r["phrase"], "material": r["material"], "copy": ci + 1, "of": r["count"],
                            "mirrored": bool(mir), "fit": round(float(f), 3), "gaussians": int(len(g)),
                            "transform": np.round(T, 6).tolist()})
            print(f"{r['phrase']} copy {ci + 1}/{r['count']}{' mirrored' if mir else ''}: fit {f:.2f}", flush=True)
        if placed and len(done) == len(cps):
            remove[sel] = True                      # every copy replaced: also the stray gaussians between them
        else:
            for idx in done:
                remove[sel[idx]] = True
    keep = v[~remove]
    common = [n for n in keep.dtype.names if all(n in p.dtype.names for p in pieces)]
    allv = np.concatenate([keep[common]] + [p[common] for p in pieces]) if pieces else keep
    asm = out / "assembly"; asm.mkdir(exist_ok=True)
    write_vertex(asm / "splat_materials.ply", allv)
    import shutil
    shutil.copy(whole / "mat" / "mat_names.json", asm / "mat_names.json")
    fin = json.loads((whole / "mat" / "finish.json").read_text())
    for r in rows:
        f2 = out / "parts" / slug(r["phrase"]) / "mat" / "finish.json"
        if f2.exists():
            fin = {**json.loads(f2.read_text()), **fin}
    (asm / "finish.json").write_text(json.dumps(fin, indent=1))
    (out / "components.json").write_text(json.dumps(listing, indent=1))
    from splat2mesh.athenea_splat import write_splat_usd
    print("USD:", write_splat_usd(asm / "splat_materials.ply", out / "assembly.usdc", y_down=True), flush=True)
    print(f"assembly: {len(keep)} gaussians of the whole kept, {sum(len(p) for p in pieces)} from "
          f"{len(pieces)} placed copies", flush=True)


if __name__ == "__main__":
    main()
