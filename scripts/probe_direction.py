"""Direction in Idol's residual stream that separates better from worse samples of the same photo.

    python scripts/probe_direction.py DIR [--out dir.npz]
DIR holds <photo>_s<seed>_acts.npy ([steps, 24, 1024]) and <photo>_s<seed>.check.json (DeepSeek score).
Per photo, activations and scores are centred (the photo's content cancels); the direction per (step, block) is
sum(score_c * act_c) over samples. Leave-one-photo-out: the held-out photo's centred activations projected on the
direction found without it must correlate with its centred scores. Reports that correlation per (step, block).
"""
import argparse, glob, json, re
from collections import defaultdict
import numpy as np

STEPS = (2, 5, 10, 20, 30, 40)


def load(d):
    by = defaultdict(list)
    for f in sorted(glob.glob(f"{d}/*.check.json")):
        b = f[:-11]
        try:
            A = np.load(b + "_acts.npy")
        except FileNotFoundError:
            continue
        by[re.sub(r"_s\d+$", "", b.split("/")[-1])].append((json.load(open(f))["score"], A))
    return {p: v for p, v in by.items() if len(v) >= 2}


def centred(by):
    out = {}
    for p, v in by.items():
        s = np.array([x for x, _ in v]); A = np.stack([a for _, a in v])
        out[p] = (s - s.mean(), A - A.mean(0))
    return out


def direction(C, skip=None):
    D = sum(np.einsum("n,nsbc->sbc", s, A) for p, (s, A) in C.items() if p != skip)
    return D / (np.linalg.norm(D, axis=-1, keepdims=True) + 1e-9)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--out")
    a = ap.parse_args()
    by = load(a.dir); C = centred(by)
    print(len(by), "photos,", sum(len(v) for v in by.values()), "samples")
    proj, sc = [], []
    for p, (s, A) in C.items():
        D = direction(C, skip=p)
        proj.append(np.einsum("nsbc,sbc->nsb", A, D)); sc.append(s)
    P = np.concatenate(proj); S = np.concatenate(sc)
    r = np.array([[np.corrcoef(P[:, i, j], S)[0, 1] for j in range(P.shape[2])] for i in range(P.shape[1])])
    np.set_printoptions(precision=2, suppress=True, linewidth=200)
    print("LOO correlation, rows = steps", STEPS, "cols = blocks 0-23\n", r)
    i, j = np.unravel_index(np.nanargmax(r), r.shape)
    print("best step", STEPS[i], "block", j, "r", round(float(r[i, j]), 3))
    if a.out:
        D = direction(C)                                          # [steps, 24, 1024]
        allA = np.concatenate([np.stack([x for _, x in v]) for v in by.values()])
        scale = np.linalg.norm(allA, axis=-1).mean(0)             # [steps, 24]
        np.savez(a.out, dir_all=D, scale_all=scale, r=r, steps=np.array(STEPS))
        print("saved", a.out)


if __name__ == "__main__":
    main()
