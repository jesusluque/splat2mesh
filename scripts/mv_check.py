"""DeepSeek as a multi-view validator: is a 3D result the photographed object from every side?

One sheet goes to deepseek-flash: the input photo, then the result rendered from several angles (labelled with the
yaw turned from the photo's view). It scores each view and the whole set; no ground truth is needed, so it works on
characters and any object we only have one photo of.

    python scripts/mv_check.py PHOTO "0=front.png" "90=left.png" "180=back.png" "270=right.png" [--json OUT.json]
Prints and writes {"views": {deg: {"identity", "plausible", "clean", "why"}}, "coherent", "overall", "worst", "score"}
(medians over --runs answers; score 1-5 = half the mean view score, a quarter each coherent and overall):
  identity  1-5  same object as the photo (shape, proportions, parts, materials, colours);
  plausible 1-5  the sides the photo cannot show are believable for this object (not mirrored, flattened, missing);
  clean     1-5  free of artefacts (holes, floaters, smears, blur, broken thin parts);
  coherent  1-5  the views agree with each other as ONE 3D object;
  overall   1-5.
"""
import argparse, json, sys
from pathlib import Path
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from splat2mesh.deepseek import ask as ds, img_url

S = 384


def sheet(photo, views):
    W = Image.new("RGB", ((len(views) + 1) * S, S + 28), "white"); d = ImageDraw.Draw(W)
    for i, (lab, p) in enumerate([("PHOTO", photo)] + [(f"{k} deg", v) for k, v in views]):
        t = Image.open(p).convert("RGBA"); bg = Image.new("RGBA", t.size, "white"); t = Image.alpha_composite(bg, t).convert("RGB")
        t.thumbnail((S, S)); W.paste(t, (i * S + (S - t.width) // 2, 28 + (S - t.height) // 2))
        d.text((i * S + 8, 6), lab, fill="black")
    return W


def check(photo, views):
    labels = ", ".join(k for k, _ in views)
    q = ("The first picture (PHOTO) is the only photo of an object. The others are renders of a 3D model made from that "
         f"photo, turned around the object by the yaw written above each ({labels} degrees; 0 = the photo's view). "
         "Judge the 3D model. For each render score 1-5: identity (same object as the photo: shape, proportions, parts, "
         "materials, colours), plausible (sides the photo cannot show are believable for this object: not mirrored, "
         "flattened, duplicated or missing parts), clean (no holes, floaters, smears, blur, broken thin parts). Then "
         "score the set: coherent (all renders agree as ONE 3D object) and overall. Be strict: 5 only if a careful "
         "3D artist would accept it. Answer JSON: {\"views\": {\"<deg>\": {\"identity\": n, \"plausible\": n, "
         "\"clean\": n, \"why\": short}}, \"coherent\": n, \"overall\": n, \"worst\": short description of the worst "
         "problem}")
    return ds([{"type": "image_url", "image_url": {"url": img_url(sheet(photo, views))}}, {"type": "text", "text": q}])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photo"); ap.add_argument("views", nargs="+"); ap.add_argument("--json")
    ap.add_argument("--sheet", help="also save the sheet sent to DeepSeek")
    ap.add_argument("--runs", type=int, default=3, help="independent answers; per-score median")
    a = ap.parse_args()
    views = [v.split("=", 1) for v in a.views]
    if a.sheet:
        sheet(a.photo, views).save(a.sheet, quality=88)
    runs = [check(a.photo, views) for _ in range(a.runs)]      # one answer is noisy (overall 2..4 on one model)
    med = lambda xs: sorted(xs)[len(xs) // 2]
    r = {"views": {}, "runs": runs}
    for k, _ in views:
        vs = [x.get("views", {}).get(k, {}) for x in runs]
        r["views"][k] = {f: med([float(v.get(f, 0)) for v in vs]) for f in ("identity", "plausible", "clean")}
        r["views"][k]["why"] = vs[0].get("why")
    for f in ("coherent", "overall"):
        r[f] = med([float(x.get(f, 0)) for x in runs])
    r["worst"] = [x.get("worst") for x in runs]
    r["score"] = round(sum(sum(v[f] for f in ("identity", "plausible", "clean")) for v in r["views"].values())
                       / (3 * len(views)) * 0.5 + 0.25 * r["coherent"] + 0.25 * r["overall"], 2)
    print(json.dumps(r, indent=1, ensure_ascii=False))
    if a.json:
        Path(a.json).write_text(json.dumps(r, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
