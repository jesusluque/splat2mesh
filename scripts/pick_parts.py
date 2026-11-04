"""Automatic choice among Iris candidates for each component (deepseek-flash, vision).

For each component: one sheet with the reference photo, the component's cut-out and the candidates (numbered); the
model scores every candidate for "this one part alone, complete, same shape, proportions and material as in the
reference, nothing else in it or attached to it" and picks the best. Below ACCEPT, no candidate is used (the
caller keeps the cut-out).

    python scripts/pick_parts.py REF.png OUT.json "phrase=cutout.png=cand1.png,cand2.png,..." ...
"""
import base64, io, json, sys, urllib.request
from pathlib import Path
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path.home() / ".claude/skills/research-deepseek/scripts"))
import dsr

ACCEPT = 3
S = 384
PROMPT = ("Image 0 (top left) is a reference photo of a whole object. Image 1 is the part '{part}' cut out of that "
          "photo. Images 2..{n} are candidate pictures of that part generated to be turned into a 3D model of the part "
          "by itself. Score each candidate 1-5: 5 = shows only '{part}', alone and complete (hidden sides completed "
          "plausibly), with the same shape, proportions, colour and material as in the reference photo; lower if it "
          "contains other parts of the object (inside it, behind it or attached), if it changes the shape or "
          "proportions, if it is cropped or if it adds things. Answer JSON: {{\"scores\": {{\"2\": s, ...}}, \"best\": "
          "index, \"why\": short reason}}")


def sheet(ref, cut, cands):
    ims = [ref, cut] + cands
    cols = min(len(ims), 4); rows = (len(ims) + cols - 1) // cols
    im = Image.new("RGB", (cols * S, rows * S), "white"); d = ImageDraw.Draw(im)
    for i, p in enumerate(ims):
        t = Image.open(p).convert("RGB"); t.thumbnail((S, S))
        x, y = (i % cols) * S, (i // cols) * S
        im.paste(t, (x + (S - t.width) // 2, y + (S - t.height) // 2))
        d.rectangle([x + 2, y + 2, x + 34, y + 26], fill="black"); d.text((x + 10, y + 7), str(i), fill="white")
    return im


def ask(im, part, n):
    b = io.BytesIO(); im.save(b, "JPEG", quality=88)
    url = "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()
    body = {"model": "deepseek-flash", "temperature": 0.1, "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                                      {"type": "text", "text": PROMPT.format(part=part, n=n)}]}]}
    req = urllib.request.Request(dsr.API, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + dsr.key()})
    return json.loads(json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]["message"]["content"])


def pick(ref, items):
    res = {}
    for phrase, cut, cands in items:
        im = sheet(ref, cut, cands)
        j = ask(im, phrase, len(cands) + 1)
        sc = {int(k): float(v) for k, v in (j.get("scores") or {}).items() if str(k).isdigit()}
        best = max(sc, key=sc.get) if sc else None
        ok = best is not None and 2 <= best < len(cands) + 2 and sc[best] >= ACCEPT
        res[phrase] = {"scores": {str(cands[k - 2]): v for k, v in sc.items() if 2 <= k < len(cands) + 2},
                       "chosen": str(cands[best - 2]) if ok else None, "why": j.get("why")}
        print(phrase, res[phrase], flush=True)
    return res


if __name__ == "__main__":
    ref, out = Path(sys.argv[1]), Path(sys.argv[2])
    items = []
    for a in sys.argv[3:]:
        ph, cut, cs = a.split("=", 2)
        items.append((ph, Path(cut), [Path(c) for c in cs.split(",")]))
    out.write_text(json.dumps(pick(ref, items), indent=1))
