"""Parts of an object by material, from its picture: what ENEAS will look for.

describe_ds: DeepSeek (vision) lists the visible parts grouped by the library's material names.
part_materials_from_prompt: a part the generation prompt names with its material takes that material.
"""
from __future__ import annotations

from .materials import LIB


def describe_ds(image, prompt, max_parts=20):
    import re
    from PIL import Image
    from splat2mesh.deepseek import ask as ds, img_url
    q = ("You prepare inputs for a segmentation model that finds one object part per short phrase. List the visible "
         "PARTS of this object grouped by MATERIAL. Rules: use only these material names: " + ", ".join(LIB) +
         f". Each part is a short noun phrase starting with 'the' that a segmenter can find (one part per phrase, "
         f"never two joined with 'and'). 3 to {max_parts} parts; cover the whole object: first the parts that make up "
         "most of its visible surface (body, clothing, footwear, hands, face, hair, large panels or boards), then smaller "
         "details. List visible glass or transparent parts as their own part, and use a transparent material (glass, "
         "crystal, gemstone) ONLY for parts you can see through in the picture: glossy or shiny is not transparent "
         "(a figurine's glossy eyes are paint). A painted figurine or model: name the "
         "material each painted part represents (steel, wood, leather, skin, hair / fur...), not the paint. Each part "
         "once." + (f' The image was generated from this description: "{prompt}".' if prompt else "") +
         ' Answer JSON: {"materials": [{"material": ..., "parts": ["the ...", ...]}]}')
    ans = ds([{"type": "image_url", "image_url": {"url": img_url(Image.open(image))}}, {"type": "text", "text": q}])
    groups, seen = {}, set()
    for g in ans.get("materials", []):
        mat = str(g.get("material", "")).strip().lower()
        if mat not in LIB:
            continue
        for p_ in g.get("parts", []):
            p_ = str(p_).strip().strip(".").lower()
            p_ = p_ if p_.startswith("the ") else "the " + p_
            if p_ in seen or len(seen) >= max_parts:
                continue
            seen.add(p_); groups.setdefault(mat, []).append(p_)
    if not groups:
        raise ValueError("empty description")
    return "; ".join(f"{m}: {', '.join(ps)}" for m, ps in groups.items())


def part_materials_from_prompt(desc, pr):
    """A part the prompt names with its material right before it ("a round grey stone base", "with a brass rim")
    takes that material. The part's noun must be the head of the phrase (followed by punctuation, the end or a
    linking word): in "gold belt buckle" the gold is the buckle's, not the belt's."""
    import re
    pairs = []
    for chunk in desc.split(";"):
        if ":" in chunk:
            mat, things = chunk.split(":", 1)
            pairs += [(mat.strip(), t.strip()) for t in things.split(",") if t.strip()]
    out = []
    for mat, part in pairs:
        noun = part.split()[-1].lower()
        for m in re.finditer(r"((?:[\w-]+ ){1,3})" + re.escape(noun) + r"(?=\s*(?:[,.;]|$| with | and | on | in | holding | standing ))", pr):
            words = m.group(1)
            hit = [k for k, v in LIB.items() if any(re.search(r"\b" + re.escape(w) + r"\b", words) for w in v["keywords"])]
            if len(hit) == 1 and hit[0] != mat:
                print(f"prompt names the material of {part}: {mat} → {hit[0]}", flush=True)
                mat = hit[0]
            break
        out.append((mat, part))
    groups = {}
    for mat, part in out:
        groups.setdefault(mat, []).append(part)
    return "; ".join(f"{m}: {', '.join(ps)}" for m, ps in groups.items())


def complete_parts(desc, image, max_new=3):
    """DeepSeek checks the description's coverage: it sees the picture and the part list and adds up to max_new large
    visible parts that are missing (a 4B describer leaves out boots or a cloak), with library material names.
    Unchanged when DeepSeek is not reachable."""
    import re
    from PIL import Image
    try:
        from splat2mesh.deepseek import ask as ds, img_url
        have = [t.strip() for c in desc.split(";") if ":" in c for t in c.split(":", 1)[1].split(",")]
        q = ("This is a picture of an object and the list of its parts used to assign materials: " + "; ".join(have) +
             f". Name up to {max_new} LARGE visible parts that are missing from the list (each a short noun phrase "
             "starting with 'the' that a segmentation model can find; not a part already listed under another name, "
             "not tiny details). For each give its material, using only these names: " + ", ".join(LIB) +
             ". If nothing large is missing, return an empty list. Answer JSON: {\"missing\": [{\"part\": ..., "
             "\"material\": ...}]}")
        ans = ds([{"type": "image_url", "image_url": {"url": img_url(Image.open(image))}}, {"type": "text", "text": q}])
    except Exception as e:
        print("coverage check skipped:", e, flush=True)
        return desc
    low = {h.lower() for h in have}
    for m in ans.get("missing", [])[:max_new]:
        part, mat = str(m.get("part", "")).strip().lower(), str(m.get("material", "")).strip().lower()
        if not part or mat not in LIB or part in low:
            continue
        part = part if part.startswith("the ") else "the " + part
        print(f"coverage check adds: {mat}: {part}", flush=True)
        desc = desc + f"; {mat}: {part}"; low.add(part)
    return desc
