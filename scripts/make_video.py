"""Social video (LinkedIn-friendly 1080x1350, 4:5) from a storyboard of stills and frame sequences, with a caption
band per scene and a footer. Frames are composed with Pillow and encoded with ffmpeg (H.264, yuv420p, 24 fps).

    python scripts/make_video.py STORYBOARD.json OUT.mp4

STORYBOARD.json:
{"footer": "text", "scenes": [
  {"title": "...", "subtitle": "...", "image": "a.png", "seconds": 3},            # a still
  {"title": "...", "frames": "dir/f_%04d.png", "count": 96},                      # a sequence, one frame each
  {"title": "...", "card": ["big line", "small line", ...], "seconds": 3}         # a text card
]}
Scenes fade in from black over 6 frames. Stills fill a 1080x1080 square under the caption band.
"""
import json, subprocess, sys, tempfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

W, H, BAND, FPS, FADE = 1080, 1350, 190, 24, 6
BG, FG, ACCENT, MUTED = (24, 24, 27), (245, 245, 245), (240, 170, 60), (160, 160, 168)
FONT = "/System/Library/Fonts/HelveticaNeue.ttc"


def font(size, bold=False):
    return ImageFont.truetype(FONT, size, index=1 if bold else 0)


def wrap(d, text, f, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if d.textlength(t, font=f) > width and cur:
            lines.append(cur); cur = w
        else:
            cur = t
    return lines + ([cur] if cur else [])


def frame(scene, img, footer, step):
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    if step:
        d.text((48, 30), step, font=font(26, True), fill=ACCENT)
    y = 66
    for line in wrap(d, scene.get("title", ""), font(44, True), W - 96)[:2]:
        d.text((48, y), line, font=font(44, True), fill=FG); y += 52
    for line in wrap(d, scene.get("subtitle", ""), font(28), W - 96)[:2]:
        d.text((48, y), line, font=font(28), fill=MUTED); y += 34
    if img is not None:
        sq = img.convert("RGB"); sq.thumbnail((W, W))
        im.paste(sq, ((W - sq.width) // 2, BAND + (W - sq.height) // 2))
    if "card" in scene:
        lines = scene["card"]; y = BAND + 260
        for i, t in enumerate(lines):
            f = font(54, True) if i == 0 else font(34)
            for l in wrap(d, t, f, W - 160):
                d.text(((W - d.textlength(l, font=f)) // 2, y), l, font=f, fill=FG if i == 0 else MUTED)
                y += 66 if i == 0 else 46
            y += 24
    d.text((48, H - 58), footer, font=font(26), fill=MUTED)
    return im


def main():
    sb = json.loads(Path(sys.argv[1]).read_text()); out = sys.argv[2]
    base = Path(sys.argv[1]).parent
    tmp = Path(tempfile.mkdtemp()); n = 0
    for s in sb["scenes"]:
        step = s.get("step", "")
        if "frames" in s:
            srcs = [base / (s["frames"] % i) for i in range(s.get("start", 0), s.get("start", 0) + s["count"])]
        else:
            srcs = [base / s["image"] if "image" in s else None] * int(s.get("seconds", 3) * FPS)
        cache = {}
        for k, src in enumerate(srcs):
            if src not in cache:
                cache.clear(); cache[src] = frame(s, Image.open(src) if src else None, sb.get("footer", ""), step)
            f = cache[src]
            if k < FADE:
                f = Image.blend(Image.new("RGB", (W, H), (0, 0, 0)), f, (k + 1) / FADE)
            f.save(tmp / f"v_{n:05d}.png"); n += 1
    from splat2mesh.athenea import ffmpeg
    subprocess.run([ffmpeg(), "-loglevel", "error", "-y", "-framerate", str(FPS), "-i", str(tmp / "v_%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", out], check=True)
    print(f"wrote {out}: {n} frames, {n / FPS:.1f} s")


if __name__ == "__main__":
    main()
