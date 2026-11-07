"""DeepSeek vision calls (JSON answers) used as describer, checker and validator: part lists, mask checks, colour
cluster names, multi-view scores. URL, model and key come from splat2mesh.config (DEEPSEEK_API_KEY or an env file).

    from splat2mesh.deepseek import ask, ask_image
    ask_image(pil_image, "question ... Answer JSON: {...}")    # → dict
"""
from __future__ import annotations

import base64
import io
import json
import time
import urllib.request

from .config import cfg, deepseek_key


def img_url(im) -> str:
    b = io.BytesIO(); im.convert("RGB").save(b, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def ask(content, retries: int = 3, timeout: int = 300) -> dict:
    """content: an OpenAI-style message content list (text and image_url parts). Returns the parsed JSON answer."""
    key = deepseek_key()
    if not key:
        raise RuntimeError("no DeepSeek key: set DEEPSEEK_API_KEY or deepseek.env_file (see splat2mesh.example.toml)")
    body = {"model": cfg("deepseek.model"), "temperature": 0.1, "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": content}]}
    req = urllib.request.Request(cfg("deepseek.url"), data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    err = None
    for i in range(retries):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=timeout))
            return json.loads(r["choices"][0]["message"]["content"])
        except Exception as e:                       # network hiccups: retry with backoff
            err = e; time.sleep(2 ** i)
    raise RuntimeError(f"DeepSeek call failed: {err}")


def ask_image(im, question: str, **kw) -> dict:
    return ask([{"type": "image_url", "image_url": {"url": img_url(im)}}, {"type": "text", "text": question}], **kw)
