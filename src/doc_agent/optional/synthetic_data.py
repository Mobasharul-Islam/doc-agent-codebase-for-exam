"""OPTIONAL — generate synthetic text-lines for tiny-label training
Activate only if your data speciality or NFR requires it (e.g. a tiny-label / data-efficient project). Off by default; CI does not require impl."""
from __future__ import annotations

import random
from pathlib import Path


def generate(cfg: dict, n: int = 50) -> list:
    """Render n (image_path, text) text-line pairs from corpus words using PIL, with random noise,
    for OCR fine-tuning when labels are scarce. Returns [(path, text), ...]."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    from ..ingest.loader import data_paths

    rng = random.Random(cfg.get("seed", 42))
    vocab = []
    for f in data_paths(cfg)["raw_dir"].rglob("*.txt"):
        vocab += f.read_text(encoding="utf-8").split()
    vocab = vocab or ["sample", "text", "line"]
    out_dir = data_paths(cfg)["interim_dir"] / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except OSError:
        font = ImageFont.load_default(28)
    pairs = []
    for i in range(n):
        text = " ".join(rng.choices(vocab, k=rng.randint(3, 7)))
        img = Image.new("L", (640, 48), 240)
        ImageDraw.Draw(img).text((8, 6), text, fill=30, font=font)
        arr = np.asarray(img, dtype=np.float32) + np.random.default_rng(i).normal(0, 8, (48, 640))
        path = Path(out_dir / f"line_{i:04d}.png")
        Image.fromarray(np.clip(arr, 0, 255).astype("uint8")).save(path)
        pairs.append((str(path), text))
    return pairs
