"""Stage 1 — ENHANCEMENT (VAE / diffusion) — generative denoise / super-resolution of degraded scans"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from torch import nn

from ..contracts import *  # noqa
from ..logging_conf import get_logger
from .loader import data_paths
from .preprocess import read_gray, write_image

log = get_logger(__name__)


class DenoiseNet(nn.Module):
    """Small residual conv encoder-decoder ('unet_small'). Trained as a one-step denoiser
    (degraded -> clean), the single-step special case of a diffusion denoiser; a multi-step
    diffusion sampler is out of scope for this study baseline."""

    def __init__(self, ch: int = 16) -> None:
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(1, ch, 3, padding=1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1), nn.ReLU())
        self.mid = nn.Sequential(nn.Conv2d(ch * 2, ch * 2, 3, padding=1), nn.ReLU())
        self.up = nn.ConvTranspose2d(ch * 2, ch, 2, stride=2)
        self.dec = nn.Sequential(
            nn.Conv2d(ch * 2, ch, 3, padding=1), nn.ReLU(), nn.Conv2d(ch, 1, 3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        m = self.mid(self.enc2(e1))
        d = self.dec(torch.cat([self.up(m), e1], dim=1))
        return torch.clamp(x + d, 0.0, 1.0)  # predict a residual correction


def degrade(clean: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Synthetic scan damage (float [0,1] array): blur + gaussian noise + salt/pepper speckle."""
    x = clean.copy()
    if rng.random() < 0.6:
        x = cv2.GaussianBlur(x, (0, 0), float(rng.uniform(0.5, 1.5)))
    x = x + rng.normal(0, rng.uniform(0.02, 0.12), x.shape).astype(np.float32)
    mask = rng.random(x.shape)
    x[mask < 0.005] = 0.0
    x[mask > 0.995] = 1.0
    return np.clip(x, 0, 1).astype(np.float32)


def cut_patches(pages: list[Page], patch: int, n_per: int, seed: int) -> list[np.ndarray]:
    """Random clean grayscale patches (float [0,1]) from the pages."""
    rng = np.random.default_rng(seed)
    out = []
    for p in pages:
        g = read_gray(p.image_path).astype(np.float32) / 255.0
        if g.shape[0] <= patch or g.shape[1] <= patch:
            continue
        for _ in range(n_per):
            y = int(rng.integers(0, g.shape[0] - patch))
            x = int(rng.integers(0, g.shape[1] - patch))
            out.append(g[y : y + patch, x : x + patch])
    return out


class Enhancer:
    """Model set by cfg['enhance']. train() fits the denoiser on synthetic degradations of the
    corpus itself; apply() uses the checkpoint if present, else a classical non-local-means fallback."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg["enhance"]
        self.paths = data_paths(cfg)
        self.ckpt = self.paths["models_dir"] / "enhancer.pt"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.train_cfg = cfg.get("train", {})
        self.seed = cfg.get("seed", 42)
        self._net: DenoiseNet | None = None

    def train(self, pages: list[Page]) -> None:
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        patches = cut_patches(
            pages,
            self.train_cfg.get("patch", 64),
            self.train_cfg.get("patches_per_page", 32),
            self.seed,
        )
        assert patches, "no pages large enough to cut training patches"
        net = DenoiseNet().to(self.device)
        opt = torch.optim.Adam(net.parameters(), lr=self.train_cfg.get("lr", 1e-3))
        bs = self.train_cfg.get("batch_size", 16)
        epochs = self.train_cfg.get("epochs", 3)
        for ep in range(epochs):
            order = rng.permutation(len(patches))
            total = 0.0
            for i in range(0, len(order), bs):
                clean = np.stack([patches[j] for j in order[i : i + bs]])
                noisy = np.stack([degrade(c, rng) for c in clean])
                xt = torch.from_numpy(noisy)[:, None].to(self.device)
                yt = torch.from_numpy(clean)[:, None].to(self.device)
                loss = nn.functional.l1_loss(net(xt), yt)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss) * len(clean)
            log.info("enhancer epoch %d/%d L1=%.4f", ep + 1, epochs, total / len(order))
        self.ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(net.state_dict(), self.ckpt)
        self._net = net.eval()

    def _load(self) -> DenoiseNet | None:
        if self._net is None and self.ckpt.exists():
            net = DenoiseNet().to(self.device)
            net.load_state_dict(torch.load(self.ckpt, map_location=self.device))
            self._net = net.eval()
        return self._net

    def enhance_image(self, gray: np.ndarray) -> np.ndarray:
        net = self._load()
        if net is None:  # classical fallback
            den = cv2.fastNlMeansDenoising(gray, None, h=10)
            blur = cv2.GaussianBlur(den, (0, 0), 1.2)
            return cv2.addWeighted(den, 1.5, blur, -0.5, 0)
        h, w = gray.shape
        ph, pw = (-h) % 4, (-w) % 4
        x = np.pad(gray.astype(np.float32) / 255.0, ((0, ph), (0, pw)), mode="edge")
        with torch.no_grad():
            y = net(torch.from_numpy(x)[None, None].to(self.device))[0, 0].cpu().numpy()
        return (y[:h, :w] * 255).round().astype(np.uint8)

    def apply(self, pages: list[Page]) -> list[Page]:
        out_dir = self.paths["interim_dir"] / "enhanced"
        out = []
        for p in pages:
            dst = out_dir / f"{p.id}.png"
            write_image(dst, self.enhance_image(read_gray(p.image_path)))
            out.append(Page(id=p.id, image_path=str(dst), doc_id=p.doc_id))
        return out


def run(pages: list[Page], cfg: dict) -> list[Page]:
    if not cfg["enhance"]["enabled"]:
        return pages
    return Enhancer(cfg).apply(pages)
