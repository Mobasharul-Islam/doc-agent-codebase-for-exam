"""Stage 9 — confidence calibration (calibrated-confidence NFR)"""
from __future__ import annotations

import numpy as np

from ..contracts import *  # noqa


class TemperatureScaler:
    """softmax(logits / T). T > 1 softens over-confident models, T < 1 sharpens."""
    def __init__(self, temperature: float) -> None:
        self.T = temperature
    def __call__(self, logits) -> np.ndarray:
        z = np.asarray(logits, dtype=float) / self.T
        z = z - z.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)


def temperature_scale(logits, labels):
    """Fit temperature on a validation set by minimising NLL (coarse log-grid, then refine).
    logits: (n, classes); labels: (n,) int. Returns a TemperatureScaler."""
    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels, dtype=int)

    def nll(T: float) -> float:
        p = TemperatureScaler(T)(logits)
        return float(-np.log(p[np.arange(len(labels)), labels] + 1e-12).mean())

    grid = np.logspace(-1, 1.3, 60)
    best = float(min(grid, key=nll))
    fine = np.linspace(best * 0.85, best * 1.15, 40)
    return TemperatureScaler(float(min(fine, key=nll)))

def ece(confidences, correct) -> float:
    """Expected calibration error with 10 equal-width bins: sum_b (n_b/n)*|acc_b - conf_b|."""
    conf = np.asarray(confidences, dtype=float)
    acc = np.asarray(correct, dtype=float)
    if len(conf) == 0:
        return 0.0
    bins = np.minimum((conf * 10).astype(int), 9)
    total = 0.0
    for b in range(10):
        m = bins == b
        if m.any():
            total += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(total)
