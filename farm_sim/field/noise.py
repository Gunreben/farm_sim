"""Vectorised 2-D fractal value noise (numpy only, deterministic per seed)."""

from __future__ import annotations

import numpy as np


def _hash(ix: np.ndarray, iy: np.ndarray, seed: int) -> np.ndarray:
    h = (ix.astype(np.int64) * 374761393 + iy.astype(np.int64) * 668265263 + seed * 2147483647) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((h ^ (h >> 16)) & 0xFFFFFF) / float(0xFFFFFF)  # [0, 1)


def value_noise(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """Smooth value noise in [-1, 1] with unit lattice spacing."""
    x0, y0 = np.floor(x), np.floor(y)
    fx, fy = x - x0, y - y0
    u, v = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    a = _hash(x0, y0, seed)
    b = _hash(x0 + 1, y0, seed)
    c = _hash(x0, y0 + 1, seed)
    d = _hash(x0 + 1, y0 + 1, seed)
    return (a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v) * 2.0 - 1.0


def fractal(x, y, seed: int, wavelength: float, octaves: int = 4, gain: float = 0.5) -> np.ndarray:
    """Sum of octaves, normalised to roughly [-1, 1]."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    total, amp, norm, f = np.zeros(np.broadcast(x, y).shape), 1.0, 0.0, 1.0 / wavelength
    for o in range(octaves):
        total += amp * value_noise(x * f, y * f, seed + 1013 * o)
        norm += amp
        amp *= gain
        f *= 2.0
    return total / norm
