"""Deterministic noise helpers shared by geometry and texture generation (numpy only)."""
import numpy as np


def fade(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def smoothstep(e0, e1, x):
    t = np.clip((np.asarray(x, dtype=np.float64) - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class Perlin2D:
    """Gradient noise, tileable with the given lattice period. Output roughly in [-1, 1]."""

    def __init__(self, seed, period=256):
        rng = np.random.default_rng(seed)
        ang = rng.random((period, period)) * 2.0 * np.pi
        self.gx = np.cos(ang).astype(np.float32)
        self.gy = np.sin(ang).astype(np.float32)
        self.period = period

    def __call__(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        p = self.period
        xi = np.floor(x)
        yi = np.floor(y)
        xf = (x - xi).astype(np.float32)
        yf = (y - yi).astype(np.float32)
        x0 = xi.astype(np.int64) % p
        y0 = yi.astype(np.int64) % p
        x1 = (x0 + 1) % p
        y1 = (y0 + 1) % p
        gx, gy = self.gx, self.gy
        n00 = gx[y0, x0] * xf + gy[y0, x0] * yf
        n10 = gx[y0, x1] * (xf - 1) + gy[y0, x1] * yf
        n01 = gx[y1, x0] * xf + gy[y1, x0] * (yf - 1)
        n11 = gx[y1, x1] * (xf - 1) + gy[y1, x1] * (yf - 1)
        u = fade(xf)
        v = fade(yf)
        a = n00 + u * (n10 - n00)
        b = n01 + u * (n11 - n01)
        return (a + v * (b - a)) * 1.4142

    def fbm(self, x, y, octaves=5, lacunarity=2.0, gain=0.5):
        total = 0.0
        amp = 1.0
        norm = 0.0
        f = 1.0
        for o in range(octaves):
            total = total + amp * self(x * f + 19.1 * o, y * f + 7.3 * o)
            norm += amp
            amp *= gain
            f *= lacunarity
        return total / norm


def smooth_random_1d(seed, n_harmonics=6, falloff=1.0, periodic=True):
    """Smooth random function of t with unit RMS. Periodic on [0, 1] when periodic=True."""
    rng = np.random.default_rng(seed)
    k = np.arange(1, n_harmonics + 1, dtype=np.float64)
    amps = rng.normal(size=n_harmonics) / k ** falloff
    phases = rng.random(n_harmonics) * 2.0 * np.pi
    freqs = k if periodic else k * (0.7 + 0.6 * rng.random(n_harmonics))
    rms = np.sqrt((amps ** 2).sum() / 2.0) + 1e-12

    def f(t):
        t = np.asarray(t, dtype=np.float64)
        out = np.zeros_like(t)
        for a, ph, fr in zip(amps, phases, freqs):
            out = out + a * np.sin(2.0 * np.pi * fr * t + ph)
        return out / rms

    return f
