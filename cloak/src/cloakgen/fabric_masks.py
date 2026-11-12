"""Tatter / hole alpha masks for the fabric atlas (system Python: numpy + scipy + opencv).

Each island is generated in its own panel parameter space (rows = t "down", cols = s "across")
using metric (stud) coordinates measured on the actual geometry, so tears, holes and frayed
edges have consistent physical sizes everywhere, then rotated/placed into the atlas.
"""
import cv2
import numpy as np
from scipy import ndimage

from . import fabric_geom, layout
from .noise import Perlin2D, smoothstep

SS = 2  # supersampling factor for anti-aliased edges


# --------------------------------------------------------------------------- param / metric helpers
def island_param_shape(isl, ss=1):
    x0, y0, w, h = isl["rect"]
    return (w * ss, h * ss) if isl["rot"] else (h * ss, w * ss)  # (n_t, n_s)


def place(atlas, isl, img):
    x0, y0, w, h = isl["rect"]
    if isl["rot"]:
        img = np.rot90(img, 1)
    atlas[y0:y0 + h, x0:x0 + w] = img


def extended_grid(part):
    rows, cols = part["rows"], part["cols"]
    P = part["verts"].reshape(rows, cols, 3)
    if part["wrap_cols"]:
        P = np.concatenate([P, P[:, :1]], axis=1)
    if part["wrap_rows"]:
        P = np.concatenate([P, P[:1]], axis=0)
    return P


def metric_maps(part, n_t, n_s):
    """Metric coordinates on the panel at pixel centres.

    X: arc length across (centred on s=0.5), Y: arc length down from t=0,
    W: full row width, H: full column length (all in studs)."""
    P = extended_grid(part)
    R, C = P.shape[:2]
    seg_c = np.linalg.norm(np.diff(P, axis=1), axis=2)
    cum_c = np.concatenate([np.zeros((R, 1)), np.cumsum(seg_c, axis=1)], axis=1)
    seg_r = np.linalg.norm(np.diff(P, axis=0), axis=2)
    cum_r = np.concatenate([np.zeros((1, C)), np.cumsum(seg_r, axis=0)], axis=0)
    mid = np.array([np.interp((C - 1) / 2.0, np.arange(C), cum_c[r]) for r in range(R)])
    Xg = cum_c - mid[:, None]
    Wg = np.repeat(cum_c[:, -1:], C, axis=1)
    Hg = np.repeat(cum_r[-1:, :], R, axis=0)
    t = (np.arange(n_t) + 0.5) / n_t
    s = (np.arange(n_s) + 0.5) / n_s
    gr, gc = np.meshgrid(t * (R - 1), s * (C - 1), indexing="ij")
    out = []
    for g in (Xg, cum_r, Wg, Hg):
        out.append(ndimage.map_coordinates(g, [gr, gc], order=1, mode="nearest"))
    S, T = np.meshgrid(s, t)
    return dict(X=out[0], Y=out[1], W=out[2], H=out[3], S=S, T=T)


# --------------------------------------------------------------------------- tatter primitives
def hem_strips(Xh, Y, Yend, rng, x_lo, x_hi, width, notch, trim, gamma, fork_prob=0.2, short_prob=0.15,
               short_extra=(0.2, 0.5)):
    """Boolean fabric mask for a hem made of tapering torn strips."""
    edges = [x_lo - rng.uniform(0.0, width[1])]
    while edges[-1] < x_hi:
        edges.append(edges[-1] + rng.uniform(*width))
    edges = np.asarray(edges)
    n = len(edges) - 1
    cen = 0.5 * (edges[1:] + edges[:-1])
    hw = 0.5 * (edges[1:] - edges[:-1])
    tr = rng.uniform(*trim, n)
    short = rng.random(n) < short_prob
    tr[short] += rng.uniform(*short_extra, short.sum())
    nt = rng.uniform(*notch, n + 1)
    gm = rng.uniform(*gamma, n)
    fork = rng.random(n) < fork_prob
    idx = np.clip(np.searchsorted(edges, Xh) - 1, 0, n - 1)
    c = cen[idx]
    h = hw[idx]
    ytip = Yend - tr[idx]
    L = np.where(Xh < c, nt[idx], nt[idx + 1])
    q = np.clip((ytip - Y) / L, 0.0, 1.0)
    allowed = h * q ** gm[idx]
    fabric = (Y <= ytip - L) | ((np.abs(Xh - c) <= allowed) & (Y <= ytip))
    fl = 0.4 * np.minimum(nt[idx], nt[idx + 1])
    qf = np.clip((ytip - Y) / fl, 0.0, 1.0)
    cut = fork[idx] & (np.abs(Xh - c) < 0.22 * h * (1.0 - qf)) & (Y > ytip - fl)
    return fabric & ~cut


def add_holes(fabric, M, rng, count, density, rx, aspect, noise, jag=0.35, tilt=0.18):
    """Cut ragged elliptical holes. density(T) in [0,1] steers placement along the panel."""
    X, Y, T = M["X"], M["Y"], M["T"]
    n_t, n_s = X.shape
    dXds = np.gradient(X, axis=1)
    dYdt = np.gradient(Y, axis=0)
    placed = 0
    tries = 0
    while placed < count and tries < count * 60:
        tries += 1
        i = rng.integers(0, n_t)
        j = rng.integers(0, n_s)
        if rng.random() > density(T[i, j]) or not fabric[i, j]:
            continue
        r_x = rng.uniform(*rx)
        r_y = r_x * rng.uniform(*aspect)
        px_x = r_x / max(abs(dXds[i, j]), 1e-6)
        px_y = r_y / max(abs(dYdt[i, j]), 1e-6)
        hi = int(px_y * 1.8) + 3
        hj = int(px_x * 1.8 + px_y * abs(tilt) * 2) + 3
        i0, i1 = max(i - hi, 0), min(i + hi + 1, n_t)
        j0, j1 = max(j - hj, 0), min(j + hj + 1, n_s)
        dx = X[i0:i1, j0:j1] - X[i, j]
        dy = Y[i0:i1, j0:j1] - Y[i, j]
        a = rng.normal(0.0, tilt)
        ca, sa = np.cos(a), np.sin(a)
        u = (ca * dx + sa * dy) / r_x
        v = (-sa * dx + ca * dy) / r_y
        nn = noise(X[i0:i1, j0:j1] * 30.0 + 100 * placed, Y[i0:i1, j0:j1] * 30.0)
        nn2 = noise(X[i0:i1, j0:j1] * 45.0, Y[i0:i1, j0:j1] * 35.0 + 50)
        inside = (u * u + v * v) < (1.0 + jag * nn + 0.15 * nn2) ** 2
        fabric[i0:i1, j0:j1] &= ~inside
        placed += 1
    return fabric


def ragged_side_edges(fabric, M, rng, noise, depth=0.02, notch_count=6, notch_depth=(0.04, 0.12), lo=True, hi=True,
                      notch_from=0.0):
    X, Y, W = M["X"], M["Y"], M["W"]
    d_lo = X + W / 2.0
    d_hi = W / 2.0 - X
    base = depth * (1.2 + 0.8 * noise(Y * 12.0, 3.1)) + 0.012 * noise(Y * 30.0, 7.7)
    keep = np.ones_like(fabric)
    y_lo = Y.min() + notch_from * (Y.max() - Y.min())
    for side, d in ((lo, d_lo), (hi, d_hi)):
        if not side:
            continue
        cut = d < base
        for _ in range(notch_count):
            yc = rng.uniform(y_lo, Y.max())
            ln = rng.uniform(0.04, 0.16)
            dp = rng.uniform(*notch_depth)
            prof = np.clip(1.0 - np.abs(Y - yc) / ln, 0.0, 1.0) ** 0.8
            cut |= d < dp * prof
        keep &= ~cut
    return fabric & keep


def keep_anchored(fabric, anchor_rows):
    """Drop fabric fragments that are not connected to the panel's attached edge."""
    lab, n = ndimage.label(fabric)
    if n == 0:
        return fabric
    top = np.unique(lab[:anchor_rows])
    top = top[top > 0]
    return np.isin(lab, top)


def jitter(M, noise, ax=0.03, ay=0.03, fx=9.0, fy=3.0, fine=0.012):
    # the fine term stays above ~0.04 stud wavelength so it survives the 1024 px Roblox texture
    X, Y = M["X"], M["Y"]
    Xj = X + ax * noise.fbm(X * fx, Y * fy, 3) + fine * noise(X * 22.0, Y * 8.0)
    Yj = Y + ay * noise.fbm(X * fx * 0.7 + 31.0, Y * fy * 0.7, 3) + fine * noise(X * 20.0 + 9.0, Y * 14.0)
    return Xj, Yj


# --------------------------------------------------------------------------- island recipes
def mask_cape(part, n_t, n_s, rng, noise):
    M = metric_maps(part, n_t, n_s)
    Xj, Yj = jitter(M, noise, 0.06, 0.06, 7.0, 2.5, 0.026)
    W_hem = M["W"][-1, n_s // 2]
    Xh = (M["S"] - 0.5) * W_hem + (Xj - M["X"])
    fabric = hem_strips(Xh, Yj, M["H"], rng, -W_hem / 2, W_hem / 2,
                        width=(0.22, 0.48), notch=(1.00, 2.10), trim=(0.0, 0.50), gamma=(0.6, 1.05),
                        fork_prob=0.3, short_prob=0.25, short_extra=(0.15, 0.6))
    fabric = ragged_side_edges(fabric, M, rng, noise, depth=0.010, notch_count=7, notch_depth=(0.03, 0.09),
                               notch_from=0.4)
    dens = lambda t: float(smoothstep(0.15, 0.85, t)) ** 1.2
    fabric = add_holes(fabric, M, rng, 55, dens, (0.035, 0.095), (1.4, 2.6), noise)
    fabric = add_holes(fabric, M, rng, 22, dens, (0.014, 0.032), (1.2, 2.2), noise)
    fabric = add_holes(fabric, M, rng, 10, lambda t: float(smoothstep(0.5, 0.8, t)), (0.010, 0.018), (4.0, 8.0), noise,
                       jag=0.25, tilt=0.08)
    return keep_anchored(fabric, max(4, n_t // 20)), M


def capelet_length(x):
    xs = np.array([-1.80, -1.66, -1.48, -1.20, -0.80, -0.42, -0.25, 0.0, 0.25, 0.40, 0.90, 1.20, 1.42])
    ls = np.array([0.30, 0.95, 1.45, 2.05, 2.25, 2.10, 1.15, 0.90, 1.00, 1.35, 1.35, 1.00, 0.45])
    return np.interp(x, xs, ls)


def mask_capelet(part, n_t, n_s, rng, noise):
    M = metric_maps(part, n_t, n_s)
    Xj, Yj = jitter(M, noise, 0.03, 0.04, 9.0, 3.0, 0.012)
    x_world = fabric_geom.CAPELET_X[0] + (fabric_geom.CAPELET_X[1] - fabric_geom.CAPELET_X[0]) * M["S"]
    Yend = capelet_length(x_world) + 0.12 * noise(x_world * 3.0, 5.5)
    fabric = hem_strips(Xj, Yj, Yend, rng, M["X"].min() - 0.1, M["X"].max() + 0.1,
                        width=(0.08, 0.19), notch=(0.30, 0.90), trim=(0.0, 0.22), gamma=(0.3, 0.7), fork_prob=0.2)
    fabric = ragged_side_edges(fabric, M, rng, noise, depth=0.02, notch_count=3)
    fabric = add_holes(fabric, M, rng, 9, lambda t: float(smoothstep(0.2, 0.7, t)), (0.02, 0.045), (1.3, 2.4), noise)
    return keep_anchored(fabric, 3 * SS), M


def mask_drape(part, n_t, n_s, rng, noise):
    M = metric_maps(part, n_t, n_s)
    Xj, Yj = jitter(M, noise, 0.025, 0.05, 10.0, 2.0, 0.012)
    fabric = hem_strips(Xj, Yj, M["H"], rng, M["X"].min() - 0.05, M["X"].max() + 0.05,
                        width=(0.13, 0.26), notch=(0.9, 1.7), trim=(0.0, 0.45), gamma=(0.3, 0.7), fork_prob=0.3)
    fabric = ragged_side_edges(fabric, M, rng, noise, depth=0.02, notch_count=8, notch_depth=(0.04, 0.10))
    fabric = add_holes(fabric, M, rng, 10, lambda t: float(smoothstep(0.35, 0.8, t)), (0.02, 0.05), (1.5, 2.8), noise)
    return keep_anchored(fabric, 3 * SS), M


def mask_hood(part, n_t, n_s, rng, noise):
    M = metric_maps(part, n_t, n_s)
    Xj, Yj = jitter(M, noise, 0.025, 0.025, 10.0, 6.0, 0.01)
    fabric = hem_strips(Xj, Yj, M["H"], rng, M["X"].min() - 0.1, M["X"].max() + 0.1,
                        width=(0.07, 0.18), notch=(0.10, 0.34), trim=(0.0, 0.09), gamma=(0.5, 1.0), fork_prob=0.15,
                        short_extra=(0.03, 0.1))
    fabric = ragged_side_edges(fabric, M, rng, noise, depth=0.03, notch_count=4)
    fabric = add_holes(fabric, M, rng, 6, lambda t: float(smoothstep(0.3, 0.8, t)), (0.015, 0.035), (1.2, 2.0), noise)
    return keep_anchored(fabric, 3 * SS), M


def mask_banner(part, n_t, n_s, rng, noise, n_points):
    M = metric_maps(part, n_t, n_s)
    Xj, Yj = jitter(M, noise, 0.008, 0.02, 20.0, 4.0, 0.006)
    W = M["W"]
    fray = 0.008 + 0.006 * noise(M["Y"] * 25.0, 1.0) + 0.004 * noise(M["Y"] * 45.0, 2.0)
    fabric = np.abs(Xj) < W / 2.0 - fray
    for _ in range(9):  # bites out of the long edges
        yc = rng.uniform(0.2, 1.0) * M["H"].max()
        ln = rng.uniform(0.03, 0.10)
        dp = rng.uniform(0.02, 0.06)
        prof = np.clip(1.0 - np.abs(M["Y"] - yc) / ln, 0.0, 1.0)
        side = 1 if rng.random() < 0.5 else -1
        fabric &= ~((side * Xj > W / 2.0 - dp * prof) & (prof > 0))
    wmin = W.min()
    strips = hem_strips(Xj, Yj, M["H"], rng, -wmin / 2, wmin / 2,
                        width=(wmin / (n_points + 0.6), wmin / (n_points - 0.4)), notch=(0.22, 0.48),
                        trim=(0.0, 0.10), gamma=(0.6, 0.95), fork_prob=0.0, short_prob=0.0)
    fabric &= strips
    fabric = add_holes(fabric, M, rng, 3, lambda t: float(smoothstep(0.5, 0.9, t)), (0.01, 0.025), (1.3, 2.0), noise)
    return keep_anchored(fabric, 3 * SS), M


def mask_fringe(part, n_t, n_s, rng, noise):
    M = metric_maps(part, n_t, n_s)
    Xj, Yj = jitter(M, noise, 0.01, 0.01, 30.0, 10.0, 0.006)
    fabric = hem_strips(Xj, Yj, M["H"] * 1.02, rng, M["X"].min() - 0.05, M["X"].max() + 0.05,
                        width=(0.03, 0.16), notch=(0.04, 0.26), trim=(0.0, 0.09), gamma=(0.45, 1.1),
                        fork_prob=0.15, short_prob=0.32, short_extra=(0.02, 0.12))
    fabric = ragged_side_edges(fabric, M, rng, noise, depth=0.03, notch_count=0)
    return keep_anchored(fabric, max(2, n_t // 8)), M


def build_masks(parts, seed=1234):
    """Returns (alpha_atlas float32 [2048,2048] in 0..1, {island: metric maps at 1x})."""
    by_island = {p["island"]: p for p in parts}
    atlas = np.zeros((layout.ATLAS_PX, layout.ATLAS_PX), dtype=np.float32)
    metrics = {}
    noise = Perlin2D(seed)
    for k, (name, isl) in enumerate(layout.FABRIC_ISLANDS.items()):
        part = by_island.get(name)
        n_t, n_s = island_param_shape(isl, SS)
        rng = np.random.default_rng(seed + 17 * k)
        if part is None:
            continue
        if name == "cape":
            fab, M = mask_cape(part, n_t, n_s, rng, noise)
        elif name == "capelet":
            fab, M = mask_capelet(part, n_t, n_s, rng, noise)
        elif name == "drape":
            fab, M = mask_drape(part, n_t, n_s, rng, noise)
        elif name == "hood":
            fab, M = mask_hood(part, n_t, n_s, rng, noise)
        elif name == "banner_front":
            fab, M = mask_banner(part, n_t, n_s, rng, noise, 3)
        elif name == "banner_back":
            fab, M = mask_banner(part, n_t, n_s, rng, noise, 4)
        elif name.startswith("fringe"):
            fab, M = mask_fringe(part, n_t, n_s, rng, noise)
        else:
            fab = np.ones((n_t, n_s), dtype=bool)
            M = metric_maps(part, n_t, n_s)
        alpha = cv2.resize(fab.astype(np.float32), (n_s // SS, n_t // SS), interpolation=cv2.INTER_AREA)
        place(atlas, isl, alpha)
        metrics[name] = {key: cv2.resize(val.astype(np.float32), (n_s // SS, n_t // SS), interpolation=cv2.INTER_AREA)
                         for key, val in M.items()}
    return atlas, metrics
