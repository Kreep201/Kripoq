"""Hardware atlas PBR maps: aged iron/silver parts, engraved brooch ring, tag, chain links and
stitched leather straps."""
import numpy as np

from . import layout
from .layout import RING_SECTION, STRAP_T, STRAP_W
from .noise import Perlin2D
from .tex_common import clamp01, lerp, normal_from_height, rgb

N = layout.ATLAS_PX
HW = layout.HW_ISLANDS

MET = dict(clean=rgb("#99958f"), dark=rgb("#35312e"), rust=rgb("#5e4232"), edge=rgb("#d2cdc3"))
LEA = dict(base=rgb("#3d2a1f"), dark=rgb("#22170f"), worn=rgb("#6b4f3d"), edge=rgb("#8d6c52"))


def section_ranges(w, h, b):
    """t-ranges of the faces of a chamfered-rectangle sweep section (see hardware.chamfer_rect_section)."""
    c = b * np.sqrt(2.0)
    # order: p0->p1 chamfer, p1->p2 (+B face), chamfer, p3->p4 (-N face), chamfer, p5->p6 (-B face), chamfer, p7->p0 (+N face)
    seg = [c, w - 2 * b, c, h - 2 * b, c, w - 2 * b, c, h - 2 * b]
    cum = np.r_[0, np.cumsum(seg)] / sum(seg)
    return dict(plusB=(cum[1], cum[2]), minusN=(cum[3], cum[4]), minusB=(cum[5], cum[6]), plusN=(cum[7], cum[8]))


def region(name):
    x0, y0, w, h = HW[name]
    return slice(y0, y0 + h), slice(x0, x0 + w), w, h


def build(ao, edge, cov, info, seed=11):
    noise = Perlin2D(seed)
    yy, xx = np.mgrid[0:N, 0:N].astype(np.float32)
    px = 0.0011  # nominal studs per pixel for procedural scale in the smart-unwrapped region
    X, Y = xx * px, yy * px
    aoc = clamp01(np.where(ao <= 0, 1.0, ao))
    nA = noise.fbm(X * 11.0, Y * 11.0, 4)
    nB = noise.fbm(X * 45.0 + 3.0, Y * 45.0, 3)
    nC = noise(X * 420.0, Y * 55.0)

    # ------------------------------------------------------------------ metal (default everywhere)
    tarnish = clamp01((1.0 - aoc) * 2.1 + 0.32 * nA + 0.16)
    rust = clamp01((nB - 0.32) * 3.0) * clamp01(tarnish * 1.6)
    e = clamp01(edge * 1.3) * clamp01((aoc - 0.5) * 3.0)
    scratch = clamp01((nC - 0.55) * 4.0) * (1.0 - tarnish)
    groove = np.zeros((N, N), np.float32)
    height = 0.00045 * noise.fbm(X * 160.0, Y * 160.0, 2)

    # engraved ticks on the brooch ring (both flat faces)
    ys, xs, w, h = region("ring")
    s = (xx[ys, xs] - xs.start) / w
    t = (yy[ys, xs] - ys.start) / h
    rr = section_ranges(*RING_SECTION)
    g = np.zeros_like(s)
    for lo, hi in (rr["minusB"], rr["plusB"]):
        face = (t > lo) & (t < hi)
        a = (t - lo) / (hi - lo)  # 0 = inner edge, 1 = outer edge
        k = s * 48.0
        d = np.abs(k - np.round(k))
        major = (np.round(k) % 4) == 0
        tick = (d < 0.09) & np.where(major, (a > 0.18) & (a < 0.9), (a > 0.55) & (a < 0.9))
        ringline = np.abs(a - 0.36) < 0.035
        g = np.maximum(g, (face & (tick | ringline)).astype(np.float32))
    groove[ys, xs] = g
    height[ys, xs] -= 0.0012 * g

    # tag face: recessed panel, rivets and a small engraved sigil
    ys, xs, w, h = region("tag_face")
    u = (xx[ys, xs] - xs.start) / w - 0.5
    v = (yy[ys, xs] - ys.start) / h
    inset = (np.abs(u) < 0.36) & (v > 0.1) & (v < 0.9)
    rivet = np.zeros_like(u)
    for rv in (0.06, 0.94):
        r = np.sqrt((u / 0.09) ** 2 + ((v - rv) / 0.05) ** 2)
        rivet = np.maximum(rivet, clamp01(1.0 - r))
    cu, cv = u / 0.36, (v - 0.5) / 0.22
    rad = np.sqrt(cu ** 2 + cv ** 2)
    sig = ((np.abs(rad - 0.55) < 0.07) | ((np.abs(cu) < 0.05) & (np.abs(cv) < 1.0)) |
           ((np.abs(cv) < 0.04) & (np.abs(cu) < 0.85))).astype(np.float32) * inset
    height[ys, xs] += -0.0015 * inset + 0.002 * rivet ** 0.5 - 0.0008 * sig
    groove[ys, xs] = np.maximum(groove[ys, xs], 0.6 * inset.astype(np.float32) + 0.4 * sig)
    tarnish[ys, xs] = clamp01(tarnish[ys, xs] + 0.25 * inset)

    # chain link template: dark inside of the loop, polished outer bend
    ys, xs, w, h = region("link")
    t = (yy[ys, xs] - ys.start) / h
    inner = np.maximum(0.0, np.cos(2 * np.pi * (t - 0.375)))
    outer = np.maximum(0.0, np.cos(2 * np.pi * (t - 0.875))) ** 4
    tarnish[ys, xs] = clamp01(0.2 + 0.6 * inner + 0.25 * nA[ys, xs])
    e[ys, xs] = 0.55 * outer

    tarnish = clamp01(tarnish + 0.6 * groove)
    col = lerp(MET["clean"], MET["dark"], tarnish * 0.92)
    col = lerp(col, MET["rust"], rust * 0.6)
    col = lerp(col, MET["edge"], clamp01(e * 0.8 + scratch * 0.3))
    metal = clamp01(0.96 - 0.55 * tarnish - 0.6 * rust)
    rough = clamp01(0.32 + 0.4 * tarnish + 0.25 * rust - 0.14 * e + 0.06 * nA)

    # ------------------------------------------------------------------ leather straps
    sr = section_ranges(STRAP_T, STRAP_W, 0.006)
    top_lo, top_hi = sr["plusN"]
    for name in ("strap_a1", "strap_a2", "strap_b"):
        ys, xs, w, h = region(name)
        L = info["strap_lengths"][name]
        s = (xx[ys, xs] - xs.start) / w
        t = (yy[ys, xs] - ys.start) / h
        along = s * L
        across = t * 2 * (STRAP_T + STRAP_W)
        grain = noise(along * 150.0, across * 150.0 + 40.0)
        crease = noise.fbm(along * 28.0, across * 3.0 + 11.0, 3)
        blotch = noise.fbm(along * 3.0, across * 3.0 + 77.0, 3)
        on_top = (t >= top_lo) & (t <= top_hi)
        a = np.clip((t - top_lo) / (top_hi - top_lo), 0, 1)
        side = ~on_top & ~((t >= sr["minusN"][0]) & (t <= sr["minusN"][1]))
        edge_w = np.where(on_top, np.exp(-np.minimum(a, 1 - a) / 0.06), 0.0) + side * 0.9
        stitch_row = on_top & ((np.abs(a - 0.1) < 0.025) | (np.abs(a - 0.9) < 0.025))
        stitch = stitch_row & ((along / 0.034) % 1.0 < 0.45)
        lc = lerp(LEA["base"], LEA["dark"], clamp01(0.35 + 0.8 * crease) * 0.55)
        lc = lerp(lc, LEA["worn"], clamp01(blotch * 0.9 + 0.2) * 0.35)
        lc = lerp(lc, LEA["edge"], clamp01(edge_w * (0.7 + 0.3 * grain)))
        lc = lerp(lc, LEA["dark"] * 0.6, stitch.astype(np.float32) * 0.9)
        lc *= (0.95 + 0.08 * grain)[..., None]
        lr = clamp01(0.62 + 0.12 * grain + 0.1 * crease - 0.12 * edge_w)
        lh = 0.0003 * grain + 0.0016 * crease - 0.0012 * stitch
        lm = np.zeros_like(lr)
        if name == "strap_a1":  # domed studs down the back strap
            k = along / 0.2
            dk = (k - np.round(k)) * 0.2
            stud_r = np.sqrt(dk ** 2 + ((a - 0.5) * (top_hi - top_lo) * 2 * (STRAP_T + STRAP_W)) ** 2)
            stud = on_top & (stud_r < 0.013) & (along > 0.15) & (along < 2.3)
            dome = clamp01(1.0 - stud_r / 0.013) * stud
            lc = lerp(lc, MET["clean"] * 0.85, stud.astype(np.float32))
            lm = np.where(stud, 0.9, lm)
            lr = np.where(stud, 0.38, lr)
            lh = lh + 0.004 * np.sqrt(dome)
        col[ys, xs] = lc
        rough[ys, xs] = lr
        metal[ys, xs] = lm
        height[ys, xs] = lh

    aoc_col = 0.55 + 0.45 * aoc
    col *= aoc_col[..., None]
    nrm = normal_from_height(height.astype(np.float32), 0.0012, 1.0)
    covered = cov > 0.5
    return clamp01(col), nrm, rough, metal, covered
