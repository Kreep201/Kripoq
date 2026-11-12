"""Fabric atlas PBR maps: colour + alpha, normal, roughness and the rune emissive mask."""
import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import layout
from .fabric_masks import island_param_shape, place
from .noise import Perlin2D, smoothstep
from .tex_common import clamp01, lerp, normal_from_height, rgb

N = layout.ATLAS_PX
TUBES = ("roll", "swag")

PAL = dict(
    black=rgb("#211e25"), navy=rgb("#2a2833"), plum=rgb("#4a3f4b"), dusty=rgb("#6f6470"),
    edge_tan=rgb("#8e7c6f"), edge_pale=rgb("#b19f8f"), speck=rgb("#b0a69d"),
    banner=rgb("#2e2e3a"), banner_dark=rgb("#212029"), rune=rgb("#e6e0ff"), glow=rgb("#9c8ed8"),
)


def unplace(atlas, isl):
    x0, y0, w, h = isl["rect"]
    reg = atlas[y0:y0 + h, x0:x0 + w]
    return np.rot90(reg, -1) if isl["rot"] else reg


def island_fields(alpha, metrics):
    """Per-pixel atlas fields: metric coords, panel t, pixel size (studs), distance to torn edges."""
    MX = np.zeros((N, N), np.float32)
    MY = np.zeros((N, N), np.float32)
    TT = np.zeros((N, N), np.float32)
    PS = np.full((N, N), 0.004, np.float32)
    ED = np.full((N, N), 1.0, np.float32)
    ISL = np.full((N, N), -1, np.int16)
    for k, (name, isl) in enumerate(layout.FABRIC_ISLANDS.items()):
        X = metrics[name + "__X"]
        Y = metrics[name + "__Y"]
        place(MX, isl, X + 37.0 * k)
        place(MY, isl, Y + 23.0 * k)
        place(TT, isl, metrics[name + "__T"])
        place(ISL, isl, np.full(X.shape, k, np.int16))
        ps = np.sqrt(np.abs(np.gradient(X, axis=1)) * np.abs(np.gradient(Y, axis=0))) + 1e-5
        ps = cv2.GaussianBlur(ps.astype(np.float32), (0, 0), 3)
        place(PS, isl, ps)
        if name.startswith(TUBES):
            continue
        fab = unplace(alpha, isl) > 0.5
        n_t, n_s = fab.shape
        pad = 10
        fp = np.zeros((n_t + 2 * pad, n_s + 2 * pad), np.uint8)
        fp[pad:-pad, pad:-pad] = fab
        fp[:pad, pad:-pad] = fab[:1]  # the attached top edge is not a torn edge
        d = cv2.distanceTransform(fp, cv2.DIST_L2, 5)[pad:-pad, pad:-pad]
        place(ED, isl, d * ps)
    return MX, MY, TT, PS, ED, ISL


# --------------------------------------------------------------------------- runes
def draw_rune(name, n_t, n_s, length, width, rng):
    """Rune painted in panel space (rows = down the banner). Returns (core, glow) float maps."""
    ss = 2
    W, H = n_s * ss, n_t * ss
    core = Image.new("L", (W, H), 0)
    faint = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(core)
    df = ImageDraw.Draw(faint)
    px = lambda X, Y: ((X / width + 0.5) * W, Y / length * H)
    lw = max(int(0.0115 / width * W), 2)

    def line(a, b, w=lw, dd=d):
        dd.line([px(*a), px(*b)], fill=255, width=w)

    if name == "banner_front":
        top, cy, r, bar, eye, bottom = 0.42, 0.92, 0.082, 1.22, 1.43, 1.66
    else:
        top, cy, r, bar, eye, bottom = 0.40, 1.08, 0.10, 1.42, 1.66, 1.92
    # keep the circle round on screen: radius in Y must use the same stud scale as X
    line((0, top), (0, cy - r))
    line((0, cy + r), (0, bottom))
    x0, y0 = px(-r, cy - r)
    x1, _ = px(r, cy - r)
    _, y1 = px(-r, cy + r)
    d.ellipse([x0, y0, x1, y1], outline=255, width=lw)
    line((-r * 1.55, cy), (-r * 0.55, cy))
    line((r * 0.55, cy), (r * 1.55, cy))
    line((-r * 0.25, cy), (r * 0.25, cy))
    line((-r * 1.55, cy - 0.025), (-r * 1.55, cy + 0.025), max(lw - 1, 2))
    line((r * 1.55, cy - 0.025), (r * 1.55, cy + 0.025), max(lw - 1, 2))
    line((-0.045, bar), (0.045, bar))
    # small arrow / tick glyphs above the circle
    line((-0.03, top + 0.03), (0, top + 0.07), max(lw - 1, 2))
    line((0.03, top + 0.03), (0, top + 0.07), max(lw - 1, 2))
    line((-0.022, top + 0.16), (0.022, top + 0.16), max(lw - 1, 2))
    # eye glyph
    ex0, ey0 = px(-0.04, eye - 0.022)
    ex1, ey1 = px(0.04, eye + 0.022)
    d.ellipse([ex0, ey0, ex1, ey1], outline=255, width=max(lw - 1, 2))
    cx_, cy_ = px(0, eye)
    rr = lw * 0.9
    d.ellipse([cx_ - rr, cy_ - rr, cx_ + rr, cy_ + rr], fill=255)
    # forked foot
    line((0, bottom), (-0.03, bottom + 0.05), max(lw - 1, 2))
    line((0, bottom), (0.03, bottom + 0.05), max(lw - 1, 2))
    # faint, non-glowing glyphs scattered near the edges
    for _ in range(7):
        gx = rng.choice([-1, 1]) * rng.uniform(0.25, 0.36) * width
        gy = rng.uniform(top - 0.15, bottom + 0.1)
        s = 0.022
        kind = rng.integers(0, 3)
        if kind == 0:
            line((gx, gy - s), (gx, gy + s), max(lw - 2, 2), df)
            line((gx, gy - s * 0.2), (gx + s * 0.8, gy - s), max(lw - 2, 2), df)
        elif kind == 1:
            line((gx - s * 0.7, gy - s), (gx + s * 0.7, gy + s), max(lw - 2, 2), df)
            line((gx + s * 0.7, gy - s), (gx - s * 0.7, gy + s), max(lw - 2, 2), df)
        else:
            line((gx, gy - s), (gx, gy + s), max(lw - 2, 2), df)
            line((gx - s * 0.6, gy), (gx + s * 0.6, gy), max(lw - 2, 2), df)
    core = np.asarray(core, np.float32) / 255.0
    faint = np.asarray(faint, np.float32) / 255.0
    core = cv2.resize(core, (n_s, n_t), interpolation=cv2.INTER_AREA)
    faint = cv2.resize(faint, (n_s, n_t), interpolation=cv2.INTER_AREA)
    sig = 0.022 / width * n_s
    glow = cv2.GaussianBlur(core, (0, 0), sig) * 1.8
    return clamp01(core), clamp01(glow), faint


# --------------------------------------------------------------------------- main
def build(alpha, metrics, ao, seed=7):
    noise = Perlin2D(seed)
    MX, MY, TT, PS, ED, ISL = island_fields(alpha, metrics)
    names = list(layout.FABRIC_ISLANDS)
    is_tube = np.isin(ISL, [names.index(n) for n in names if n.startswith(TUBES)])
    is_banner = np.isin(ISL, [names.index("banner_front"), names.index("banner_back")])
    is_cape_like = np.isin(ISL, [names.index(n) for n in ("cape", "drape", "capelet", "hood")])

    n1 = noise.fbm(MX * 1.1, MY * 1.1, 4)
    n2 = noise.fbm(MX * 4.0 + 5.0, MY * 4.0, 4)
    n3 = noise.fbm(MX * 16.0, MY * 2.2 + 9.0, 3)  # streaks along the hang direction
    n4 = noise(MX * 60.0, MY * 60.0)
    wear_t = np.where(is_cape_like, smoothstep(0.3, 0.95, TT), 0.15).astype(np.float32)

    col = lerp(PAL["black"], PAL["navy"], clamp01(0.5 + 0.9 * n1))
    col = lerp(col, PAL["plum"], clamp01(0.45 + 1.1 * n2) * 0.38)
    dust = clamp01(0.15 + 0.9 * n2 + 0.5 * wear_t - 0.25)
    col = lerp(col, PAL["dusty"], dust * 0.42 + 0.38 * wear_t * clamp01(0.5 + n1))
    col *= (1.0 + 0.07 * n3)[..., None]
    weave = 0.5 + 0.5 * np.sin(2 * np.pi * MX / 0.03) * np.sin(2 * np.pi * MY / 0.03)
    col *= (0.96 + 0.07 * weave)[..., None]

    # banners: cooler grey-blue cloth
    bcol = lerp(PAL["banner_dark"], PAL["banner"], clamp01(0.5 + 0.8 * n1 + 0.3 * n2))
    bcol *= (1.0 + 0.12 * n3)[..., None]
    col = np.where(is_banner[..., None], bcol, col)

    # torn edges and hole rims: dusty tan fraying (signature look of the concept)
    edge_noise = 0.65 + 0.35 * noise.fbm(MX * 25.0, MY * 25.0, 3)
    ew_scale = np.where(is_banner, 0.045, 0.034)
    ew_gain = np.where(is_banner, 1.0, 0.62)
    ew = np.exp(-ED / ew_scale) * edge_noise * ew_gain
    ew2 = np.exp(-ED / 0.008) * ew_gain
    col = lerp(col, PAL["edge_tan"], clamp01(ew * (0.75 + 0.25 * wear_t)) * 0.8)
    col = lerp(col, PAL["edge_pale"], clamp01(ew2 * edge_noise) * 0.55)
    # lint / dust specks
    sp = noise(MX * 110.0 + 3.0, MY * 110.0)
    specks = clamp01((sp - 0.62) * 6.0) * clamp01(0.5 + n2)
    col = lerp(col, PAL["speck"], specks * 0.55)

    # baked occlusion
    ao2 = cv2.resize(ao, (N, N), interpolation=cv2.INTER_LINEAR)
    ao2 = np.where(ao2 <= 0.0, 1.0, ao2)
    col *= (0.5 + 0.5 * clamp01(ao2) ** 0.85)[..., None]

    # runes on both banners
    emis = np.zeros((N, N), np.float32)
    rng = np.random.default_rng(seed)
    for name, length, width in (("banner_front", 2.24, 0.34), ("banner_back", 2.70, 0.44)):
        isl = layout.FABRIC_ISLANDS[name]
        n_t, n_s = island_param_shape(isl)
        core, glow, faint = draw_rune(name, n_t, n_s, length, width, rng)
        Rc = np.zeros((N, N), np.float32)
        Rg = np.zeros((N, N), np.float32)
        Rf = np.zeros((N, N), np.float32)
        place(Rc, isl, core)
        place(Rg, isl, glow)
        place(Rf, isl, faint)
        wear = clamp01(0.8 + 0.5 * noise.fbm(MX * 30.0, MY * 30.0, 3))
        Rc *= wear
        col = lerp(col, PAL["glow"], clamp01(Rg) * 0.35)
        col = lerp(col, PAL["rune"], clamp01(Rc) * 0.92)
        col = lerp(col, PAL["edge_tan"], Rf * 0.35)
        emis = np.maximum(emis, clamp01(Rc * 1.0 + Rg * 0.45))

    # roughness
    rough = 0.9 + 0.05 * n2 - 0.06 * clamp01(ew) - 0.25 * clamp01(emis)
    rough = np.where(is_banner, rough - 0.04, rough)

    # height (studs) -> normal
    wrinkle = noise.fbm(MX * 5.0, MY * 1.8, 4)
    fibre = noise(MX * 140.0, MY * 35.0)
    h = (0.002 * weave + 0.02 * wrinkle + 0.0008 * n4
         + 0.0012 * fibre * clamp01(ew * 1.5) + 0.0025 * clamp01(ew2))
    h = np.where(is_tube, h + 0.004 * noise.fbm(MX * 9.0, MY * 9.0, 3), h)
    nrm = normal_from_height(h.astype(np.float32), PS, 1.0)
    col = clamp01(col)
    return col, nrm, clamp01(rough), emis
