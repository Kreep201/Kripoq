"""Procedural geometry for the cloth parts of the cloak.

Coordinates (Blender): 1 unit = 1 Roblox stud, Z up, the character faces -Y, feet at z=0.
R6 body: torso x[-1,1] y[-0.5,0.5] z[2,4]; arms x[+-1,+-2]; legs z[0,2]; head ~r0.6 z[3.9,5.1].

Every part is returned as a dict with:
  verts (V,3) float, faces (F,4) int quads, s/t per vertex (panel parameters used for UVs),
  island (atlas island name), wrap flags so the builder can generate seam-correct UVs.
"""
import numpy as np

from .noise import smoothstep, smooth_random_1d

TORSO_TOP = 4.0
BODY_HALF_X = 2.0   # torso + arms are one 4 x 1 block between z=2 and z=4
BODY_HALF_Y = 0.5


# --------------------------------------------------------------------------- helpers
def centripetal_catmull_rom(P, samples_per_seg=24, alpha=0.5):
    """P: (K, 3). Returns a dense polyline passing through every control point."""
    P = np.asarray(P, dtype=np.float64)
    ext = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        t0 = 0.0
        t1 = t0 + max(np.linalg.norm(p1 - p0), 1e-6) ** alpha
        t2 = t1 + max(np.linalg.norm(p2 - p1), 1e-6) ** alpha
        t3 = t2 + max(np.linalg.norm(p3 - p2), 1e-6) ** alpha
        last = i == len(ext) - 3
        ts = np.linspace(t1, t2, samples_per_seg + (1 if last else 0), endpoint=last)[:, None]
        a1 = (t1 - ts) / (t1 - t0) * p0 + (ts - t0) / (t1 - t0) * p1
        a2 = (t2 - ts) / (t2 - t1) * p1 + (ts - t1) / (t2 - t1) * p2
        a3 = (t3 - ts) / (t3 - t2) * p2 + (ts - t2) / (t3 - t2) * p3
        b1 = (t2 - ts) / (t2 - t0) * a1 + (ts - t0) / (t2 - t0) * a2
        b2 = (t3 - ts) / (t3 - t1) * a2 + (ts - t1) / (t3 - t1) * a3
        out.append((t2 - ts) / (t2 - t1) * b1 + (ts - t1) / (t2 - t1) * b2)
    return np.vstack(out)


def resample_polyline(poly, n):
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    target = np.linspace(0.0, cum[-1], n)
    return np.stack([np.interp(target, cum, poly[:, k]) for k in range(3)], axis=1), cum[-1]


def grid_quads(rows, cols, wrap_rows=False, wrap_cols=False):
    """Quads for a (rows x cols) vertex grid, index = r*cols + c. Winding gives normal = dP/dc x dP/dr."""
    faces = []
    for r in range(rows - 1 + int(wrap_rows)):
        r1 = (r + 1) % rows
        for c in range(cols - 1 + int(wrap_cols)):
            c1 = (c + 1) % cols
            faces.append((r * cols + c, r * cols + c1, r1 * cols + c1, r1 * cols + c))
    return np.asarray(faces, dtype=np.int64)


def grid_part(P, island, wrap_rows=False, wrap_cols=False, double_sided=True, name=None):
    rows, cols, _ = P.shape
    den_c = cols if wrap_cols else cols - 1
    den_r = rows if wrap_rows else rows - 1
    s = np.tile(np.arange(cols) / den_c, rows)
    t = np.repeat(np.arange(rows) / den_r, cols)
    # per-face-corner panel parameters, unwrapped so seams get s or t = 1.0
    corner_st = []
    for r in range(rows - 1 + int(wrap_rows)):
        for c in range(cols - 1 + int(wrap_cols)):
            corner_st.append([(c / den_c, r / den_r), ((c + 1) / den_c, r / den_r),
                              ((c + 1) / den_c, (r + 1) / den_r), (c / den_c, (r + 1) / den_r)])
    return dict(name=name or island, verts=P.reshape(-1, 3).copy(), faces=grid_quads(rows, cols, wrap_rows, wrap_cols),
                corner_st=np.asarray(corner_st, dtype=np.float64),
                s=s, t=t, rows=rows, cols=cols, wrap_rows=wrap_rows, wrap_cols=wrap_cols,
                island=island, double_sided=double_sided)


def horizontal_normals(P):
    """Unit horizontal normals of a (rows, cols, 3) grid, perpendicular to the column direction."""
    du = np.gradient(P, axis=1)
    n = np.stack([du[..., 1], -du[..., 0], np.zeros(du.shape[:2])], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    return n


def wrap_over_body(start_xy, direction_xy, dist, off, half_x=BODY_HALF_X, half_y=BODY_HALF_Y, z_top=TORSO_TOP):
    """Lay a pattern column over the top of the torso/arm block and down its side.

    start_xy: (C,2) start points on the top face; direction_xy: (C,2) unit walking directions;
    dist: (R,) distances along the pattern. Returns (R, C, 3) positions offset by `off`
    from the block, with a rounded bend over the edge.
    """
    start_xy = np.asarray(start_xy, dtype=np.float64)
    d = np.asarray(direction_xy, dtype=np.float64)
    C = len(start_xy)
    with np.errstate(divide="ignore", invalid="ignore"):
        tx = np.where(np.abs(d[:, 0]) > 1e-9, (np.sign(d[:, 0]) * half_x - start_xy[:, 0]) / d[:, 0], np.inf)
        ty = np.where(np.abs(d[:, 1]) > 1e-9, (np.sign(d[:, 1]) * half_y - start_xy[:, 1]) / d[:, 1], np.inf)
    t_edge = np.maximum(np.minimum(tx, ty), 0.0)
    n = np.zeros((C, 2))
    hit_x = tx < ty
    n[hit_x, 0] = np.sign(d[hit_x, 0])
    n[~hit_x, 1] = np.sign(d[~hit_x, 1])
    edge = start_xy + t_edge[:, None] * d
    arc = 0.5 * np.pi * off
    R = np.asarray(dist, dtype=np.float64)[:, None]
    past = R - t_edge[None, :]
    out = np.zeros((len(R), C, 3))
    on_top = past <= 0.0
    in_arc = (past > 0.0) & (past <= arc)
    below = past > arc
    beta = np.clip(past / max(off, 1e-6), 0.0, 0.5 * np.pi)
    for k in range(2):
        top_val = start_xy[None, :, k] + R * d[None, :, k]
        arc_val = edge[None, :, k] + n[None, :, k] * off * np.sin(beta)
        below_val = edge[None, :, k] + n[None, :, k] * off
        out[..., k] = np.where(on_top, top_val, np.where(in_arc, arc_val, below_val))
    out[..., 2] = np.where(on_top, z_top + off,
                           np.where(in_arc, z_top + off * np.cos(beta), z_top - (past - arc)))
    face_normal = np.zeros((len(R), C, 3))
    face_normal[..., 0] = np.where(on_top, 0.0, n[None, :, 0])
    face_normal[..., 1] = np.where(on_top, 0.0, n[None, :, 1])
    face_normal[..., 2] = np.where(on_top, 1.0, 0.0)
    return out, face_normal, past


# --------------------------------------------------------------------------- cape
CAPE_PHI0 = np.radians(102.0)


def cape_levels(phi):
    """Control cross-sections of the cape, top (on the shoulders) to hem."""
    c = np.cos(phi)
    s = np.sin(phi)
    hem_z = 0.15 + 0.47 * ((1.0 - c) / (1.0 - np.cos(CAPE_PHI0))) ** 1.15
    top_z = 3.97 + 0.02 * c
    levels = []
    # (a, b, c, z)
    levels.append((1.18, 0.42, 0.20, 4.052 + 0.03 * c))
    levels.append((1.32, 0.12, 0.565, top_z))
    for w, a, b, cc in ((0.17, 1.55, 0.25, 0.60), (0.45, 2.00, 0.50, 0.56),
                        (0.72, 2.42, 0.90, 0.50), (1.0, 2.64, 1.34, 0.45)):
        levels.append((a, b, cc, top_z + (hem_z - top_z) * w))
    pts = []
    for a, b, cc, z in levels:
        pts.append(np.stack([a * s, cc + b * c, z * np.ones_like(s)], axis=-1))
    return np.stack(pts, axis=0)  # (K, C, 3)


def build_cape(cols=89, rows=49, seed=11):
    u = np.linspace(0.0, 1.0, cols)
    phi = (u - 0.5) * 2.0 * CAPE_PHI0
    L = cape_levels(phi)
    P = np.zeros((rows, cols, 3))
    col_len = np.zeros(cols)
    for j in range(cols):
        dense = centripetal_catmull_rom(L[:, j, :], samples_per_seg=40)
        P[:, j, :], col_len[j] = resample_polyline(dense, rows)

    v = np.linspace(0.0, 1.0, rows)[:, None]
    uu = u[None, :]
    base = P.copy()  # fold-free surface: bones are placed on it
    nrm = horizontal_normals(P)
    # orient outward (away from the body axis)
    radial = P[..., :2] - np.array([0.0, 0.25])
    sign = np.sign((nrm[..., :2] * radial).sum(-1, keepdims=True))
    sign[sign == 0] = 1
    nrm *= sign

    # gravity folds that fan out from the shoulders
    ps1 = smooth_random_1d(seed, 5)
    ps2 = smooth_random_1d(seed + 1, 7)
    pamp = smooth_random_1d(seed + 2, 4)
    N = 7.0 + 4.5 * v
    phase = 2.0 * np.pi * (N * (uu - 0.5) + 0.12 * ps1(uu)) + 0.4
    F = np.sin(phase)
    F = np.sign(F) * np.abs(F) ** 0.75
    F += 0.35 * v * np.sin(2.0 * np.pi * (2.1 * N * (uu - 0.5) + 0.2 * ps2(uu)) + 1.3)
    A = (0.165 * smoothstep(0.03, 0.62, v) + 0.035 * v) * (0.8 + 0.2 * pamp(uu))
    banner_u = 0.5 + np.degrees(np.arcsin(0.12 / 2.2)) / np.degrees(2 * CAPE_PHI0)
    A *= 1.0 - 0.7 * np.exp(-((uu - banner_u) / 0.045) ** 2) * smoothstep(0.05, 0.25, v)
    # ragged hem: strips hang slightly independently
    hem_noise = smooth_random_1d(seed + 3, 40, falloff=0.35)
    A_hem = 0.05 * smoothstep(0.62, 1.0, v) * hem_noise(uu)
    # strips at the sides splay outwards so the silhouette breaks up like the concept
    A_hem += 0.11 * smoothstep(0.55, 1.0, v) * np.abs(np.sin(phi))[None, :] ** 2 * (0.6 + 0.4 * hem_noise(uu * 1.3 + 0.2))
    P += nrm * (A * F + A_hem)[..., None]
    # strips at the very bottom curl back a little
    P[..., 2] += 0.03 * smoothstep(0.85, 1.0, v) * hem_noise(uu * 0.5 + 0.3)

    part = grid_part(P, "cape", name="cape")
    part["col_len"] = col_len
    part["base"] = base
    return part


# --------------------------------------------------------------------------- cowl rolls
ROLLS = [
    # rx, ry_front, ry_back, z_side, front_sag, back_rise, half_h, half_w
    (0.755, 0.700, 0.760, 4.400, 0.22, 0.14, 0.092, 0.068),
    (0.885, 0.705, 0.860, 4.290, 0.32, 0.13, 0.102, 0.072),
    (1.010, 0.690, 0.950, 4.185, 0.45, 0.11, 0.112, 0.075),
    (1.140, 0.670, 1.020, 4.095, 0.58, 0.08, 0.120, 0.078),
]


def roll_center(i, theta, seed=21):
    rx, ryf, ryb, zs, sag, rise, _, _ = ROLLS[i]
    c = np.cos(theta)
    s = np.sin(theta)
    front = smoothstep(-0.15, 0.35, c)
    ry = ryb + (ryf - ryb) * front
    wob = smooth_random_1d(seed + 7 * i, 6)
    t01 = theta / (2 * np.pi)
    r_scale = 1.0 + 0.025 * wob(t01)
    x = rx * s * r_scale
    y = -ry * c * r_scale
    z = zs - sag * np.clip(c, 0, 1) ** 1.6 + rise * np.clip(-c, 0, 1) ** 2 + 0.018 * wob(t01 + 0.37)
    return np.stack([x, y, z], axis=-1)


def roll_frame(i, theta, seed=21):
    """Centre, tangent, major (up/in along the stack) and minor (outward) axes of roll i."""
    C = roll_center(i, theta, seed)
    eps = 1e-3
    T = roll_center(i, theta + eps, seed) - roll_center(i, theta - eps, seed)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    j = min(i + 1, len(ROLLS) - 1)
    k = i if i + 1 < len(ROLLS) else i - 1
    D = roll_center(j, theta, seed) - roll_center(k, theta, seed)
    D -= (D * T).sum(1, keepdims=True) * T
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    major = -D
    minor = np.cross(T, major)
    flip = np.sign((minor[:, :2] * C[:, :2]).sum(1) + 1e-9)
    minor *= flip[:, None]
    return C, T, major, minor


def build_roll(i, n_path=60, n_cross=8, seed=21):
    theta = np.linspace(0.0, 2.0 * np.pi, n_path, endpoint=False)
    C, T, major, minor = roll_frame(i, theta, seed)
    _, _, _, _, _, _, hh, hw = ROLLS[i]
    t01 = theta / (2 * np.pi)
    bulge = smooth_random_1d(seed + 31 * i, 12, falloff=0.6)
    pinch = 1.0 + 0.14 * bulge(t01)
    front = np.clip(np.cos(theta), 0, 1)
    h = hh * pinch * (1.0 + 0.18 * front)
    w = hw * (1.0 + 0.10 * bulge(t01 + 0.5))
    ang = np.linspace(0.0, 2.0 * np.pi, n_cross, endpoint=False)
    # slightly flattened, asymmetric section: a folded band of cloth rather than a tube
    ca = np.cos(ang)[None, :]
    sa = np.sin(ang)[None, :]
    prof_major = ca * (1.0 + 0.08 * sa)
    prof_minor = sa * (0.85 + 0.25 * np.clip(ca, 0, 1))
    P = (C[:, None, :] + h[:, None, None] * prof_major[..., None] * major[:, None, :]
         + w[:, None, None] * prof_minor[..., None] * minor[:, None, :])
    # rows = cross-section, cols = along the path (so the long direction runs along atlas x)
    P = np.transpose(P, (1, 0, 2))
    part = grid_part(P, "roll%d" % i, wrap_rows=True, wrap_cols=True, double_sided=False)
    return part


# --------------------------------------------------------------------------- hood swags (lowered hood)
SWAGS = [
    # rx, ry, cy, z_end, z_mid, half_h, half_w, alpha_max(deg)
    (1.00, 0.68, 0.32, 4.34, 4.08, 0.130, 0.100, 104.0),
    (1.08, 0.80, 0.32, 4.28, 3.78, 0.145, 0.105, 100.0),
    (1.14, 0.88, 0.32, 4.21, 3.48, 0.155, 0.110, 96.0),
]


def swag_center(k, a01, seed=41):
    rx, ry, cy, z_end, z_mid, _, _, amax = SWAGS[k]
    alpha = np.radians(amax) * (2.0 * a01 - 1.0)
    w = np.cos(np.clip(alpha / np.radians(amax), -1, 1) * np.pi / 2.0)
    wob = smooth_random_1d(seed + 5 * k, 5, periodic=False)
    x = rx * np.sin(alpha)
    y = cy + ry * np.cos(alpha) + 0.02 * wob(a01)
    z = z_end + (z_mid - z_end) * w ** 1.35 + 0.015 * wob(a01 + 0.4)
    return np.stack([x, y, z], axis=-1)


def swag_frame(k, a, seed=41):
    C = swag_center(k, a, seed)
    eps = 1e-3
    T = swag_center(k, a + eps, seed) - swag_center(k, a - eps, seed)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    out = C.copy()
    out[:, 0] *= 0.6
    out[:, 1] -= 0.1
    out[:, 2] = 0.0
    out -= (out * T).sum(1, keepdims=True) * T
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    major = np.cross(out, T)
    major *= np.sign(major[:, 2:3] + 1e-9)
    minor = np.cross(T, major)
    minor *= np.sign((minor[:, :2] * (C[:, :2] - [0, 0.1])).sum(1, keepdims=True) + 1e-9)
    return C, T, major, minor


def build_swag(k, n_path=36, n_cross=8, seed=41):
    a = np.linspace(0.0, 1.0, n_path)
    C = swag_center(k, a, seed)
    T = np.gradient(C, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    # cross-section frame: "down-out" major axis, outward minor
    out = C.copy()
    out[:, 0] *= 0.6
    out[:, 1] -= 0.1
    out[:, 2] = 0.0
    out -= (out * T).sum(1, keepdims=True) * T
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    major = np.cross(out, T)
    major *= np.sign(major[:, 2:3] + 1e-9)  # point up
    minor = np.cross(T, major)
    minor *= np.sign((minor[:, :2] * (C[:, :2] - [0, 0.1])).sum(1, keepdims=True) + 1e-9)
    _, _, _, _, _, hh, hw, _ = SWAGS[k]
    taper = 0.35 + 0.65 * np.sin(np.pi * a) ** 0.5
    bulge = smooth_random_1d(seed + 13 * k, 9, falloff=0.6, periodic=False)
    h = hh * taper * (1.0 + 0.12 * bulge(a))
    w = hw * taper
    ang = np.linspace(0.0, 2.0 * np.pi, n_cross, endpoint=False)
    ca = np.cos(ang)[None, :]
    sa = np.sin(ang)[None, :]
    P = (C[:, None, :] + h[:, None, None] * ca[..., None] * major[:, None, :]
         + w[:, None, None] * (sa * (0.85 + 0.25 * np.clip(ca, 0, 1)))[..., None] * minor[:, None, :])
    P = np.transpose(P, (1, 0, 2))
    return grid_part(P, "swag%d" % k, wrap_rows=True, wrap_cols=False, double_sided=False)


def build_hood_sheet(cols=44, rows=14, seed=51):
    """Fabric body of the lowered hood behind the swags (prevents see-through gaps)."""
    a = np.linspace(0.0, 1.0, cols)
    # top edge: behind the neck under roll 1, ends on the shoulders
    rx, ry, cy, amax = 0.90, 0.50, 0.30, 108.0
    alpha = np.radians(amax) * (2.0 * a - 1.0)
    top = np.stack([rx * np.sin(alpha), cy + ry * np.cos(alpha), 4.30 - 0.12 * np.abs(np.sin(alpha))], -1)
    mids = [swag_center(k, a, 41) for k in range(3)]
    bottom = mids[-1].copy()
    w = np.cos(np.clip(alpha / np.radians(amax), -1, 1) * np.pi / 2.0)
    bottom[:, 2] -= 0.16 * w + 0.03
    bottom[:, 1] -= 0.04
    ctrl = [top] + [m - np.array([0, 0.055, 0]) for m in mids] + [bottom]
    P = np.zeros((rows, cols, 3))
    for j in range(cols):
        dense = centripetal_catmull_rom(np.stack([c[j] for c in ctrl]), 20)
        P[:, j], _ = resample_polyline(dense, rows)
    return grid_part(P, "hood", name="hood")


# --------------------------------------------------------------------------- capelet / front drape
CAPELET_X = (-1.78, 1.42)


def build_capelet(cols=52, rows=28, seed=61):
    """Tattered mantle lying over the shoulders and hanging down the chest."""
    s = np.linspace(0.0, 1.0, cols)
    x = CAPELET_X[0] + (CAPELET_X[1] - CAPELET_X[0]) * s
    neck = np.sqrt(np.clip(0.74 ** 2 - x ** 2, 0, None))
    y0 = np.where(np.abs(x) < 0.74, -0.62 * neck / 0.74 + 0.0, 0.0) + 0.16 * smoothstep(0.7, 1.3, np.abs(x))
    start = np.stack([x, y0], -1)
    d = np.tile([[0.0, -1.0]], (cols, 1))
    length = 2.3
    dist = np.linspace(0.0, length, rows)
    P, fn, past = wrap_over_body(start, d, dist, off=0.03)
    fold = smooth_random_1d(seed, 14, falloff=0.5)
    F = np.sin(2 * np.pi * (9.0 * s + 0.15 * fold(s)))[None, :]
    A = 0.035 * smoothstep(0.05, 0.9, past) + 0.01
    P += fn * (A * F + 0.012)[..., None] * (past > 0)[..., None]
    P[..., 1] -= 0.02 * smoothstep(0.0, 1.5, past) * (1 + 0.5 * fold(s * 1.7 + 0.2))[None, :]
    return grid_part(P, "capelet")


DRAPE_X = (0.28, 1.16)


def drape_base_y(z):
    """Front-left drape offset from the body (without folds); shared with the front banner."""
    return -0.5 - 0.058 - 0.07 * smoothstep(2.1, 0.9, z)


def build_drape(cols=16, rows=56, seed=71):
    """Front-left panel that comes over the left shoulder and hangs down to the knee."""
    s = np.linspace(0.0, 1.0, cols)
    x = DRAPE_X[0] + (DRAPE_X[1] - DRAPE_X[0]) * s
    start = np.stack([x, -0.06 + 0.10 * s], -1)
    d = np.tile([[0.0, -1.0]], (cols, 1))
    length = 3.47
    dist = np.linspace(0.0, length, rows)
    P, fn, past = wrap_over_body(start, d, dist, off=0.058)
    z = P[..., 2].copy()
    # fold-free base (bones are placed on it): hangs slightly away from the legs below the torso
    base = P.copy()
    base[..., 1] -= 0.07 * smoothstep(2.1, 0.9, z)
    base[..., 0] += 0.06 * smoothstep(2.0, 0.9, z) * (s[None, :] - 0.3)
    # vertical folds
    fold = smooth_random_1d(seed, 6)
    F = np.sin(2 * np.pi * (2.4 * s + 0.1 * fold(s)) + 0.6)[None, :]
    A = 0.045 * smoothstep(0.1, 2.5, past) + 0.008
    P += fn * (A * F)[..., None] * (past > 0)[..., None]
    P[..., 1] -= (0.07 * smoothstep(2.1, 0.9, z) + 0.02 * smoothstep(3.0, 1.0, z) * fold(s * 0.8 + 0.3)[None, :])
    P[..., 0] += 0.06 * smoothstep(2.0, 0.9, z) * (s[None, :] - 0.3)
    part = grid_part(P, "drape")
    part["base"] = base
    return part


# --------------------------------------------------------------------------- frayed fringes
FRINGES = [("roll", 0), ("roll", 1), ("roll", 2), ("roll", 3), ("swag", 0), ("swag", 1), ("swag", 2)]


def build_fringe(kind, idx, cols=44, rows=4, seed=81):
    """Short tattered lip hanging from the lower edge of a roll or swag."""
    if kind == "roll":
        hh, hw = ROLLS[idx][6], ROLLS[idx][7]
        span = 130.0 if idx < 3 else 88.0
        theta = np.radians(np.linspace(-span, span, cols))
        C, T, major, minor = roll_frame(idx, theta)
        attach = C - major * hh * 0.55 + minor * hw * 0.70
        if idx < 3:
            C2, _, major2, minor2 = roll_frame(idx + 1, theta)
            h2, w2 = ROLLS[idx + 1][6], ROLLS[idx + 1][7]
            end = C2 + minor2 * (w2 * 1.05 + 0.014) + major2 * h2 * 0.05
        else:
            end = attach + np.array([0.0, 0.0, -0.26]) + minor * 0.06
    else:
        hh, hw = SWAGS[idx][5], SWAGS[idx][6]
        a = np.linspace(0.04, 0.96, cols)
        C, T, major, minor = swag_frame(idx, a)
        taper = 0.35 + 0.65 * np.sin(np.pi * a) ** 0.5
        attach = C - major * (hh * taper * 0.55)[:, None] + minor * (hw * taper * 0.75)[:, None]
        if idx < 2:
            C2, _, major2, minor2 = swag_frame(idx + 1, a)
            w2 = SWAGS[idx + 1][6] * taper
            end = C2 + minor2 * (w2 * 1.05 + 0.014)[:, None]
        else:
            end = attach + np.array([0.0, 0.0, -0.27]) + minor * 0.05
    t = np.linspace(0.0, 1.0, rows)[:, None, None]
    bulge = minor[None] * (0.02 * np.sin(np.pi * t))
    P = attach[None] + (end - attach)[None] * t + bulge
    rnd = smooth_random_1d(seed + 3 * idx + (0 if kind == "roll" else 50), 16, falloff=0.4)
    P[..., 2] += 0.012 * rnd(np.linspace(0, 1, cols))[None, :] * t[..., 0]
    return grid_part(P, "fringe_%s%d" % (kind, idx))


# --------------------------------------------------------------------------- banners
def build_banner_front(cols=6, rows=34, seed=91):
    s = np.linspace(0.0, 1.0, cols)
    t = np.linspace(0.0, 1.0, rows)
    z = 3.66 - (3.66 - 1.42) * t
    w = 0.34 - 0.04 * t
    xc = 0.56 + 0.025 * np.sin(t * 2.2)
    X = xc[:, None] + (s[None, :] - 0.5) * w[:, None]
    wave = smooth_random_1d(seed, 4, periodic=False)
    Y = drape_base_y(z)[:, None] - 0.065 - 0.012 * wave(t)[:, None] - 0.01 * np.cos(np.pi * (s[None, :] - 0.5))
    Z = np.repeat(z[:, None], cols, 1)
    return grid_part(np.stack([X, Y, Z], -1), "banner_front")


def build_banner_back(cape, cols=7, rows=40, seed=93):
    """Banner hanging over the cape's back centre; sits on the cape surface."""
    s = np.linspace(0.0, 1.0, cols)
    t = np.linspace(0.0, 1.0, rows)
    z = 3.62 - (3.62 - 0.92) * t
    w = 0.44 - 0.06 * t
    xc = 0.12 + 0.02 * np.sin(t * 2.0)
    X = xc[:, None] + (s[None, :] - 0.5) * w[:, None]
    Z = np.repeat(z[:, None], cols, 1)
    cv = cape["verts"]
    back = cv[cv[:, 1] > 0.3]
    Y = np.zeros_like(X)
    for i in range(rows):
        for j in range(cols):
            d2 = (back[:, 0] - X[i, j]) ** 2 + (back[:, 2] - Z[i, j]) ** 2
            nn = np.argpartition(d2, 6)[:6]
            wts = 1.0 / (np.sqrt(d2[nn]) + 1e-3)
            Y[i, j] = (back[nn, 1] * wts).sum() / wts.sum()
    # smooth along the banner and sit just outside the cape
    for _ in range(4):
        Y[1:-1] = 0.25 * Y[:-2] + 0.5 * Y[1:-1] + 0.25 * Y[2:]
    Y = Y.max(axis=1, keepdims=True) + 0.045 + 0.008 * np.cos(np.pi * (s[None, :] - 0.5))
    return grid_part(np.stack([X, np.repeat(Y, 1, 1), Z], -1), "banner_back")


# the back banner rides on the cape, so it lives in the cape mesh and shares its bones
GROUP_OF = {"cape": "cape", "banner_back": "cape", "hood": "cowl", "capelet": "drapes", "drape": "drapes",
            "banner_front": "drapes"}


def group_of(name):
    if name in GROUP_OF:
        return GROUP_OF[name]
    return "cowl"


def build_all_fabric():
    cape = build_cape()
    parts = [cape]
    parts += [build_roll(i) for i in range(len(ROLLS))]
    parts += [build_swag(k) for k in range(len(SWAGS))]
    parts.append(build_hood_sheet())
    parts += [build_fringe(kind, idx) for kind, idx in FRINGES]
    parts.append(build_capelet())
    parts.append(build_drape())
    parts.append(build_banner_front())
    parts.append(build_banner_back(cape))
    for p in parts:
        p["group"] = group_of(p["name"])
    return parts


if __name__ == "__main__":
    for p in build_all_fabric():
        v = p["verts"]
        print(p["name"], v.shape, p["faces"].shape, "min", v.min(0).round(3), "max", v.max(0).round(3))
