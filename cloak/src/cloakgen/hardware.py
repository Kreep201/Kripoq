"""Hardware for the cloak (runs inside Blender): leather straps, buckles, keeper, spiked bird-skull
brooch, chains, O-ring and hanging tag. Geometry is generated with numpy; placement uses BVH queries
against the (single-sided, outward-facing) cloth so straps lie on the fabric and chains rest on it."""
import math

import numpy as np
from mathutils import Vector

from . import bl_util, layout
from .fabric_geom import centripetal_catmull_rom, resample_polyline

PX = layout.ATLAS_PX
STRAP_W = layout.STRAP_W
STRAP_T = layout.STRAP_T


# --------------------------------------------------------------------------- generic geometry
def _unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def frames(path, hint, closed=False):
    if closed:
        T = np.roll(path, -1, 0) - np.roll(path, 1, 0)
    else:
        T = np.gradient(path, axis=0)
    T = _unit(T)
    hint = np.broadcast_to(np.asarray(hint, dtype=np.float64), path.shape)
    N = _unit(hint - (hint * T).sum(-1, keepdims=True) * T)
    B = np.cross(T, N)
    return T, N, B


def sweep(path, N, B, section, closed=False):
    """Sweep a CCW (N->B) section along a path. Returns verts, quads, corner (s, t) params."""
    n, k = len(path), len(section)
    V = path[:, None, :] + section[None, :, :1] * N[:, None, :] + section[None, :, 1:2] * B[:, None, :]
    rows = n if closed else n - 1
    den = n if closed else n - 1
    # around-parameter proportional to section perimeter so wide faces get matching texel density
    seg = np.linalg.norm(np.roll(section, -1, 0) - section, axis=1)
    tc = np.r_[0.0, np.cumsum(seg)] / seg.sum()
    faces, cst = [], []
    for i in range(rows):
        i1 = (i + 1) % n
        for j in range(k):
            j1 = (j + 1) % k
            faces.append((i * k + j, i * k + j1, i1 * k + j1, i1 * k + j))
            cst.append(((i / den, tc[j]), (i / den, tc[j + 1]), ((i + 1) / den, tc[j + 1]), ((i + 1) / den, tc[j])))
    return V.reshape(-1, 3), np.asarray(faces), np.asarray(cst, dtype=np.float64)


def circle_section(r, k, phase=0.0, sx=1.0, sy=1.0):
    a = phase + np.arange(k) * 2 * np.pi / k
    return np.stack([sx * r * np.cos(a), sy * r * np.sin(a)], 1)


def chamfer_rect_section(w, h, bevel):
    hw, hh = w / 2, h / 2
    pts = []
    for cx, cy, a0 in ((hw - bevel, hh - bevel, 0), (-hw + bevel, hh - bevel, 90),
                       (-hw + bevel, -hh + bevel, 180), (hw - bevel, -hh + bevel, 270)):
        for q in (0, 1):
            a = math.radians(a0 + 90 * q)
            pts.append((cx + bevel * math.cos(a), cy + bevel * math.sin(a)))
    return np.asarray(pts)


def rounded_rect_path(L, W, rc, n_corner=3):
    pts = []
    for cx, cy, a0 in ((L / 2 - rc, W / 2 - rc, 0), (-L / 2 + rc, W / 2 - rc, 90),
                       (-L / 2 + rc, -W / 2 + rc, 180), (L / 2 - rc, -W / 2 + rc, 270)):
        for q in range(n_corner):
            a = math.radians(a0 + 90 * q / (n_corner - 1))
            pts.append((cx + rc * math.cos(a), cy + rc * math.sin(a), 0.0))
    return np.asarray(pts)


def stadium_path(Ls, rc, n):
    perim = 2 * Ls + 2 * math.pi * rc
    pts = []
    for d in np.arange(n) / n * perim:
        if d < Ls:
            p = (-Ls / 2 + d, -rc)
        elif d < Ls + math.pi * rc:
            a = -math.pi / 2 + (d - Ls) / rc
            p = (Ls / 2 + rc * math.cos(a), rc * math.sin(a))
        elif d < 2 * Ls + math.pi * rc:
            p = (Ls / 2 - (d - Ls - math.pi * rc), rc)
        else:
            a = math.pi / 2 + (d - 2 * Ls - math.pi * rc) / rc
            p = (-Ls / 2 + rc * math.cos(a), rc * math.sin(a))
        pts.append((p[0], p[1], 0.0))
    return np.asarray(pts)


def closed_planar_tube(path, section):
    """Tube around a closed path lying in the local XY plane; normals point outwards."""
    T = _unit(np.roll(path, -1, 0) - np.roll(path, 1, 0))
    N = _unit(np.cross(T, [0.0, 0.0, 1.0]))  # in-plane, outward for a CCW path
    B = np.cross(T, N)
    return sweep(path, N, B, section, closed=True)


def transform(verts, M):
    v = np.c_[verts, np.ones(len(verts))] @ np.asarray(M).T
    return v[:, :3]


def basis_matrix(origin, x_axis, y_axis, z_axis):
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = x_axis, y_axis, z_axis, origin
    return M


class MeshBuilder:
    """Accumulates parts of one object; uv is either per-corner (F,4,2) or None (smart unwrap later)."""

    def __init__(self):
        self.verts, self.faces, self.uvs = [], [], []
        self.n = 0

    def add(self, verts, faces, corner_uv=None):
        faces = np.asarray(faces)
        self.verts.append(np.asarray(verts, dtype=np.float64))
        quads = [tuple(int(i) + self.n for i in f) for f in faces]
        self.faces += quads
        if corner_uv is None:
            corner_uv = np.zeros((len(faces), faces.shape[1], 2))
        self.uvs.append(np.asarray(corner_uv).reshape(-1, 2))
        self.n += len(verts)

    def build(self, name, collection, smooth=True):
        V = np.vstack(self.verts)
        uv = np.vstack(self.uvs)
        faces, loop_uv = [], []
        k = 0
        for f in self.faces:
            corners = list(range(k, k + len(f)))
            k += len(f)
            # quads with a repeated last index encode triangles
            clean, cl_uv = [], []
            for vi, ci in zip(f, corners):
                if vi not in clean:
                    clean.append(vi)
                    cl_uv.append(uv[ci])
            if len(clean) >= 3:
                faces.append(clean)
                loop_uv += cl_uv
        me = bl_util.bpy.data.meshes.new(name)
        me.from_pydata([tuple(v) for v in V], [], faces)
        uvl = me.uv_layers.new(name="UVMap")
        uvl.data.foreach_set("uv", np.asarray(loop_uv, dtype=np.float32).ravel())
        me.validate(clean_customdata=False)
        for poly in me.polygons:
            poly.use_smooth = smooth
        ob = bl_util.bpy.data.objects.new(name, me)
        collection.objects.link(ob)
        return ob


def region_uv(region, cst, swap=False):
    """Map (s along, t around) params into an atlas region (x, y, w, h)."""
    x0, y0, w, h = region
    s, t = cst[..., 0], cst[..., 1]
    if swap:
        s, t = t, s
    return np.stack([(x0 + s * w) / PX, 1.0 - (y0 + t * h) / PX], -1)


# --------------------------------------------------------------------------- scene queries
def raycast_snap(bvh, pts, dirs, offset, reach=0.6):
    """Snap points onto the first surface hit when looking along `dirs` (inwards)."""
    out = pts.copy()
    nrm = np.zeros_like(pts)
    hit_ok = np.zeros(len(pts), bool)
    for i, (p, d) in enumerate(zip(pts, dirs)):
        d = Vector(d).normalized()
        origin = Vector(p) - d * reach
        loc, n, _, _ = bvh.ray_cast(origin, d, reach * 2.5)
        if loc is None:
            continue
        n = Vector(n)
        if n.dot(d) > 0:
            n = -n
        out[i] = np.asarray(loc + n * offset)
        nrm[i] = np.asarray(n)
        hit_ok[i] = True
    # fill misses from neighbours
    for i in range(len(pts)):
        if not hit_ok[i]:
            j = np.argmin(np.where(hit_ok, np.abs(np.arange(len(pts)) - i), 10 ** 9))
            nrm[i] = nrm[j] if hit_ok.any() else -np.asarray(dirs[i])
    return out, _unit(nrm)


def push_out(bvh, pts, clearance, probe=0.25):
    out = pts.copy()
    for i, p in enumerate(pts):
        loc, n, _, dist = bvh.find_nearest(Vector(p), probe)
        if loc is None:
            continue
        d = (Vector(p) - loc).dot(n)
        if d < clearance:
            out[i] = np.asarray(loc + n * clearance)
    return out


def smooth_polyline(P, iters=3, w=0.5):
    P = P.copy()
    for _ in range(iters):
        P[1:-1] = (1 - w) * P[1:-1] + w * 0.5 * (P[:-2] + P[2:])
    return P


# --------------------------------------------------------------------------- straps
def strap_path(bvh, ctrl, dirs, spacing=0.045):
    ctrl = np.asarray(ctrl, dtype=np.float64)
    dense = centripetal_catmull_rom(ctrl, 30)
    total = np.linalg.norm(np.diff(dense, axis=0), axis=1).sum()
    n = max(int(total / spacing), 4)
    path, _ = resample_polyline(dense, n)
    # interpolate the inward look directions along the path
    seg = np.linalg.norm(np.diff(ctrl, axis=0), axis=1)
    cum_c = np.r_[0, np.cumsum(seg)] / max(seg.sum(), 1e-9)
    cum_p = np.linspace(0, 1, n)
    D = _unit(np.stack([np.interp(cum_p, cum_c, np.asarray(dirs)[:, k]) for k in range(3)], 1))
    snapped, nrm = raycast_snap(bvh, path, D, STRAP_T * 0.5 + 0.007)
    for _ in range(2):
        snapped = smooth_polyline(snapped, 2, 0.5)
        snapped = push_out(bvh, snapped, STRAP_T * 0.5 + 0.006)
    nrm = smooth_polyline(nrm, 4, 0.6)
    path, length = resample_polyline(snapped, n)
    return path, _unit(nrm), length


def strap_mesh(mb, path, nrm, regions):
    T, N, B = frames(path, nrm)
    # section in (N = surface normal, B = across) plane: thin, wide band
    sec = chamfer_rect_section(STRAP_T, STRAP_W, 0.006)
    n = len(path)
    if len(regions) == 1:
        V, F, cst = sweep(path, N, B, sec)
        mb.add(V, F, region_uv(regions[0], cst))
        return
    half = n // 2
    for k, (a, b) in enumerate(((0, half + 1), (half, n))):
        V, F, cst = sweep(path[a:b], N[a:b], B[a:b], sec)
        mb.add(V, F, region_uv(regions[k], cst))


def strap_frame_at(path, nrm, s):
    """Position/orientation (T along, B across, N out) at fraction s of a strap path."""
    T, N, B = frames(path, nrm)
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum = np.r_[0, np.cumsum(seg)] / seg.sum()
    i = int(np.clip(np.searchsorted(cum, s) - 1, 0, len(path) - 2))
    f = (s - cum[i]) / max(cum[i + 1] - cum[i], 1e-9)
    p = path[i] * (1 - f) + path[i + 1] * f
    return p, T[i], N[i], B[i]


# --------------------------------------------------------------------------- metal pieces (smart-unwrapped)
def buckle(mb, origin, T, N, B, scale=1.0):
    """Rectangular buckle frame with a centre bar and prong, lying on a strap."""
    L, W = 0.17 * scale, (STRAP_W + 0.05) * scale
    frame_path = rounded_rect_path(L, W, 0.026 * scale, 3)
    sec = chamfer_rect_section(0.028 * scale, 0.022 * scale, 0.006 * scale)
    M = basis_matrix(origin + N * (STRAP_T * 0.5 + 0.004), T, B, N)
    V, F, _ = closed_planar_tube(frame_path, sec)
    mb.add(transform(V, M), F)
    # centre bar across the frame
    bar = np.array([[-0.012 * scale, -W / 2 + 0.01, 0.0], [-0.012 * scale, W / 2 - 0.01, 0.0]])
    rod(mb, transform(bar, M), 0.009 * scale, 6)
    # prong from the bar to the far side, slightly raised
    prong = np.array([[-0.012 * scale, 0.0, 0.004], [0.06 * scale, 0.0, 0.012], [L / 2 + 0.004, 0.0, 0.008]])
    rod(mb, transform(prong, M), 0.0075 * scale, 6, taper=0.6)


def keeper(mb, origin, T, N, B):
    """Metal keeper loop around the strap."""
    path = rounded_rect_path(STRAP_T + 0.03, STRAP_W + 0.022, 0.01, 3)  # in (N, B) plane
    sec = chamfer_rect_section(0.010, 0.045, 0.003)
    M = basis_matrix(origin, N, B, T)
    V, F, _ = closed_planar_tube(path, sec)
    mb.add(transform(V, M), F)


def rod(mb, pts, r, k=6, taper=1.0):
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) > 2:
        pts = centripetal_catmull_rom(pts, 4)
    up = np.array([0.0, 0.0, 1.0])
    d = pts[-1] - pts[0]
    hint = np.cross(d, up) if np.linalg.norm(np.cross(d, up)) > 1e-6 else np.array([1.0, 0, 0])
    T, N, B = frames(pts, hint)
    radii = r * np.linspace(1.0, taper, len(pts))
    V = []
    F = []
    for i, (p, n_, b_, rr) in enumerate(zip(pts, N, B, radii)):
        sec = circle_section(rr, k)
        V += [p + s[0] * n_ + s[1] * b_ for s in sec]
    V = np.asarray(V)
    for i in range(len(pts) - 1):
        for j in range(k):
            j1 = (j + 1) % k
            F.append((i * k + j, i * k + j1, (i + 1) * k + j1, (i + 1) * k + j))
    # end caps as fans
    c0 = len(V)
    V = np.vstack([V, pts[0], pts[-1]])
    tris = []
    last = (len(pts) - 1) * k
    for j in range(k):
        j1 = (j + 1) % k
        tris.append((j1, j, c0, c0))
        tris.append((last + j, last + j1, c0 + 1, c0 + 1))
    mb.add(V, np.asarray(F + tris))


def spike(mb, base_center, direction, side, length, half_w, half_d, sides=4):
    d = _unit(np.asarray(direction, dtype=np.float64))
    s = _unit(np.asarray(side, dtype=np.float64))
    u = np.cross(d, s)
    ring = []
    for j in range(sides):
        a = 2 * math.pi * j / sides + math.pi / sides
        ring.append(base_center + s * half_w * math.cos(a) + u * half_d * math.sin(a))
    tip = base_center + d * length
    V = np.vstack([ring, tip, base_center])
    F = []
    for j in range(sides):
        j1 = (j + 1) % sides
        F.append((j, j1, sides, sides))
        F.append((j1, j, sides + 1, sides + 1))
    mb.add(V, F)


def skull(mb, M):
    """Raven skull with a long downward beak; built in brooch-local space (X right, Z up, -Y front)."""
    nu, nv = 18, 12
    rx, ry, rz = 0.068, 0.062, 0.078
    V = []
    for i in range(nv + 1):
        th = math.pi * i / nv
        for j in range(nu):
            ph = 2 * math.pi * j / nu
            x, y, z = math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)
            V.append([x, y, z])
    V = np.asarray(V)
    P = V * [rx, ry, rz]
    P[:, 1] = np.where(P[:, 1] > 0, P[:, 1] * 0.55, P[:, 1])  # flat back against the brooch
    P[:, 0] *= 1.0 - 0.28 * np.clip(-V[:, 2], 0, 1)          # narrower towards the jaw
    for sx in (-1, 1):  # eye sockets
        c = _unit(np.array([0.48 * sx, -0.78, 0.30]))
        ang = np.arccos(np.clip(V @ c, -1, 1))
        dent = np.clip(1 - ang / 0.42, 0, 1) ** 1.5
        P -= (V * [rx, ry, rz]) * dent[:, None] * 0.30
        rim = np.exp(-((ang - 0.48) / 0.08) ** 2)
        P += (V * [rx, ry, rz]) * rim[:, None] * 0.05
    P += [0.0, -0.032, 0.042]
    F = []
    for i in range(nv):
        for j in range(nu):
            j1 = (j + 1) % nu
            F.append((i * nu + j, i * nu + j1, (i + 1) * nu + j1, (i + 1) * nu + j))
    mb.add(transform(P, M), F)
    # beak: elliptical section tapering to a point, slight downward hook
    n_r, k = 11, 8
    path = np.array([[0, -0.072, 0.012], [0, -0.092, -0.06], [0, -0.098, -0.15], [0, -0.092, -0.24], [0, -0.078, -0.305]])
    path = centripetal_catmull_rom(path, 6)
    path, _ = resample_polyline(path, n_r)
    T, N, B = frames(path, [1.0, 0.0, 0.0])
    t = np.linspace(0, 1, n_r)
    hw = 0.030 * (1 - t) ** 0.85 + 0.0015
    hd = 0.026 * (1 - t) ** 0.7 + 0.0015
    BV = []
    for p, n_, b_, a_, d_ in zip(path, N, B, hw, hd):
        for s in circle_section(1.0, k, math.pi / k):
            BV.append(p + s[0] * a_ * n_ + s[1] * d_ * b_)
    BV = np.asarray(BV)
    BF = []
    for i in range(n_r - 1):
        for j in range(k):
            j1 = (j + 1) % k
            BF.append((i * k + j, i * k + j1, (i + 1) * k + j1, (i + 1) * k + j))
    tip = len(BV)
    BV = np.vstack([BV, path[-1] + T[-1] * 0.008])
    last = (n_r - 1) * k
    for j in range(k):
        BF.append((last + j, last + (j + 1) % k, tip, tip))
    mb.add(transform(BV, M), BF)


BROOCH_R = layout.BROOCH_R
RING_SECTION = layout.RING_SECTION


def brooch(mb_metal, mb_ring, center, facing, up, scale=1.33):
    """Spiked ring brooch with a raven skull. `facing` points out of the chest."""
    f = _unit(np.asarray(facing, dtype=np.float64))
    upv = _unit(np.asarray(up) - np.dot(up, f) * f)
    right = np.cross(upv, f)
    # local: X right, Y = -facing (into the chest), Z up
    M = basis_matrix(center, right, -f, upv)
    R = BROOCH_R
    circ = np.array([[R * math.cos(a), 0.0, R * math.sin(a)] for a in np.linspace(0, 2 * math.pi, 48, endpoint=False)])
    T = _unit(np.roll(circ, -1, 0) - np.roll(circ, 1, 0))
    Nn = _unit(circ)                         # radial
    Bb = np.cross(T, Nn)                     # +/- Y (depth)
    sec = chamfer_rect_section(*RING_SECTION)
    V, F, cst = sweep(circ, Nn, Bb, sec, closed=True)
    mb_ring.add(transform(V, M), F, region_uv(layout.HW_ISLANDS["ring"], cst))
    spikes = [(84, 0.18, 0.020), (0, 0.14, 0.018), (180, 0.15, 0.018), (268, 0.07, 0.016),
              (43, 0.085, 0.014), (137, 0.075, 0.014), (222, 0.09, 0.014), (318, 0.08, 0.014)]
    for ang, ln, w in spikes:
        a = math.radians(ang)
        d = np.array([math.cos(a), 0.0, math.sin(a)])
        base = d * (R - 0.016)
        side = np.array([0.0, 1.0, 0.0])
        Vd = np.vstack([base, base + d])
        tip_dir = transform(Vd, M)
        spike(mb_metal, tip_dir[0], tip_dir[1] - tip_dir[0], transform(np.vstack([[0, 0, 0], side]), M)[1] - M[:3, 3],
              ln * scale + 0.016, w * scale, w * scale * 0.75, 4)
    for ang in (0, 90, 180, 270):  # spokes behind the skull
        a = math.radians(ang + 45)
        p0 = np.array([0.04 * math.cos(a), 0.010, 0.04 * math.sin(a)])
        p1 = np.array([(R - 0.012) * math.cos(a), 0.010, (R - 0.012) * math.sin(a)])
        rod(mb_metal, transform(np.vstack([p0, p1]), M), 0.009, 4)
    S = np.diag([scale, scale, scale, 1.0])
    skull(mb_metal, M @ S)
    # attachment points left/right on the ring (world)
    return transform(np.array([[-R - 0.006, -0.004, -0.03], [R + 0.006, -0.004, -0.04]]), M)


def oring(mb, center, normal, r=0.045, w=0.011):
    n = _unit(np.asarray(normal, dtype=np.float64))
    a = _unit(np.cross(n, [0.0, 0.0, 1.0])) if abs(n[2]) < 0.9 else np.array([1.0, 0, 0])
    b = np.cross(n, a)
    M = basis_matrix(center, a, b, n)
    path = np.array([[r * math.cos(t), r * math.sin(t), 0.0] for t in np.linspace(0, 2 * math.pi, 22, endpoint=False)])
    V, F, _ = closed_planar_tube(path, circle_section(w, 6))
    mb.add(transform(V, M), F)


def tag(mb_metal, mb_face, top_center, facing):
    """Rectangular hanging tag with a small ring; the front face gets its own UV island."""
    f = _unit(np.asarray(facing, dtype=np.float64))
    up = np.array([0.0, 0.0, 1.0])
    right = _unit(np.cross(up, f))
    M = basis_matrix(top_center, right, f, up)
    w, h, d, rc = 0.08, 0.14, 0.016, 0.014
    outline = rounded_rect_path(w, h, rc, 3)[:, :2] + [0.0, -h / 2 - 0.025]
    n = len(outline)
    front = np.c_[outline[:, 0], np.full(n, d / 2), outline[:, 1]]
    back = np.c_[outline[:, 0], np.full(n, -d / 2), outline[:, 1]]
    V = np.vstack([front, back, [[0, d / 2, -h / 2 - 0.025], [0, -d / 2, -h / 2 - 0.025]]])
    F, UVc = [], []
    x0, y0, rw, rh = layout.HW_ISLANDS["tag_face"]

    def fuv(p):
        return ((x0 + (p[0] / w + 0.5) * rw) / PX, 1 - (y0 + (0.5 - (p[2] + h / 2 + 0.025) / h) * rh) / PX)

    for j in range(n):  # sides
        j1 = (j + 1) % n
        F.append((j, n + j, n + j1, j1))
        UVc.append([(0, 0)] * 4)
    sides = MeshBuilderFace(V, F, UVc)
    cf, cb = 2 * n, 2 * n + 1
    for j in range(n):  # front (outward, +f) and back caps
        j1 = (j + 1) % n
        sides.F.append((j1, j, cf, cf))
        sides.UV.append([fuv(V[j1]), fuv(V[j]), fuv(V[cf]), fuv(V[cf])])
        sides.F.append((n + j, n + j1, cb, cb))
        sides.UV.append([(0, 0)] * 4)
    Vw = transform(V, M)
    face_idx = [i for i, uvq in enumerate(sides.UV) if uvq[0] != (0, 0)]
    other_idx = [i for i in range(len(sides.F)) if i not in face_idx]
    mb_face.add(Vw, [sides.F[i] for i in face_idx], np.asarray([sides.UV[i] for i in face_idx]))
    mb_metal.add(Vw, [sides.F[i] for i in other_idx])
    oring(mb_metal, transform(np.array([[0, 0, -0.004]]), M)[0], right, 0.016, 0.0045)


class MeshBuilderFace:
    def __init__(self, V, F, UV):
        self.V, self.F, self.UV = V, F, UV


# --------------------------------------------------------------------------- chains
LINK_RC, LINK_RW, LINK_LS = 0.024, 0.0088, 0.034
LINK_PITCH = LINK_LS + 2 * (LINK_RC - LINK_RW)


def link_template():
    path = stadium_path(LINK_LS, LINK_RC, 12)
    V, F, cst = closed_planar_tube(path, circle_section(LINK_RW, 4, math.pi / 4))
    return V, F, region_uv(layout.HW_ISLANDS["link"], cst)


def chain(mb, bvh, a, b, sag, clearance=0.03, twist=0.0):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    x = np.linspace(0, 1, 60)[:, None]
    P = a + (b - a) * x + np.array([0, 0, -1.0]) * 4 * sag * x * (1 - x)
    for _ in range(3):
        P = push_out(bvh, P, clearance)
        P = smooth_polyline(P, 2, 0.5)
    P = push_out(bvh, P, clearance)
    length = np.linalg.norm(np.diff(P, axis=0), axis=1).sum()
    n_links = max(int(length / LINK_PITCH), 1)
    pts, _ = resample_polyline(P, n_links + 1)
    LV, LF, LUV = link_template()
    for i in range(n_links):
        c = 0.5 * (pts[i] + pts[i + 1])
        ax = _unit(pts[i + 1] - pts[i])
        loc, nrm, _, _ = bvh.find_nearest(Vector(c), 0.5)
        ref = np.asarray(nrm) if loc is not None else np.array([0, 0, 1.0])
        side = _unit(np.cross(ax, ref)) if np.linalg.norm(np.cross(ax, ref)) > 1e-6 else _unit(np.cross(ax, [0, 0, 1.0]))
        ang = twist + (math.pi / 2 if i % 2 else 0.0)
        up = np.cross(side, ax)
        s2 = side * math.cos(ang) + up * math.sin(ang)
        u2 = np.cross(ax, s2)
        M = basis_matrix(c, ax, s2, u2)
        mb.add(transform(LV, M), LF, LUV)
    return pts[-1], n_links
