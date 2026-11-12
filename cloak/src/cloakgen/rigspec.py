"""Bone layout of the cloak (numpy only; shared by the Blender build, exporters and generators).

Joints sit on the fold-free base surface of a panel at panel parameters (s across, t down).
Pinned joints follow the torso exactly, so they get no bone of their own: their skin weights are
merged into the mesh's root bone. Every remaining bone influences the mesh, which matters because
Studio's importer drops zero-influence bones by default.
"""
import numpy as np

SYSTEMS = {
    # 9 chains around the back of the cape, 7 joints from the shoulders to the hem
    "Cape": dict(part="cape", s=np.linspace(0.035, 0.965, 9), t=np.linspace(0.0, 1.0, 7), pinned=1),
    # 2 chains down the front-left drape; the top 4 joints are held by the bandolier
    "Drape": dict(part="drape", s=np.array([0.17, 0.83]), t=np.linspace(0.0, 1.0, 7), pinned=4),
}

# Bone names must be unique across the whole FBX, so each mesh gets its own prefixed set.
# chained=False parents every bone straight to the root (for the hardware mirror of the cape
# chains, where only the joints that actually carry hardware get a bone).
BONE_SETS = {
    "cape": dict(prefix="CP", system="Cape", chained=True),
    "drapes": dict(prefix="DR", system="Drape", chained=True),
    "hardware": dict(prefix="HW", system="Cape", chained=False),
}


def joint_key(system, k, j):
    return "%s_%d_%d" % (system, k, j)


def bone_name(prefix, key):
    return "%s_%s" % (prefix, key)


def root_name(prefix):
    return prefix + "_Root"


def is_pinned(system, j):
    return j < SYSTEMS[system]["pinned"]


def sample_grid(G, s, t):
    """Bilinear sample of a (rows, cols, 3) grid at panel parameters (s, t) in [0, 1]."""
    rows, cols, _ = G.shape
    x = float(np.clip(s, 0.0, 1.0)) * (cols - 1)
    y = float(np.clip(t, 0.0, 1.0)) * (rows - 1)
    c0 = min(int(np.floor(x)), cols - 2)
    r0 = min(int(np.floor(y)), rows - 2)
    fx, fy = x - c0, y - r0
    a = G[r0, c0] * (1 - fx) + G[r0, c0 + 1] * fx
    b = G[r0 + 1, c0] * (1 - fx) + G[r0 + 1, c0 + 1] * fx
    return a * (1 - fy) + b * fy


def joint_positions(system, base):
    """(K, J, 3) joint rest positions (Blender build frame) sampled on a panel's base grid."""
    spec = SYSTEMS[system]
    return np.array([[sample_grid(base, s, t) for t in spec["t"]] for s in spec["s"]])


def _bracket(values, x):
    i = int(np.clip(np.searchsorted(values, x) - 1, 0, len(values) - 2))
    f = float(np.clip((x - values[i]) / (values[i + 1] - values[i]), 0.0, 1.0))
    return i, f


def panel_weights(system, s, t):
    """Bilinear joint weights of a panel point: [(k, j, w), ...] with at most 4 entries."""
    spec = SYSTEMS[system]
    k, fs = _bracket(spec["s"], s)
    j, ft = _bracket(spec["t"], t)
    out = []
    for dk, dj, w in ((0, 0, (1 - fs) * (1 - ft)), (1, 0, fs * (1 - ft)), (0, 1, (1 - fs) * ft), (1, 1, fs * ft)):
        if w > 1e-4:
            out.append((k + dk, j + dj, w))
    total = sum(w for _, _, w in out)
    return [(kk, jj, w / total) for kk, jj, w in out]


def weights_to_bones(prefix, system, joint_weights):
    """Map joint weights to bone names, merging pinned joints into the root bone."""
    acc = {}
    for k, j, w in joint_weights:
        name = root_name(prefix) if is_pinned(system, j) else bone_name(prefix, joint_key(system, k, j))
        acc[name] = acc.get(name, 0.0) + w
    return sorted(acc.items(), key=lambda kv: -kv[1])
