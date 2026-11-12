"""Armature and skin weights for the cloak (runs inside Blender).

Coordinates: the Blender build frame (Z up, character faces -Y, torso centre at z=3).
rig.json is written in R6 Torso-local Roblox coordinates (studs, X right, Y up, -Z forward).
"""
import json

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector, geometry
from mathutils.bvhtree import BVHTree

from . import rigspec

TORSO_CENTER = Vector((0.0, 0.0, 3.0))
# Blender build frame -> Roblox torso-local: (x, y, z) -> (-x, z - 3, y)
TO_ROBLOX = Matrix(((-1, 0, 0, 0), (0, 0, 1, -3.0), (0, 1, 0, 0), (0, 0, 0, 1)))
MESH_OF_SET = {"cape": "CloakCape", "drapes": "CloakDrapes", "hardware": "CloakHardware"}
HARDWARE_REACH = 0.2  # hardware closer than this to the cape (on its back half) rides on the cape


class ParamSurface:
    """Nearest-point lookup on a panel grid that returns the panel parameters (s, t)."""

    def __init__(self, part):
        rows, cols = part["rows"], part["cols"]
        P = part["verts"].reshape(rows, cols, 3)
        S = part["s"].reshape(rows, cols)
        T = part["t"].reshape(rows, cols)
        verts, tris, st = [], [], []
        for r in range(rows - 1):
            for c in range(cols - 1):
                quad = [(r, c), (r, c + 1), (r + 1, c + 1), (r + 1, c)]
                for tri in ((0, 1, 2), (0, 2, 3)):
                    idx = []
                    for q in tri:
                        rr, cc = quad[q]
                        idx.append(len(verts))
                        verts.append(Vector(P[rr, cc]))
                        st.append(Vector((S[rr, cc], T[rr, cc], 0.0)))
                    tris.append(idx)
        self.verts, self.tris, self.st = verts, tris, st
        self.bvh = BVHTree.FromPolygons(verts, tris)

    def nearest(self, p):
        loc, _, ti, dist = self.bvh.find_nearest(Vector(p))
        if loc is None:
            return None, None, None
        a, b, c = self.tris[ti]
        uvw = geometry.barycentric_transform(loc, self.verts[a], self.verts[b], self.verts[c],
                                             self.st[a], self.st[b], self.st[c])
        return float(uvw.x), float(uvw.y), dist


def set_weights(ob, per_vertex):
    """per_vertex: list (vertex order) of [(bone_name, weight), ...]."""
    groups = {}
    for vi, entries in enumerate(per_vertex):
        for name, w in entries:
            g = groups.get(name)
            if g is None:
                g = ob.vertex_groups.get(name) or ob.vertex_groups.new(name=name)
                groups[name] = g
            g.add([vi], float(w), "REPLACE")


def weight_fabric_part(ob, part, surfaces):
    """Skin a freshly created fabric object (vertex order == part grid order). Returns the set name or None."""
    name = part["name"]
    n = len(part["verts"])
    if name == "cape":
        per = [rigspec.weights_to_bones("CP", "Cape", rigspec.panel_weights("Cape", part["s"][i], part["t"][i]))
               for i in range(n)]
        set_name = "cape"
    elif name == "drape":
        per = [rigspec.weights_to_bones("DR", "Drape", rigspec.panel_weights("Drape", part["s"][i], part["t"][i]))
               for i in range(n)]
        set_name = "drapes"
    elif name in ("banner_back", "banner_front"):
        system, prefix, src = ("Cape", "CP", "cape") if name == "banner_back" else ("Drape", "DR", "drape")
        per = []
        for p in part["verts"]:
            s, t, _ = surfaces[src].nearest(p)
            per.append(rigspec.weights_to_bones(prefix, system, rigspec.panel_weights(system, s, t)))
        set_name = "cape" if name == "banner_back" else "drapes"
    elif name == "capelet":
        per = [[(rigspec.root_name("DR"), 1.0)]] * n
        set_name = "drapes"
    else:
        return None
    set_weights(ob, per)
    return set_name


def _components(me):
    """Connected vertex components (union-find over edges)."""
    parent = list(range(len(me.vertices)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for e in me.edges:
        a, b = find(e.vertices[0]), find(e.vertices[1])
        if a != b:
            parent[a] = b
    comps = {}
    for i in range(len(parent)):
        comps.setdefault(find(i), []).append(i)
    return list(comps.values())


def _hardware_weights(p, cape_surface):
    if p[1] > 0.25:  # back half of the body
        s, t, dist = cape_surface.nearest(p)
        if s is not None and dist < HARDWARE_REACH:
            return rigspec.weights_to_bones("HW", "Cape", rigspec.panel_weights("Cape", s, t))
    return [(rigspec.root_name("HW"), 1.0)]


def weight_hardware(objs, cape_surface):
    """Straps bend per vertex; rigid pieces (links, rings, buckles, tag...) get one weight set each."""
    for ob in objs:
        me = ob.data
        co = np.array([v.co[:] for v in me.vertices])
        per = [None] * len(co)
        if ob.name == "hw_leather":
            for i, p in enumerate(co):
                per[i] = _hardware_weights(p, cape_surface)
        else:
            for comp in _components(me):
                w = _hardware_weights(co[comp].mean(0), cape_surface)
                for i in comp:
                    per[i] = w
        set_weights(ob, per)


def used_bones(objs):
    used = set()
    for ob in objs:
        names = {g.index: g.name for g in ob.vertex_groups}
        for v in ob.data.vertices:
            for g in v.groups:
                if g.weight > 1e-4:
                    used.add(names[g.group])
    return used


def build_armature(joints, used, collection):
    """joints: {system: (K, J, 3) rest positions}. Creates one armature holding all prefixed bone sets."""
    arm = bpy.data.armatures.new("CloakRig")
    rig = bpy.data.objects.new("CloakRig", arm)
    collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    eb = arm.edit_bones

    def add(name, head, tail, parent, normal=None):
        b = eb.new(name)
        b.head, b.tail = Vector(head), Vector(tail)
        if normal is not None:
            b.align_roll(Vector(normal))
        b.parent = parent
        b.use_connect = False
        b.use_deform = True
        return b

    for set_name, spec in rigspec.BONE_SETS.items():
        prefix, system = spec["prefix"], spec["system"]
        root = add(rigspec.root_name(prefix), TORSO_CENTER, TORSO_CENTER + Vector((0, 0, 0.4)), None)
        J = joints[system]
        K, NJ, _ = J.shape
        for k in range(K):
            prev = root
            for j in range(NJ):
                if rigspec.is_pinned(system, j):
                    continue
                name = rigspec.bone_name(prefix, rigspec.joint_key(system, k, j))
                if not spec["chained"] and name not in used:
                    continue
                head = J[k, j]
                tail = J[k, j + 1] if j + 1 < NJ else head + (head - J[k, j - 1]) * 0.4
                across = J[min(k + 1, K - 1), j] - J[max(k - 1, 0), j]
                normal = np.cross(across, tail - head)
                b = add(name, head, tail, prev if spec["chained"] else root, normal)
                if spec["chained"]:
                    prev = b
    bpy.ops.object.mode_set(mode="OBJECT")
    return rig


def bind(objs, rig):
    for ob in objs:
        ob.parent = rig
        ob.matrix_parent_inverse = Matrix.Identity(4)
        mod = ob.modifiers.new("Armature", "ARMATURE")
        mod.object = rig


def _m_list(M):
    return [float(M[r][c]) for r in range(4) for c in range(4)]


def write_rig_json(path, joints, rig):
    out = {"space": "R6 Torso local (studs, X right, Y up, -Z forward)", "joints": [], "bones": []}
    for system, J in joints.items():
        K, NJ, _ = J.shape
        for k in range(K):
            for j in range(NJ):
                p = TO_ROBLOX @ Vector(J[k, j])
                out["joints"].append(dict(key=rigspec.joint_key(system, k, j), system=system, chain=k, index=j,
                                          pinned=rigspec.is_pinned(system, j), rest=[p.x, p.y, p.z]))
    prefix_mesh = {spec["prefix"]: MESH_OF_SET[s] for s, spec in rigspec.BONE_SETS.items()}
    for b in rig.data.bones:
        prefix, _, key = b.name.partition("_")
        out["bones"].append(dict(name=b.name, parent=b.parent.name if b.parent else None, mesh=prefix_mesh[prefix],
                                 key=None if key == "Root" else key, matrix=_m_list(TO_ROBLOX @ b.matrix_local)))
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    return out
