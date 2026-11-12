"""Builds the cloak scene inside Blender: mannequin, cloth panels (masked + double sided), hardware,
and bakes the occlusion / edge masks used by the texture generators.

Run via: blender -b --factory-startup -P build_scene.py -- BUILD_DIR [--no-bake]
"""
import json
import math
import os
import sys
import time

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)

import bmesh  # noqa: E402
import bpy  # noqa: E402
import numpy as np  # noqa: E402
from mathutils import Vector  # noqa: E402
from mathutils.bvhtree import BVHTree  # noqa: E402

from cloakgen import bl_util, fabric_geom, hardware, layout  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
BUILD = os.path.abspath(argv[0] if argv else os.path.join(SRC, "..", "build"))
BAKE = "--no-bake" not in argv


def new_collection(name):
    col = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(col)
    return col


def delete_transparent_faces(ob, alpha, threshold=0.03, pad=2):
    """Remove faces whose texture footprint is fully transparent (alpha is a [H,W] atlas, V up)."""
    me = ob.data
    n = alpha.shape[0]
    bm = bmesh.new()
    bm.from_mesh(me)
    uv = bm.loops.layers.uv.active
    kill = []
    for f in bm.faces:
        us = [l[uv].uv for l in f.loops]
        xs = [u.x * n for u in us]
        ys = [(1.0 - u.y) * n for u in us]
        x0, x1 = max(int(min(xs)) - pad, 0), min(int(max(xs)) + pad + 1, n)
        y0, y1 = max(int(min(ys)) - pad, 0), min(int(max(ys)) + pad + 1, n)
        if x1 <= x0 or y1 <= y0 or alpha[y0:y1, x0:x1].max() < threshold:
            kill.append(f)
    bmesh.ops.delete(bm, geom=kill, context="FACES")
    loose = [v for v in bm.verts if not v.link_faces]
    bmesh.ops.delete(bm, geom=loose, context="VERTS")
    bm.to_mesh(me)
    bm.free()
    return len(kill)


def combined_bvh(objs):
    verts, polys = [], []
    for ob in objs:
        base = len(verts)
        mw = ob.matrix_world
        verts += [mw @ v.co for v in ob.data.vertices]
        polys += [[base + i for i in p.vertices] for p in ob.data.polygons]
    return BVHTree.FromPolygons(verts, polys)


def push_out_of_body(ob, bvh, min_gap=0.012, max_probe=0.3):
    moved = 0
    for v in ob.data.vertices:
        loc, nrm, _, _ = bvh.find_nearest(v.co, max_probe)
        if loc is None:
            continue
        if (v.co - loc).dot(nrm) < min_gap:
            v.co = loc + nrm * min_gap
            moved += 1
    return moved


def snap_point(bvh, p, d, offset=0.0, reach=0.8):
    d = Vector(d).normalized()
    loc, n, _, _ = bvh.ray_cast(Vector(p) - d * reach, d, reach * 2.5)
    if loc is None:
        return np.asarray(p, dtype=np.float64)
    n = Vector(n)
    if n.dot(d) > 0:
        n = -n
    return np.asarray(loc + n * offset)


def nearest_on_path(path, target):
    i = int(np.argmin(np.linalg.norm(path - np.asarray(target), axis=1)))
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum = np.r_[0, np.cumsum(seg)]
    return cum[i] / cum[-1]


# --------------------------------------------------------------------------- hardware assembly
def build_hardware(fabric_objs, body_objs, col):
    hw = hardware
    banners = [o for o in fabric_objs if o.name.startswith("banner")]
    bvh_strap = combined_bvh([o for o in fabric_objs if o not in banners] + body_objs)
    info = {}

    mb_leather = hw.MeshBuilder()
    # bandolier: back (O-ring) -> over the right shoulder -> diagonally across the chest, under the front banner
    ctrl_a = [(-0.70, 1.30, 2.47), (-0.76, 1.20, 2.95), (-0.80, 1.10, 3.45), (-0.83, 1.05, 3.88),
              (-0.85, 0.80, 4.40), (-0.86, 0.25, 4.62), (-0.84, -0.30, 4.52), (-0.78, -0.80, 4.05),
              (-0.60, -1.00, 3.72), (-0.32, -1.00, 3.36), (0.00, -1.00, 3.00), (0.30, -1.00, 2.68),
              (0.60, -1.00, 2.40)]
    dirs_a = [(0, -1, 0), (0, -1, 0), (0, -1, 0), (0, -1, -0.3), (0, -0.5, -1), (0, 0, -1), (0, 0.4, -1),
              (0, 1, -0.5), (0, 1, 0), (0, 1, 0), (0, 1, 0), (0, 1, 0), (0, 1, 0)]
    path_a, nrm_a, len_a = hw.strap_path(bvh_strap, ctrl_a, dirs_a)
    hw.strap_mesh(mb_leather, path_a, nrm_a, [layout.HW_ISLANDS["strap_a1"], layout.HW_ISLANDS["strap_a2"]])
    # short chest strap from under the right arm into the bandolier
    ctrl_b = [(-1.03, -1.0, 2.86), (-0.62, -1.0, 2.98), (-0.20, -1.0, 3.12)]
    dirs_b = [(0, 1, 0)] * 3
    path_b, nrm_b, len_b = hw.strap_path(bvh_strap, ctrl_b, dirs_b)
    hw.strap_mesh(mb_leather, path_b, nrm_b, [layout.HW_ISLANDS["strap_b"]])
    leather = mb_leather.build("hw_leather", col)
    info["strap_lengths"] = {"strap_a1": len_a / 2, "strap_a2": len_a / 2, "strap_b": len_b}

    mb_metal = hw.MeshBuilder()
    mb_ring = hw.MeshBuilder()
    mb_face = hw.MeshBuilder()
    s1 = nearest_on_path(path_a, (-0.74, -0.75, 3.95))
    hw.buckle(mb_metal, *hw.strap_frame_at(path_a, nrm_a, s1))
    hw.keeper(mb_metal, hw.strap_frame_at(path_a, nrm_a, s1 + 0.045)[0], *hw.strap_frame_at(path_a, nrm_a, s1 + 0.045)[1:])
    hw.buckle(mb_metal, *hw.strap_frame_at(path_b, nrm_b, 0.52), scale=0.92)
    pk, tk, nk, bk = hw.strap_frame_at(path_b, nrm_b, 0.28)
    hw.keeper(mb_metal, pk, tk, nk, bk)

    bvh_all = combined_bvh(fabric_objs + body_objs + [leather])
    brooch_c = snap_point(bvh_all, (0.56, -1.2, 3.64), (0, 1, 0), 0.05)
    attach = hw.brooch(mb_metal, mb_ring, brooch_c, facing=(0.08, -1.0, 0.25), up=(0, 0, 1))
    info["brooch_center"] = [float(v) for v in brooch_c]

    ring_c = path_a[0] + (path_a[0] - path_a[2]) / np.linalg.norm(path_a[0] - path_a[2]) * 0.05
    ring_n = nrm_a[0]
    hw.oring(mb_metal, ring_c, ring_n, 0.045, 0.011)
    tag_top = ring_c + np.array([0, 0, -0.045]) + ring_n * 0.01

    links = hw.MeshBuilder()
    total_links = 0
    for a, b, sag in (
            (attach[0], np.asarray(hw.strap_frame_at(path_a, nrm_a, nearest_on_path(path_a, (-0.02, -0.7, 3.02)))[0]), 0.10),
            (attach[1], snap_point(bvh_all, (1.10, -0.55, 4.35), (0, 0.3, -1), 0.02), 0.30),
            (snap_point(bvh_all, (0.95, 0.55, 4.45), (0, 0, -1), 0.02), ring_c + np.array([0, 0, 0.03]), 0.22),
            (snap_point(bvh_all, (0.80, 1.5, 3.98), (0, -1, 0), 0.02),
             snap_point(bvh_all, (0.22, 1.5, 3.45), (0, -1, 0), 0.02), 0.13)):
        _, n_l = hw.chain(links, bvh_all, a, b, sag)
        total_links += n_l
    end, n_l = hw.chain(links, bvh_all, ring_c + np.array([0, 0, -0.04]), tag_top + np.array([0, 0.02, -0.28]), 0.0,
                        clearance=0.025)
    total_links += n_l
    hw.tag(mb_metal, mb_face, end + np.array([0, 0, -0.015]), facing=(-0.25, 1.0, 0.0))
    info["links"] = total_links

    metal = mb_metal.build("hw_metal", col)
    ring = mb_ring.build("hw_ring", col)
    tagface = mb_face.build("hw_tagface", col)
    chains = links.build("hw_links", col)
    for ob in (metal, ring, tagface, chains, leather):
        bm = bmesh.new()
        bm.from_mesh(ob.data)
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
        bm.to_mesh(ob.data)
        bm.free()
        ob.data.set_sharp_from_angle(angle=math.radians(42))
    smart_unwrap(metal, layout.HW_ISLANDS["metal"])
    return [metal, ring, tagface, chains, leather], info


def smart_unwrap(ob, region):
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = ob
    ob.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(55), island_margin=0.006, area_weight=0.0)
    bpy.ops.object.mode_set(mode="OBJECT")
    x0, y0, w, h = region
    sc = min(w, h) / layout.ATLAS_PX
    uv = np.zeros(len(ob.data.loops) * 2, dtype=np.float32)
    ob.data.uv_layers.active.data.foreach_get("uv", uv)
    uv = uv.reshape(-1, 2)
    uv[:, 0] = x0 / layout.ATLAS_PX + uv[:, 0] * sc
    uv[:, 1] = 1.0 - (y0 / layout.ATLAS_PX + (1.0 - uv[:, 1]) * sc)
    ob.data.uv_layers.active.data.foreach_set("uv", uv.ravel())


# --------------------------------------------------------------------------- baking
def bake_emit(objs, image, build_shader, samples):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = samples
    sc.render.bake.margin = 8
    sc.render.bake.use_clear = True
    saved = {ob.name: list(ob.data.materials) for ob in objs}
    mat = bpy.data.materials.new("bake_" + image.name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    emit = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    build_shader(nt, emit.inputs["Color"])
    img_node = nt.nodes.new("ShaderNodeTexImage")
    img_node.image = image
    nt.nodes.active = img_node
    for ob in objs:
        ob.data.materials.clear()
        ob.data.materials.append(mat)
    bpy.ops.object.select_all(action="DESELECT")
    for ob in objs:
        ob.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.bake(type="EMIT", margin=8, use_clear=True)
    for ob in objs:
        ob.data.materials.clear()
        for m in saved[ob.name]:
            ob.data.materials.append(m)


def ao_shader(distance):
    def build(nt, sink):
        ao = nt.nodes.new("ShaderNodeAmbientOcclusion")
        ao.samples = 8
        ao.inputs["Distance"].default_value = distance
        nt.links.new(ao.outputs["AO"], sink)
    return build


def edge_shader(radius):
    def build(nt, sink):
        bev = nt.nodes.new("ShaderNodeBevel")
        bev.samples = 8
        bev.inputs["Radius"].default_value = radius
        geo = nt.nodes.new("ShaderNodeNewGeometry")
        dot = nt.nodes.new("ShaderNodeVectorMath")
        dot.operation = "DOT_PRODUCT"
        nt.links.new(bev.outputs["Normal"], dot.inputs[0])
        nt.links.new(geo.outputs["Normal"], dot.inputs[1])
        inv = nt.nodes.new("ShaderNodeMath")
        inv.operation = "SUBTRACT"
        inv.inputs[0].default_value = 1.0
        nt.links.new(dot.outputs["Value"], inv.inputs[1])
        mul = nt.nodes.new("ShaderNodeMath")
        mul.operation = "MULTIPLY"
        mul.use_clamp = True
        mul.inputs[1].default_value = 6.0
        nt.links.new(inv.outputs["Value"], mul.inputs[0])
        nt.links.new(mul.outputs["Value"], sink)
    return build


def const_shader(value):
    def build(nt, sink):
        sink.default_value = (value, value, value, 1.0)
    return build


def new_bake_image(name, size=2048):
    im = bpy.data.images.new(name, size, size, alpha=False, float_buffer=True)
    im.colorspace_settings.name = "Non-Color"
    return im


def save_npy(image, path):
    """Store the raw baked values (top-left origin), bypassing colour management."""
    w, h = image.size
    arr = np.empty(w * h * 4, dtype=np.float32)
    image.pixels.foreach_get(arr)
    np.save(path, arr.reshape(h, w, 4)[::-1, :, 0].copy())


# --------------------------------------------------------------------------- main
def build():
    t0 = time.time()
    bl_util.clear_scene()
    alpha = np.load(os.path.join(BUILD, "fabric_alpha.npy"))
    mann = new_collection("Mannequin")
    body_objs = bl_util.load_mannequin(os.path.join(SRC, "..", "reference", "R6_mannequin.obj"))
    for ob in body_objs:
        bpy.context.scene.collection.objects.unlink(ob)
        mann.objects.link(ob)
    bvh_body = combined_bvh(body_objs)

    cols = {g: new_collection("Cloak_" + g) for g in ("cape", "cowl", "drapes", "hardware")}
    fabric_objs, sheets, stats = [], [], {}
    for part in fabric_geom.build_all_fabric():
        ob = bl_util.object_from_part(part, cols[part["group"]])
        if part["double_sided"]:
            bl_util.orient_outward(ob)
            removed = delete_transparent_faces(ob, alpha)
            sheets.append(ob)
        else:
            bl_util.recalc_normals_closed(ob)
            removed = 0
        moved = push_out_of_body(ob, bvh_body)
        fabric_objs.append(ob)
        stats[part["name"]] = (removed, moved)

    hw_objs, info = build_hardware(fabric_objs, body_objs, cols["hardware"])
    print("hardware:", info)

    if BAKE:
        tb = time.time()
        fab_ao = new_bake_image("bake_fabric_ao", 1024)
        bake_emit(fabric_objs, fab_ao, ao_shader(0.35), 8)
        save_npy(fab_ao, os.path.join(BUILD, "bake_fabric_ao.npy"))
        bakeable = [o for o in hw_objs if o.name != "hw_links"]
        hw_ao = new_bake_image("bake_hw_ao")
        bake_emit(bakeable, hw_ao, ao_shader(0.06), 8)
        save_npy(hw_ao, os.path.join(BUILD, "bake_hw_ao.npy"))
        hw_edge = new_bake_image("bake_hw_edge")
        bake_emit(bakeable, hw_edge, edge_shader(0.005), 4)
        save_npy(hw_edge, os.path.join(BUILD, "bake_hw_edge.npy"))
        hw_cov = new_bake_image("bake_hw_cov")
        bake_emit(hw_objs, hw_cov, const_shader(1.0), 1)
        save_npy(hw_cov, os.path.join(BUILD, "bake_hw_cov.npy"))
        print("bakes done in %.1fs" % (time.time() - tb))

    for ob in sheets:
        bl_util.make_double_sided(ob)
    for name, (removed, moved) in stats.items():
        ob = bpy.data.objects[name]
        print("  %-14s faces=%5d removed=%5d pushed=%4d" % (name, len(ob.data.polygons), removed, moved))
    tri_counts = {}
    for g, col in cols.items():
        tris = sum(sum(len(p.vertices) - 2 for p in ob.data.polygons) for ob in col.objects)
        tri_counts[g] = tris
        print("GROUP %-8s triangles=%d" % (g, tris))
    info["triangles"] = tri_counts
    with open(os.path.join(BUILD, "build_info.json"), "w") as fh:
        json.dump(info, fh, indent=2)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(BUILD, "cloak_scene.blend"))
    print("scene built in %.1fs" % (time.time() - t0))


build()
