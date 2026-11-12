"""Re-import the exported files and check them against roblox_parts.json; renders the re-imported OBJ
on the original mannequin as visual proof of placement.

blender -b --factory-startup -P verify_export.py -- EXPORT_DIR OUT_DIR
"""
import json
import os
import sys

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)

import bpy  # noqa: E402
import numpy as np  # noqa: E402

from cloakgen import bl_util  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:]
EXP, OUT = os.path.abspath(argv[0]), os.path.abspath(argv[1])
os.makedirs(OUT, exist_ok=True)
expected = json.load(open(os.path.join(EXP, "roblox_parts.json")))["parts"]


def measure(objs):
    """Bounding boxes in Roblox torso space; Blender scene here uses the build frame (Z up, faces -Y)."""
    out = {}
    for ob in objs:
        if ob.type != "MESH":
            continue
        mw = np.array(ob.matrix_world)
        co = np.array([v.co[:] for v in ob.data.vertices])
        co = co @ mw[:3, :3].T + mw[:3, 3]
        rb = np.stack([-co[:, 0], co[:, 2], co[:, 1]], 1)
        lo, hi = rb.min(0), rb.max(0)
        name = ob.name.split(".")[0]
        out[name] = dict(size=hi - lo, center=(hi + lo) / 2, tris=sum(len(p.vertices) - 2 for p in ob.data.polygons))
    return out


def check(label, objs):
    got = measure(objs)
    ok = True
    for name, exp in expected.items():
        g = got.get(name)
        if g is None:
            print("%s: MISSING %s (found %s)" % (label, name, sorted(got)))
            ok = False
            continue
        ds = np.abs(g["size"] - np.array(exp["size"])).max()
        dc = np.abs(g["center"] - np.array(exp["center"])).max()
        good = ds < 0.01 and dc < 0.01 and g["tris"] == exp["triangles"]
        ok &= good
        print("%s %-14s tris=%6d size_err=%.4f centre_err=%.4f %s" % (label, name, g["tris"], ds, dc, "OK" if good else "MISMATCH"))
    return ok


results = {}
for kind in ("fbx", "obj"):
    bl_util.clear_scene()
    path = os.path.join(EXP, "ForgottenOathsCloak." + kind)
    if kind == "fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    else:
        bpy.ops.wm.obj_import(filepath=path, forward_axis="Z", up_axis="Y")
    results[kind] = check(kind.upper(), list(bpy.context.scene.objects))

# visual proof: re-imported OBJ (torso-local) on the original R6 mannequin (torso centre at z = 3)
for ob in bpy.context.scene.objects:
    ob.location.z += 3.0
mann = bl_util.load_mannequin(os.path.join(SRC, "..", "reference", "R6_mannequin.obj"))
body = bl_util.simple_material("Body", (0.42, 0.42, 0.43), 0.6)
for ob in mann:
    bl_util.assign_material(ob, body)
tex = os.path.join(EXP, "..", "textures", "roblox_1024")
fab = bl_util.pbr_material("Fabric", tex, "Cloak_Fabric", True, False, True)
hw = bl_util.pbr_material("Hardware", tex, "Cloak_Hardware", False, True, False)
for ob in bpy.context.scene.objects:
    if ob.name.startswith("Cloak"):
        bl_util.assign_material(ob, hw if ob.name.startswith("CloakHardware") else fab)
bl_util.setup_world()
bl_util.studio_lights()
bl_util.shadow_floor()
bl_util.setup_render(360, 480, samples=16)
bpy.context.scene.view_settings.exposure = -3.2
bl_util.render_views(OUT, ["front", "back"], 360, 480)
print("VERIFY", "PASS" if all(results.values()) else "FAIL", results)
