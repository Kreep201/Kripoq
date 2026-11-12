"""Step 4: export Roblox-ready meshes (FBX, OBJ, GLB) plus a source .blend.

blender -b BUILD/cloak_scene.blend -P export.py -- EXPORT_DIR TEXTURE_DIR

Geometry is written in studs, relative to the R6 Torso centre, in Roblox axes (Y up, character faces -Z).
"""
import json
import math
import os
import sys

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)

import bmesh  # noqa: E402
import bpy  # noqa: E402
import numpy as np  # noqa: E402

from cloakgen import bl_util  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:]
OUT = os.path.abspath(argv[0])
TEX = os.path.abspath(argv[1])
os.makedirs(OUT, exist_ok=True)

GROUPS = [("cape", "CloakCape"), ("cowl", "CloakCowl"), ("drapes", "CloakDrapes"), ("hardware", "CloakHardware")]
TORSO_CENTER_Z = 3.0

# drop render-only helpers
for ob in list(bpy.data.objects):
    if ob.name in ("FaceDecal", "ShadowFloor") or ob.type != "MESH":
        bpy.data.objects.remove(ob, do_unlink=True)

fabric_mat = bl_util.pbr_material("Cloak_Fabric", TEX, "Cloak_Fabric", True, False, True, tint=False,
                                  emissive_strength=1.0)
hw_mat = bl_util.pbr_material("Cloak_Hardware", TEX, "Cloak_Hardware", False, True, False)

parts = {}
for group, name in GROUPS:
    col = bpy.data.collections["Cloak_" + group]
    objs = [o for o in col.objects if o.type == "MESH"]
    for o in objs:
        bl_util.assign_material(o, hw_mat if group == "hardware" else fabric_mat)
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.join()
    ob = bpy.context.view_layer.objects.active
    ob.name = name
    ob.data.name = name
    me = ob.data
    me.transform(__import__("mathutils").Matrix.Translation((0.0, 0.0, -TORSO_CENTER_Z)))
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.triangulate(bm, faces=bm.faces[:], quad_method="BEAUTY", ngon_method="BEAUTY")
    bm.to_mesh(me)
    bm.free()
    me.update()
    co = np.empty(len(me.vertices) * 3, dtype=np.float64)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    rb = np.stack([-co[:, 0], co[:, 2], co[:, 1]], 1)  # Blender -> Roblox torso-local
    lo, hi = rb.min(0), rb.max(0)
    parts[name] = dict(triangles=len(me.polygons), size=[round(float(v), 4) for v in hi - lo],
                       center=[round(float(v), 4) for v in (hi + lo) / 2])
    print("%-14s tris=%6d size=%s center=%s" % (name, len(me.polygons), parts[name]["size"], parts[name]["center"]))

# keep only the four export meshes
for ob in list(bpy.data.objects):
    if ob.name not in dict(GROUPS).values():
        bpy.data.objects.remove(ob, do_unlink=True)
for col in list(bpy.data.collections):
    if col.name == "Mannequin":
        bpy.data.collections.remove(col)

bpy.ops.object.select_all(action="SELECT")
bpy.ops.export_scene.fbx(filepath=os.path.join(OUT, "ForgottenOathsCloak.fbx"), use_selection=True,
                         object_types={"MESH"}, axis_forward="Z", axis_up="Y", apply_unit_scale=True,
                         apply_scale_options="FBX_SCALE_UNITS", bake_space_transform=True,
                         mesh_smooth_type="FACE", use_tspace=True, use_mesh_modifiers=True,
                         path_mode="COPY", embed_textures=True, add_leaf_bones=False)
bpy.ops.wm.obj_export(filepath=os.path.join(OUT, "ForgottenOathsCloak.obj"), export_selected_objects=True,
                      forward_axis="Z", up_axis="Y", export_materials=True, export_normals=True, export_uv=True,
                      path_mode="RELATIVE", export_triangulated_mesh=True)
bpy.ops.export_scene.gltf(filepath=os.path.join(OUT, "ForgottenOathsCloak.glb"), export_format="GLB",
                          use_selection=True, export_tangents=True, export_yup=True, export_apply=True)

with open(os.path.join(OUT, "roblox_parts.json"), "w") as fh:
    json.dump({"units": "studs", "space": "R6 Torso local (X right, Y up, -Z forward)", "parts": parts}, fh, indent=2)

bpy.ops.file.make_paths_relative()
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "ForgottenOathsCloak_source.blend"))
print("export done")
