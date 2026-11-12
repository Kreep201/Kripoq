"""Render views of the built cloak scene.

blender -b BUILD/cloak_scene.blend -P render_views.py -- OUT_DIR MODE(clay|final) WIDTH HEIGHT SAMPLES [views...]
Set CLOAK_TEX to the texture folder (defaults to ../textures/roblox_1024, i.e. what Roblox receives).
"""
import os
import sys

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)

import bpy  # noqa: E402

from cloakgen import bl_util  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:]
out_dir, mode = argv[0], argv[1]
width, height, samples = int(argv[2]), int(argv[3]), int(argv[4])
views = argv[5:] or ["front", "back", "left", "back34", "right"]
BUILD = os.path.dirname(bpy.data.filepath)
TEX = os.environ.get("CLOAK_TEX") or os.path.join(SRC, "..", "textures", "roblox_1024")


def clay_cloth():
    mat = bl_util.simple_material("ClayCloth", (0.03, 0.028, 0.036), 0.85)
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    t = nt.nodes.new("ShaderNodeTexImage")
    t.image = bpy.data.images.load(os.path.join(BUILD, "fabric_alpha.png"), check_existing=True)
    t.image.colorspace_settings.name = "Non-Color"
    nt.links.new(t.outputs["Color"], bsdf.inputs["Alpha"])
    return mat


body_mat = bl_util.simple_material("Body", (0.42, 0.42, 0.43), 0.6)
for ob in bpy.data.collections["Mannequin"].objects:
    bl_util.assign_material(ob, body_mat)
bl_util.face_decal()

if mode == "clay":
    cloth = clay_cloth()
    metal = bl_util.simple_material("ClayMetal", (0.5, 0.5, 0.5), 0.4, 1.0)
else:
    fabric_mat = bl_util.pbr_material("Fabric", TEX, "Cloak_Fabric", True, False, True)
    hw_mat = bl_util.pbr_material("Hardware", TEX, "Cloak_Hardware", False, True, False)

for col in bpy.data.collections:
    if not col.name.startswith("Cloak"):
        continue
    for ob in col.objects:
        is_hw = col.name.lower().endswith("hardware") or ob.name.startswith(("hw_", "CloakHardware"))
        if mode == "clay":
            bl_util.assign_material(ob, metal if is_hw else cloth)
        else:
            bl_util.assign_material(ob, hw_mat if is_hw else fabric_mat)

bl_util.setup_world()
bl_util.studio_lights()
bl_util.shadow_floor()
bl_util.setup_render(width, height, samples=samples)
bpy.context.scene.view_settings.exposure = -3.2
bl_util.render_views(out_dir, views, width, height)
