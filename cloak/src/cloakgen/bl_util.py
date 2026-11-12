"""Blender-side helpers (import only inside Blender)."""
import math
import os

import bmesh
import bpy
import numpy as np
from mathutils import Vector

from . import layout

ROBLOX_TO_BLENDER_OFFSET = (-9.0, 5.0, -8.0)  # torso centre x, feet y, torso centre z in the source OBJ


def roblox_to_blender(p):
    """Mannequin OBJ (Roblox studs, Y up, character faces -Z) -> Blender (Z up, faces -Y, feet at 0)."""
    x, y, z = p
    ox, oy, oz = ROBLOX_TO_BLENDER_OFFSET
    return (-(x - ox), z - oz, y - oy)


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def load_mannequin(path):
    verts, faces_by_group = [], {}
    group = None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            if ln.startswith("v "):
                verts.append(roblox_to_blender(tuple(map(float, ln.split()[1:4]))))
            elif ln.startswith("g "):
                group = ln[2:].strip()
                faces_by_group.setdefault(group, [])
            elif ln.startswith("f "):
                idx = [int(tok.split("/")[0]) - 1 for tok in ln.split()[1:]]
                faces_by_group.setdefault(group, []).append(idx)
    names = {"R61": "RightLeg", "R62": "LeftLeg", "R63": "RightArm", "R64": "LeftArm", "R65": "Torso", "R66": "Head"}
    objs = []
    for g, faces in faces_by_group.items():
        used = sorted({i for f in faces for i in f})
        remap = {o: n for n, o in enumerate(used)}
        me = bpy.data.meshes.new("R6_" + names.get(g, g))
        me.from_pydata([verts[i] for i in used], [], [[remap[i] for i in f] for f in faces])
        me.validate()
        ob = bpy.data.objects.new(me.name, me)
        bpy.context.scene.collection.objects.link(ob)
        for poly in me.polygons:
            poly.use_smooth = g == "R66"
        objs.append(ob)
    return objs


def part_uvs(part):
    isl = layout.FABRIC_ISLANDS[part["island"]] if part["island"] in layout.FABRIC_ISLANDS else None
    st = part["corner_st"]
    if isl is None:
        return st.reshape(-1, 2)
    u, v = layout.param_to_uv(isl, st[..., 0], st[..., 1])
    return np.stack([u, v], axis=-1).reshape(-1, 2)


def mesh_from_arrays(name, verts, faces, loop_uvs=None, smooth=True, collection=None):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in np.asarray(verts, dtype=np.float64)], [], [tuple(int(i) for i in f) for f in faces])
    if loop_uvs is not None:
        uvl = me.uv_layers.new(name="UVMap")
        uvl.data.foreach_set("uv", np.asarray(loop_uvs, dtype=np.float32).ravel())
    me.validate(clean_customdata=False)
    for poly in me.polygons:
        poly.use_smooth = smooth
    ob = bpy.data.objects.new(name, me)
    (collection or bpy.context.scene.collection).objects.link(ob)
    return ob


def object_from_part(part, collection=None):
    return mesh_from_arrays(part["name"], part["verts"], part["faces"], part_uvs(part), True, collection)


def orient_outward(ob, ref_y=0.0):
    """Flip a sheet so its normals face away from the body axis on average."""
    me = ob.data
    total = 0.0
    for poly in me.polygons:
        c = poly.center
        total += poly.normal.dot(Vector((c.x, c.y - ref_y, 0.0)))
    if total < 0:
        bm = bmesh.new()
        bm.from_mesh(me)
        bmesh.ops.reverse_faces(bm, faces=bm.faces[:])
        bm.to_mesh(me)
        bm.free()


def recalc_normals_closed(ob):
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.to_mesh(ob.data)
    bm.free()


def make_double_sided(ob, gap=0.004):
    """Append a flipped copy of every face, nudged inwards, so the sheet renders from both sides."""
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bm.normal_update()
    uv = bm.loops.layers.uv.active
    vmap = {}
    for v in bm.verts[:]:
        nv = bm.verts.new(v.co - v.normal * gap)
        vmap[v] = nv
    for f in bm.faces[:]:
        loops = list(f.loops)[::-1]
        nf = bm.faces.new([vmap[l.vert] for l in loops])
        nf.smooth = f.smooth
        nf.material_index = f.material_index
        if uv is not None:
            for nl, ol in zip(nf.loops, loops):
                nl[uv].uv = ol[uv].uv
    bm.to_mesh(ob.data)
    bm.free()


def assign_material(ob, mat):
    ob.data.materials.clear()
    ob.data.materials.append(mat)


def pbr_material(name, tex_dir, prefix, has_alpha, has_metal, has_emissive, emissive_tint=(0.78, 0.70, 1.0),
                 emissive_strength=1.6, tint=True):
    """Material built only from the exported maps, so renders show exactly what Roblox receives."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    uv = nt.nodes.new("ShaderNodeUVMap")
    uv.uv_map = "UVMap"

    def tex(fname, non_color):
        n = nt.nodes.new("ShaderNodeTexImage")
        img = bpy.data.images.load(os.path.join(tex_dir, fname), check_existing=True)
        if non_color:
            img.colorspace_settings.name = "Non-Color"
        n.image = img
        nt.links.new(uv.outputs["UV"], n.inputs["Vector"])
        return n

    col = tex(prefix + "_Color.png", False)
    nt.links.new(col.outputs["Color"], bsdf.inputs["Base Color"])
    if has_alpha:
        nt.links.new(col.outputs["Alpha"], bsdf.inputs["Alpha"])
    rough = tex(prefix + "_Roughness.png", True)
    nt.links.new(rough.outputs["Color"], bsdf.inputs["Roughness"])
    if has_metal:
        met = tex(prefix + "_Metalness.png", True)
        nt.links.new(met.outputs["Color"], bsdf.inputs["Metallic"])
    nm = tex(prefix + "_Normal.png", True)
    nmap = nt.nodes.new("ShaderNodeNormalMap")
    nt.links.new(nm.outputs["Color"], nmap.inputs["Color"])
    nt.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
    if has_emissive:
        em = tex(prefix + "_Emissive.png", True)
        if tint:
            mix = nt.nodes.new("ShaderNodeMix")
            mix.data_type = "RGBA"
            mix.blend_type = "MULTIPLY"
            mix.inputs["Factor"].default_value = 1.0
            mix.inputs[7].default_value = (*emissive_tint, 1.0)
            nt.links.new(em.outputs["Color"], mix.inputs[6])
            nt.links.new(mix.outputs[2], bsdf.inputs["Emission Color"])
        else:
            nt.links.new(em.outputs["Color"], bsdf.inputs["Emission Color"])
        bsdf.inputs["Emission Strength"].default_value = emissive_strength
    if hasattr(mat, "surface_render_method") and has_alpha:
        mat.surface_render_method = "DITHERED"
    return mat


def simple_material(name, color, roughness=0.6, metallic=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    return mat


def face_decal(head_radius=0.603):
    """Classic Roblox smile on a cylindrical patch in front of the head (render-only)."""
    size = 512
    img = np.zeros((size, size, 4), dtype=np.float32)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    span = 0.86  # studs covered by the decal
    X = (xx / size - 0.5) * span
    Z = (0.5 - yy / size) * span  # relative to decal centre (z = 4.5)
    ink = np.zeros((size, size), dtype=np.float32)
    for ex in (-0.135, 0.135):
        d = ((X - ex) / 0.042) ** 2 + ((Z - 0.075) / 0.075) ** 2
        ink = np.maximum(ink, np.clip((1.0 - d) * 6.0, 0, 1))
    r = np.sqrt(X ** 2 + (Z - 0.03) ** 2)
    smile = np.clip(1.0 - np.abs(r - 0.205) / 0.017, 0, 1) * (Z < -0.075) * (np.abs(X) < 0.17)
    ink = np.maximum(ink, np.clip(smile * 3.0, 0, 1))
    img[..., 3] = ink
    img = img[::-1].copy()  # Blender images are bottom-up
    im = bpy.data.images.new("FaceDecal", size, size, alpha=True)
    im.pixels.foreach_set(img.ravel())
    im.pack()
    ang = span / head_radius
    n_a, n_z = 24, 2
    verts, uvs, faces = [], [], []
    for iz in range(n_z):
        for ia in range(n_a):
            a = (ia / (n_a - 1) - 0.5) * ang
            z = 4.5 + (iz / (n_z - 1) - 0.5) * span
            verts.append((head_radius * math.sin(a), -head_radius * math.cos(a), z))
    loop_uvs = []
    for ia in range(n_a - 1):
        q = (ia, ia + 1, n_a + ia + 1, n_a + ia)
        faces.append(q)
        for k in q:
            loop_uvs.append(((k % n_a) / (n_a - 1), k // n_a))
    ob = mesh_from_arrays("FaceDecal", verts, faces, loop_uvs)
    mat = bpy.data.materials.new("FaceDecal")
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.01, 0.01, 0.01, 1)
    bsdf.inputs["Roughness"].default_value = 0.5
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = im
    nt.links.new(tex.outputs["Alpha"], bsdf.inputs["Alpha"])
    mat.blend_method = "HASHED" if hasattr(mat, "blend_method") else None
    assign_material(ob, mat)
    return ob


# --------------------------------------------------------------------------- rendering
def setup_render(width, height, samples=32, denoise=True, transparent=True):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = samples
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.adaptive_threshold = 0.02
    sc.cycles.use_denoising = denoise
    try:
        sc.cycles.denoiser = "OPENIMAGEDENOISE"
    except Exception:
        pass
    sc.cycles.max_bounces = 8
    sc.cycles.transparent_max_bounces = 24
    sc.cycles.diffuse_bounces = 3
    sc.cycles.glossy_bounces = 3
    sc.render.resolution_x = width
    sc.render.resolution_y = height
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = transparent
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA"
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.look = "AgX - Medium High Contrast"
    sc.view_settings.exposure = 0.0
    sc.render.threads_mode = "AUTO"


def setup_world(strength=1.0, color=(0.045, 0.045, 0.048)):
    world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (*color, 1.0)
    bg.inputs["Strength"].default_value = strength


def add_area_light(name, loc, target, power, size, color=(1, 1, 1)):
    ld = bpy.data.lights.new(name, "AREA")
    ld.energy = power
    ld.size = size
    ld.color = color
    ob = bpy.data.objects.new(name, ld)
    bpy.context.scene.collection.objects.link(ob)
    ob.location = loc
    direction = Vector(target) - Vector(loc)
    ob.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    return ob


def studio_lights():
    """Key/fill/rim rig parented to an empty so it can turn with each camera."""
    rig = bpy.data.objects.new("LightRig", None)
    bpy.context.scene.collection.objects.link(rig)
    lights = [
        add_area_light("Key", (-5.5, -7.5, 8.5), (0, 0, 2.6), 5200, 5.0, (1.0, 0.97, 0.94)),
        add_area_light("Fill", (7.5, -5.0, 4.5), (0, 0, 2.4), 1500, 6.0, (0.92, 0.95, 1.0)),
        add_area_light("RimL", (-6.0, 7.0, 6.5), (0, 0, 2.8), 3800, 4.0, (0.95, 0.93, 1.0)),
        add_area_light("RimR", (6.5, 6.5, 5.0), (0, 0, 2.6), 2600, 4.0, (1.0, 0.97, 0.95)),
        add_area_light("Top", (0.0, 0.0, 11.0), (0, 0, 2.5), 1400, 6.0),
    ]
    for ob in lights:
        ob.parent = rig
    return rig


def shadow_floor():
    bpy.ops.mesh.primitive_plane_add(size=40, location=(0, 0, 0))
    fl = bpy.context.active_object
    fl.name = "ShadowFloor"
    fl.is_shadow_catcher = True
    return fl


def add_camera(name, loc, target, ortho_scale=None, lens=85.0):
    cd = bpy.data.cameras.new(name)
    if ortho_scale:
        cd.type = "ORTHO"
        cd.ortho_scale = ortho_scale
    else:
        cd.lens = lens
    cd.clip_start = 0.1
    cd.clip_end = 100
    ob = bpy.data.objects.new(name, cd)
    bpy.context.scene.collection.objects.link(ob)
    ob.location = loc
    direction = Vector(target) - Vector(loc)
    ob.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    return ob


VIEWS = {
    # name: (location, target, ortho_scale or None, lens)
    "front": ((0.0, -30.0, 2.75), (0.0, 0.0, 2.75), 6.3, None),
    "back": ((0.0, 30.0, 2.75), (0.0, 0.0, 2.75), 6.3, None),
    "left": ((30.0, 0.0, 2.75), (0.0, 0.0, 2.75), 6.3, None),
    "right": ((-30.0, 0.0, 2.75), (0.0, 0.0, 2.75), 6.3, None),
    "back34": ((11.5, 14.5, 7.2), (0.0, 0.0, 2.55), None, 85.0),
    "front34": ((-10.5, -15.0, 6.0), (0.0, 0.0, 2.6), None, 85.0),
    "chest": ((0.0, -30.0, 3.45), (0.0, 0.0, 3.45), 2.6, None),
    "upperback": ((0.0, 30.0, 3.3), (0.0, 0.0, 3.3), 3.0, None),
    "brooch": ((0.62, -30.0, 3.58), (0.62, 0.0, 3.58), 0.95, None),
    "buckle": ((-0.55, -30.0, 3.45), (-0.55, 0.0, 3.45), 1.25, None),
    "hem": ((-0.75, 30.0, 1.3), (-0.75, 0.0, 1.3), 1.45, None),
    "rune": ((0.12, 30.0, 2.2), (0.12, 0.0, 2.2), 1.5, None),
}


def render_views(out_dir, names, width, height):
    os.makedirs(out_dir, exist_ok=True)
    sc = bpy.context.scene
    rig = bpy.data.objects.get("LightRig")
    paths = {}
    for nm in names:
        loc, tgt, ortho, lens = VIEWS[nm]
        cam = add_camera("Cam_" + nm, loc, tgt, ortho, lens or 85.0)
        if rig is not None:
            rig.rotation_euler = (0.0, 0.0, math.atan2(loc[0] - tgt[0], -(loc[1] - tgt[1])))
        sc.camera = cam
        sc.render.filepath = os.path.join(out_dir, nm + ".png")
        bpy.ops.render.render(write_still=True)
        paths[nm] = sc.render.filepath
    return paths
