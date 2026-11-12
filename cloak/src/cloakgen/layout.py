"""Texture atlas layouts shared by the geometry builder and the texture generators.

Rects are (x, y, w, h) in pixels of a 2048x2048 authoring image with a top-left origin.
Each fabric island is parametrised by (s, t): s runs across the panel, t runs "down" the
panel (top edge -> hem). Unrotated islands map s -> +x and t -> +y. Rotated islands
(rot=True) map t -> +x and s -> -y (a 90 degree rotation, never a mirror).
"""

ATLAS_PX = 2048

FABRIC_ISLANDS = {
    "cape": dict(rect=(0, 0, 2048, 1085), rot=False),
    "hood": dict(rect=(0, 1095, 690, 200), rot=False),
    "swag0": dict(rect=(0, 1300, 690, 106), rot=False),
    "swag1": dict(rect=(0, 1412, 690, 106), rot=False),
    "swag2": dict(rect=(0, 1524, 690, 106), rot=False),
    "roll0": dict(rect=(0, 1648, 1270, 96), rot=False),
    "roll1": dict(rect=(0, 1748, 1270, 96), rot=False),
    "roll2": dict(rect=(0, 1848, 1270, 96), rot=False),
    "roll3": dict(rect=(0, 1948, 1270, 96), rot=False),
    "drape": dict(rect=(700, 1095, 700, 184), rot=True),
    "banner_back": dict(rect=(700, 1290, 620, 100), rot=True),
    "banner_front": dict(rect=(700, 1400, 520, 84), rot=True),
    "capelet": dict(rect=(1410, 1095, 638, 540), rot=False),
}
for _i, _nm in enumerate(["fringe_roll0", "fringe_roll1", "fringe_roll2", "fringe_roll3",
                          "fringe_swag0", "fringe_swag1", "fringe_swag2"]):
    FABRIC_ISLANDS[_nm] = dict(rect=(1280, 1648 + 56 * _i, 768, 54), rot=False)

# Hardware atlas regions (x, y, w, h), top-left origin. "metal" holds the smart-unwrapped parts.
HW_ISLANDS = {
    "metal": (0, 0, 1740, 1740),
    "tag_face": (1760, 0, 280, 420),
    "ring": (0, 1750, 1024, 110),
    "link": (1040, 1750, 160, 80),
    "strap_a1": (0, 1866, 1020, 90),
    "strap_a2": (1028, 1866, 1020, 90),
    "strap_b": (0, 1960, 620, 84),
}

# Hardware dimensions shared by geometry (Blender) and texture generation (system Python).
STRAP_W = 0.15
STRAP_T = 0.026
BROOCH_R = 0.20
RING_SECTION = (0.056, 0.036, 0.009)  # radial width, depth, chamfer


def param_to_px(island, s, t, px=ATLAS_PX):
    x0, y0, w, h = island["rect"]
    if island["rot"]:
        return x0 + t * w, y0 + (1.0 - s) * h
    return x0 + s * w, y0 + t * h


def param_to_uv(island, s, t, px=ATLAS_PX):
    x, y = param_to_px(island, s, t, px)
    return x / px, 1.0 - y / px
