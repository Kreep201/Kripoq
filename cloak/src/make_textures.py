"""Step 3: generate the PBR texture sets. Usage: python3 make_textures.py BUILD_DIR TEXTURE_DIR"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cloakgen import tex_common as tc  # noqa: E402
from cloakgen import tex_fabric, tex_hardware  # noqa: E402

build = sys.argv[1]
out = sys.argv[2]
hi_dir = os.path.join(out, "source_2048")
lo_dir = os.path.join(out, "roblox_1024")
os.makedirs(hi_dir, exist_ok=True)
os.makedirs(lo_dir, exist_ok=True)
t0 = time.time()


def write_set(prefix, color, alpha, normal, rough, metal=None, emis=None):
    for d, size in ((hi_dir, 2048), (lo_dir, 1024)):
        c = color if size == 2048 else tc.downsample(color, size)
        a = None
        if alpha is not None:
            a = alpha if size == 2048 else tc.downsample_alpha(alpha, size)
        n = normal if size == 2048 else tc.downsample_normal(normal, size)
        tc.save_rgb(os.path.join(d, prefix + "_Color.png"), c, a)
        tc.save_normal(os.path.join(d, prefix + "_Normal.png"), n)
        tc.save_gray(os.path.join(d, prefix + "_Roughness.png"), rough if size == 2048 else tc.downsample(rough, size))
        if metal is not None:
            tc.save_gray(os.path.join(d, prefix + "_Metalness.png"), metal if size == 2048 else tc.downsample(metal, size))
        if emis is not None:
            tc.save_gray(os.path.join(d, prefix + "_Emissive.png"), emis if size == 2048 else tc.downsample(emis, size))


# fabric
alpha = np.load(os.path.join(build, "fabric_alpha.npy"))
metrics = dict(np.load(os.path.join(build, "fabric_metrics.npz")))
ao = np.load(os.path.join(build, "bake_fabric_ao.npy"))
col, nrm, rough, emis = tex_fabric.build(alpha, metrics, ao)
opaque = alpha > 0.5
col = tc.bleed(col, opaque)
nrm = tc.bleed(nrm, opaque)
rough = tc.bleed(rough, opaque)
write_set("Cloak_Fabric", col, alpha, nrm, rough, emis=emis)
print("fabric textures %.1fs" % (time.time() - t0))

# hardware
info = json.load(open(os.path.join(build, "build_info.json")))
h_ao = np.load(os.path.join(build, "bake_hw_ao.npy"))
h_edge = np.load(os.path.join(build, "bake_hw_edge.npy"))
h_cov = np.load(os.path.join(build, "bake_hw_cov.npy"))
col, nrm, rough, metal, covered = tex_hardware.build(h_ao, h_edge, h_cov, info)
col = tc.bleed(col, covered)
nrm = tc.bleed(nrm, covered)
rough = tc.bleed(rough, covered)
metal = tc.bleed(metal, covered)
write_set("Cloak_Hardware", col, None, nrm, rough, metal=metal)
print("all textures %.1fs" % (time.time() - t0))
