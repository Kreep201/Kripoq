"""Step 1: generate fabric tatter masks (system Python). Usage: python3 make_masks.py BUILD_DIR"""
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cloakgen import fabric_geom, fabric_masks  # noqa: E402

out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "build")
os.makedirs(out, exist_ok=True)
t0 = time.time()
parts = fabric_geom.build_all_fabric()
alpha, metrics = fabric_masks.build_masks(parts)
np.save(os.path.join(out, "fabric_alpha.npy"), alpha)
cv2.imwrite(os.path.join(out, "fabric_alpha.png"), np.clip(alpha * 255 + 0.5, 0, 255).astype(np.uint8))
np.savez_compressed(os.path.join(out, "fabric_metrics.npz"),
                    **{"%s__%s" % (isl, key): val for isl, d in metrics.items() for key, val in d.items()})
print("masks done in %.1fs, coverage %.3f" % (time.time() - t0, float(alpha.mean())))
