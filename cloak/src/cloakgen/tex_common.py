"""Shared helpers for texture generation (system Python: numpy, scipy, opencv, Pillow)."""
import cv2
import numpy as np
from PIL import Image
from scipy import ndimage


def rgb(hexstr):
    h = hexstr.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float32) / 255.0


def lerp(a, b, t):
    t = np.asarray(t, dtype=np.float32)
    if t.ndim == 2 and np.ndim(a) >= 1 and np.shape(a)[-1] == 3:
        t = t[..., None]
    return a + (b - a) * t


def clamp01(x):
    return np.clip(x, 0.0, 1.0)


def normal_from_height(h, px_size, strength=1.0):
    """Tangent-space OpenGL normal map (+Y = up in UV space) from a height field in studs."""
    dhdx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) * 0.5 / px_size
    dhdy = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) * 0.5 / px_size
    n = np.stack([-dhdx * strength, dhdy * strength, np.ones_like(h)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return n


def bleed(img, mask):
    """Copy the nearest valid colour into invalid pixels (prevents dark halos when mip-mapping)."""
    if mask.all():
        return img
    idx = ndimage.distance_transform_edt(~mask, return_distances=False, return_indices=True)
    return img[idx[0], idx[1]]


def to_u8(x):
    return np.clip(np.asarray(x) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def save_rgb(path, rgb_img, alpha=None):
    arr = to_u8(rgb_img)
    if alpha is not None:
        arr = np.dstack([arr, to_u8(alpha)])
        Image.fromarray(arr, "RGBA").save(path, optimize=True)
    else:
        Image.fromarray(arr, "RGB").save(path, optimize=True)


def save_gray(path, img):
    Image.fromarray(to_u8(img), "L").save(path, optimize=True)


def save_normal(path, n):
    save_rgb(path, n * 0.5 + 0.5)


def downsample(img, size):
    return cv2.resize(img.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)


def downsample_normal(n, size):
    m = downsample(n, size)
    return m / (np.linalg.norm(m, axis=-1, keepdims=True) + 1e-8)


def downsample_alpha(a, size):
    return clamp01(downsample(a, size))
