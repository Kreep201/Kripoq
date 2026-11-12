"""Compose a presentation sheet in the layout of the concept board. Usage: compose_sheet.py VIEWS_DIR OUT_PNG"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BG = (38, 38, 40)
INK = (205, 202, 198)
DIM = (150, 147, 144)
LINE = (88, 86, 84)


def font(size):
    for name in ("DejaVuSerif.ttf", "DejaVuSans.ttf"):
        for base in ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/dejavu"):
            p = os.path.join(base, name)
            if os.path.exists(p):
                return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def view(views, name, size, crop=True):
    im = Image.open(os.path.join(views, name + ".png")).convert("RGBA")
    if crop:
        a = np.asarray(im)[..., 3]
        ys, xs = np.nonzero(a > 10)
        if len(xs):
            pad = int(0.04 * max(im.size))
            box = (max(xs.min() - pad, 0), max(ys.min() - pad, 0),
                   min(xs.max() + pad, im.width), min(ys.max() + pad, im.height))
            im = im.crop(box)
    bg = Image.new("RGBA", im.size, BG + (255,))
    bg.alpha_composite(im)
    bg = bg.convert("RGB")
    bg.thumbnail(size, Image.LANCZOS)
    canvas = Image.new("RGB", size, BG)
    canvas.paste(bg, ((size[0] - bg.width) // 2, (size[1] - bg.height) // 2))
    return canvas


def cover(views, name, size):
    """Fill the cell completely (centre crop) - used for the square close-up renders."""
    im = Image.open(os.path.join(views, name + ".png")).convert("RGBA")
    bg = Image.new("RGBA", im.size, BG + (255,))
    bg.alpha_composite(im)
    bg = bg.convert("RGB")
    s = max(size[0] / bg.width, size[1] / bg.height)
    bg = bg.resize((int(bg.width * s + 0.5), int(bg.height * s + 0.5)), Image.LANCZOS)
    x = (bg.width - size[0]) // 2
    y = (bg.height - size[1]) // 2
    return bg.crop((x, y, x + size[0], y + size[1]))


def heading(d, x, y, w, text, f):
    tw = d.textlength(text, font=f)
    d.text((x + (w - tw) / 2, y), text, font=f, fill=INK)
    d.line([(x + w * 0.1, y + f.size + 10), (x + w * 0.9, y + f.size + 10)], fill=LINE, width=1)


def sp(text):
    return " ".join(text)


def main(views, out):
    W, H = 1800, 1960
    sheet = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(sheet)
    f_big, f_mid, f_small = font(32), font(22), font(16)

    big = (560, 760)
    for i, (nm, title) in enumerate((("front", "FRONT VIEW"), ("back", "BACK VIEW"), ("left", "SIDE VIEW (LEFT)"))):
        x = 30 + i * (big[0] + 25)
        heading(d, x, 20, big[0], sp(title), f_mid)
        sheet.paste(view(views, nm, big, crop=False), (x, 70))

    y2 = 860
    d.line([(30, y2), (W - 30, y2)], fill=LINE, width=1)
    d.text((30, y2 + 16), sp("DETAILS"), font=f_mid, fill=INK)
    det = (250, 330)
    for i, (nm, title) in enumerate((("brooch", "BROOCH & CHAINS"), ("hem", "FABRIC TEXTURE"),
                                     ("rune", "RUNE DETAIL"), ("buckle", "LEATHER STRAP & BUCKLE"))):
        x = 30 + i * (det[0] + 14)
        sheet.paste(cover(views, nm, det), (x, y2 + 62))
        d.rectangle([x, y2 + 62, x + det[0] - 1, y2 + 62 + det[1] - 1], outline=LINE)
        tw = d.textlength(title, font=f_small)
        d.text((x + (det[0] - tw) / 2, y2 + 62 + det[1] + 10), title, font=f_small, fill=DIM)
    for x, w, nm, title in ((1110, 330, "back34", "BACK ANGLE (3/4)"), (1460, 310, "right", "SIDE VIEW (RIGHT)")):
        heading(d, x, y2 + 16, w, title, f_small)
        sheet.paste(view(views, nm, (w, 400)), (x, y2 + 58))

    y3 = 1330
    d.line([(30, y3), (1080, y3)], fill=LINE, width=1)
    d.text((30, y3 + 16), sp("COLOR PALETTE"), font=f_mid, fill=INK)
    swatches = [(26, 25, 30), (40, 39, 50), (74, 63, 75), (61, 42, 31), (153, 149, 143), (156, 142, 216)]
    for i, c in enumerate(swatches):
        cx, cy = 75 + i * 100, y3 + 120
        d.ellipse([cx - 38, cy - 38, cx + 38, cy + 38], fill=c, outline=(110, 108, 106), width=2)
    for k, t in enumerate((sp("THE WEIGHT"), sp("OF FORGOTTEN OATHS"))):
        tw = d.textlength(t, font=f_big)
        d.text((30 + (1050 - tw) / 2, y3 + 230 + k * 48), t, font=f_big, fill=INK)
    d.line([(30 + 1050 / 2 - 70, y3 + 340), (30 + 1050 / 2 + 70, y3 + 340)], fill=LINE, width=1)
    heading(d, 1150, y3 + 12, 620, "FRONT ANGLE (3/4)", f_small)
    sheet.paste(view(views, "front34", (620, 540)), (1150, y3 + 56))
    sheet = sheet.crop((0, 0, W, y3 + 610))
    sheet.save(out)
    print("sheet", out, sheet.size)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
