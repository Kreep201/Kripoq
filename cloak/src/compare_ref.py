"""Place renders next to the matching reference views for side-by-side comparison."""
import os
import sys

from PIL import Image, ImageDraw

REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reference", "concept_sheet.png")
REF_BOXES = {
    "front": (15, 25, 315, 455),
    "back": (325, 25, 650, 455),
    "left": (670, 25, 945, 470),
    "back34": (580, 510, 790, 795),
    "right": (785, 510, 945, 795),
}
BG = (42, 42, 42)


def flatten(path, size):
    im = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", im.size, BG + (255,))
    bg.alpha_composite(im)
    bg = bg.convert("RGB")
    bg.thumbnail(size, Image.LANCZOS)
    return bg


def main(render_dir, out_path, views):
    ref = Image.open(REF).convert("RGB")
    cell_h = 480
    cells = []
    for v in views:
        rp = os.path.join(render_dir, v + ".png")
        if not os.path.exists(rp):
            continue
        r = flatten(rp, (400, cell_h))
        box = REF_BOXES.get(v)
        if box:
            rc = ref.crop(box)
            rc = rc.resize((int(rc.width * cell_h / rc.height), cell_h), Image.LANCZOS)
        else:
            rc = Image.new("RGB", (10, cell_h), BG)
        pair = Image.new("RGB", (rc.width + r.width + 12, cell_h + 26), BG)
        pair.paste(rc, (0, 26))
        pair.paste(r, (rc.width + 12, 26))
        d = ImageDraw.Draw(pair)
        d.text((6, 6), v.upper() + "  reference | build", fill=(220, 220, 220))
        cells.append(pair)
    W = max(c.width for c in cells)
    rows = []
    # two pairs per row
    for i in range(0, len(cells), 2):
        row = cells[i:i + 2]
        w = sum(c.width for c in row) + 20 * (len(row) - 1)
        canvas = Image.new("RGB", (w, cell_h + 26), BG)
        x = 0
        for c in row:
            canvas.paste(c, (x, 0))
            x += c.width + 20
        rows.append(canvas)
    W = max(r.width for r in rows)
    sheet = Image.new("RGB", (W, sum(r.height for r in rows) + 10 * len(rows)), BG)
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height + 10
    sheet.save(out_path)
    print("saved", out_path, sheet.size)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:] or ["front", "back", "left", "back34", "right"])
