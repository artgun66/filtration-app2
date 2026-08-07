"""Generate the PWA icons.

Drawn rather than shipped as binaries so the shape and colour stay editable and match
web/src/styles.css. A shield with a tick: recognisable at 48px on a home screen, which
rules out anything with text in it.

    python make_icons.py        # writes ../public/icons/
"""
import os

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "public", "icons")

ACCENT = (31, 111, 178)          # --safe-accent
FIELD = (255, 255, 255)
SIZES = [180, 192, 512]          # 180 is what iOS uses for the home screen


def shield(size, margin):
    """Shield outline as a polygon, in a `size` box inset by `margin`."""
    x0, y0 = margin, margin
    x1, y1 = size - margin, size - margin
    w = x1 - x0
    h = y1 - y0
    mid = x0 + w / 2
    waist = y0 + h * 0.55
    # ImageDraw closes the polygon itself, so the first point is not repeated.
    return [
        (mid, y0),
        (x1, y0 + h * 0.14),
        (x1, waist),
        (mid, y1),
        (x0, waist),
        (x0, y0 + h * 0.14),
    ]


def draw(size):
    # 4x supersampling: PIL has no antialiased polygon fill, and the diagonals of the
    # shield and the tick alias badly at 192px without it.
    scale = 4
    n = size * scale
    img = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # Rounded square background, so maskable icons have something to crop.
    d.rounded_rectangle([0, 0, n - 1, n - 1], radius=int(n * 0.22), fill=ACCENT)
    d.polygon(shield(n, int(n * 0.24)), fill=FIELD)

    # Tick, drawn as a thick two-segment line inside the shield.
    w = int(n * 0.055)
    d.line([(n * 0.40, n * 0.50), (n * 0.47, n * 0.585), (n * 0.62, n * 0.40)],
           fill=ACCENT, width=w, joint="curve")
    for pt in [(n * 0.40, n * 0.50), (n * 0.62, n * 0.40)]:
        d.ellipse([pt[0] - w / 2, pt[1] - w / 2, pt[0] + w / 2, pt[1] + w / 2],
                  fill=ACCENT)

    return img.resize((size, size), Image.LANCZOS)


os.makedirs(OUT, exist_ok=True)
for s in SIZES:
    path = os.path.join(OUT, f"icon-{s}.png")
    draw(s).save(path, "PNG")
    print(f"wrote {os.path.relpath(path, HERE)}")
