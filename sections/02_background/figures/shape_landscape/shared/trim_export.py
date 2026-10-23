"""Normalise the inline representation icons: trim transparent margin to the content
bounding box, then re-pad by a uniform fraction. This removes the large empty borders
in the geometric renders (01-05 are ~70% margin) so every icon's content fills its frame
consistently. Output goes to export_png/inline/ (non-destructive; idempotent).

Run:  python sections/02_background/figures/shape_landscape/shared/trim_export.py
"""
import glob, os
from PIL import Image

SRC = "sections/02_background/figures/shape_landscape/export_png"
DST = os.path.join(SRC, "inline")
PAD = 0.05          # uniform padding as fraction of the content's larger side
# extra bottom padding (fraction of larger side) for icons whose label floats high and
# leaves the base looking cramped -- balances the latent-code icon (z sits well above the stack)
EXTRA_BOTTOM = {"07_latent_code.png": 0.14}

os.makedirs(DST, exist_ok=True)
for f in sorted(glob.glob(os.path.join(SRC, "0*_*.png"))):
    im = Image.open(f).convert("RGBA")
    bbox = im.split()[-1].getbbox()          # alpha bounding box = content
    im = im.crop(bbox)
    w, h = im.size
    pad = round(PAD * max(w, h))
    extra = round(EXTRA_BOTTOM.get(os.path.basename(f), 0.0) * max(w, h))
    out = Image.new("RGBA", (w + 2 * pad, h + 2 * pad + extra), (0, 0, 0, 0))
    out.paste(im, (pad, pad))                 # extra space accrues at the bottom
    out.save(os.path.join(DST, os.path.basename(f)))
    print(f"{os.path.basename(f):22s} content={w}x{h} -> padded={out.size[0]}x{out.size[1]}")
