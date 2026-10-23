"""Compile the four TikZ representation icons (SSM, latent, INR, deformation) and export
them to export_png/. Run AFTER editing any icon macro or the render preambles so the PNGs
pick up the correct thesis typeface (lmodern). The five geometric icons (01-05) come from
gen_representations.py instead.

Run (from repo root):
  python sections/02_background/figures/shape_landscape/shared/export_icons.py
Then run trim_export.py to refresh export_png/inline/.
"""
import subprocess, os
import numpy as np
from PIL import Image, ImageChops

SL = "sections/02_background/figures/shape_landscape"
DPI = 600
# (render driver, \sel value or None, output basename)
JOBS = [
    ("render_icon.tex",   "SSM",    "06_ssm"),
    ("render_icon.tex",   "Latent", "07_latent_code"),
    ("render_icon.tex",   "INR",    "08_inr"),
    ("render_deform.tex", None,     "09_deformation"),
]

def compile_pdf(driver, sel, jobname):
    if sel:
        arg = f"\\def\\sel{{{sel}}}\\input{{{driver}}}"
    else:
        arg = f"\\input{{{driver}}}"
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
         f"-jobname={jobname}", arg],
        cwd=SL, check=True, stdout=subprocess.DEVNULL)
    return os.path.join(SL, jobname + ".pdf")

def pdf_to_png(pdf, out):
    stem = pdf[:-4]
    subprocess.run(["pdftoppm", "-r", str(DPI), "-png", pdf, stem], check=True)
    raw = stem + "-1.png"
    im = Image.open(raw).convert("RGBA")
    bbox = ImageChops.difference(im.convert("L"), Image.new("L", im.size, 255)).getbbox()
    im = im.crop(bbox)
    arr = np.array(im)
    white = (arr[..., 0] > 250) & (arr[..., 1] > 250) & (arr[..., 2] > 250)
    arr[..., 3] = np.where(white, 0, arr[..., 3])
    Image.fromarray(arr).save(os.path.join(SL, "export_png", out + ".png"))
    for f in (raw, pdf, stem + ".aux", stem + ".log"):
        if os.path.exists(f):
            os.remove(f)

for driver, sel, out in JOBS:
    pdf = compile_pdf(driver, sel, "tmp_" + out)
    pdf_to_png(pdf, out)
    print(f"exported export_png/{out}.png")
