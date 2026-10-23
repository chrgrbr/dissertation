"""SDF v2 -- flat 2D liver cross-section as a distance field.
Diverging fill for sign + evenly spaced iso-distance rings + bold zero-level set (the shape).
Run from repo root: python sections/02_background/figures/shape_landscape/shared/gen_sdf_v2.py
"""
import os
import numpy as np, nibabel as nib
from scipy import ndimage as ndi
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors

SEG = "/Users/chris/Documents/CONVEA/MWE_demonstrator/data/cases/abdomen_mrct_0006/fixed_seg.nii.gz"
SL = "sections/02_background/figures/shape_landscape"

img = nib.as_closest_canonical(nib.load(SEG))
d = np.asarray(img.dataobj); liver = (d == 1)
xs, ys, zs = np.where(liver)
sl = tuple(slice(max(c.min() - 2, 0), c.max() + 3) for c in (xs, ys, zs))
b = liver[sl] > 0.5

# axial slice with the largest liver area -> a recognisable cross-section
areas = b.sum(axis=(0, 1))
k = int(areas.argmax())
m = b[:, :, k].T  # transpose so the outline sits upright-ish

# 2D signed distance (negative inside, positive outside)
sdf = ndi.distance_transform_edt(~m) - ndi.distance_transform_edt(m)
# pad a margin so outside rings have room
pad = 12
mpad = np.pad(m, pad)
sdf = ndi.distance_transform_edt(~mpad) - ndi.distance_transform_edt(mpad)

vmax = np.abs(sdf).max()
step = 4  # ring spacing in voxels
levels = np.arange(-np.floor(vmax / step) * step, vmax, step)

fig, ax = plt.subplots(figsize=(4, 4))
norm = colors.Normalize(-vmax, vmax)
ax.imshow(sdf, cmap=cm.RdBu, norm=norm, origin="lower")
# thin iso-distance rings
ax.contour(sdf, levels=levels[levels != 0], colors="k", linewidths=0.4, alpha=0.5)
# bold zero level set = the shape boundary
ax.contour(sdf, levels=[0], colors="k", linewidths=2.2)
ax.set_axis_off()
fig.savefig(os.path.join(SL, "representations/4_sdf/versions/v2_render.png"),
            dpi=200, bbox_inches="tight", pad_inches=0.02, transparent=True)
plt.close(fig)
print("wrote v2")
