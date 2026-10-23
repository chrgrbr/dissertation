"""Bake the CHOSEN slice composition with several colormaps for comparison.
plane: az=-14 el=0 pos=10 alpha=1 ; view elev=30 azim=-60. Thin frame; two-slope norm so the
exterior gradient is visible. Run from repo root:
python sections/02_background/figures/shape_landscape/shared/gen_sdf_colormaps.py
"""
import os
import numpy as np, nibabel as nib
from scipy import ndimage as ndi
from skimage import measure
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors, cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

SEG = "/Users/chris/Documents/CONVEA/MWE_demonstrator/data/cases/abdomen_mrct_0006/fixed_seg.nii.gz"
BASE = np.array([0.737, 0.816, 0.902])
SL = "sections/02_background/figures/shape_landscape"
MARGIN, GRID, PAD = 8.0, 240, 18
AZ, EL, POS = -14, 0, 10.0
VELEV, VAZIM = 30.0, -60.0

img = nib.as_closest_canonical(nib.load(SEG)); d = np.asarray(img.dataobj); liver = (d == 1)
xs, ys, zs = np.where(liver); sl = tuple(slice(max(c.min()-2, 0), c.max()+3) for c in (xs, ys, zs))
liver = np.pad(liver[sl].astype(float), PAD); nx, ny, nz = liver.shape; b = liver > 0.5
verts, faces, _, _ = measure.marching_cubes(ndi.gaussian_filter(liver, 1.6), level=0.5, step_size=3)
tri = verts[faces]; cents = tri.mean(1)
sdf = ndi.distance_transform_edt(~b) - ndi.distance_transform_edt(b)
P = np.argwhere(b).astype(float); cen = P.mean(0)
er, ar = np.radians(30), np.radians(-120)   # locked-pose basis for light/perp only
cam = np.array([np.cos(er)*np.cos(ar), np.cos(er)*np.sin(ar), np.sin(er)]); up = np.array([0, 0, 1.])
perpH = np.cross(cam, up); perpH /= np.linalg.norm(perpH)
light = cam*0.4 + up*0.55 + perpH*0.35; light /= np.linalg.norm(light)
e1 = tri[:, 1]-tri[:, 0]; e2 = tri[:, 2]-tri[:, 0]; fn = np.cross(e1, e2)
fn /= (np.linalg.norm(fn, axis=1, keepdims=True)+1e-9); fn *= np.sign(((cents-cen)*fn).sum(1))[:, None]

a, e = np.radians(AZ), np.radians(EL)
n = np.array([np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)]); n /= np.linalg.norm(n)
u = np.cross(up, n); u /= np.linalg.norm(u); v = np.cross(n, u)
origin = cen + POS*n
pu, pv = (P-cen)@u, (P-cen)@v
s = np.linspace(pu.min()-MARGIN, pu.max()+MARGIN, GRID); t = np.linspace(pv.min()-MARGIN, pv.max()+MARGIN, GRID)
S, T = np.meshgrid(s, t, indexing="ij"); coords = origin[None, None, :] + S[..., None]*u + T[..., None]*v
C = ndi.map_coordinates(sdf, [coords[..., 0], coords[..., 1], coords[..., 2]], order=1, mode="constant", cval=sdf.max())
keep = (cents-origin)@n <= 0
inten = 0.86 + 0.14*np.clip((fn[keep]*light).sum(1), 0, 1); FC = np.clip(BASE[None, :]*inten[:, None], 0, 1)
ds, dt = s[1]-s[0], t[1]-t[0]; eps = 0.6*n

# two-slope norm: full color on each side of zero (exterior no longer washed out)
tsnorm = colors.TwoSlopeNorm(vmin=C.min(), vcenter=0.0, vmax=max(C.max(), 1.0))

CMAPS = {
    "muted":     colors.LinearSegmentedColormap.from_list("m", ["#c98b8b", "#f0d9d9", "#f4f4fb", "#dfe3f5", "#b9c2e8"]),
    "rdbu":      matplotlib.colormaps["RdBu"],              # inside red -> outside blue (classic signed)
    "blueorange":colors.LinearSegmentedColormap.from_list("bo", ["#c1651a", "#f2c9a0", "#f7f7f7", "#a9c4de", "#2b6499"]),
    "teal":      colors.LinearSegmentedColormap.from_list("t", ["#b34b3f", "#e7b6ac", "#f6f6f4", "#a9cfc9", "#2f7d70"]),
}


def render(cmap, iso, outline, tag):
    fig = plt.figure(figsize=(4, 4)); ax = fig.add_subplot(111, projection="3d")
    rgba = cmap(tsnorm(C)); rgba[..., 3] = 1.0
    ax.add_collection3d(Poly3DCollection(tri[keep], facecolors=FC, edgecolors="none"))
    ax.plot_surface(coords[..., 0], coords[..., 1], coords[..., 2], facecolors=rgba,
                    shade=False, rstride=1, cstride=1, antialiased=True, zorder=50)
    # thin frame -- tiny camera-ward offset: enough to win the depth sort (no far-edge z-fight
    # jaggies) yet too small to project into a visible slab wall
    feps = 0.25 * n
    edge = np.array([origin+su*u+sv*v+feps for su, sv in
                     [(s[0], t[0]), (s[-1], t[0]), (s[-1], t[-1]), (s[0], t[-1]), (s[0], t[0])]])
    ax.plot(edge[:, 0], edge[:, 1], edge[:, 2], color="k", lw=0.8, zorder=90)
    if iso:
        for lvl in range(-20, 21, 5):
            if lvl == 0:
                continue
            for cn in measure.find_contours(C, lvl):
                p3 = origin+(s[0]+cn[:, 0]*ds)[:, None]*u+(t[0]+cn[:, 1]*dt)[:, None]*v+eps
                ax.plot(p3[:, 0], p3[:, 1], p3[:, 2], color="k", lw=0.35, alpha=0.4, zorder=95)
    if outline:
        for cn in measure.find_contours(C, 0.0):
            p3 = origin+(s[0]+cn[:, 0]*ds)[:, None]*u+(t[0]+cn[:, 1]*dt)[:, None]*v+eps
            ax.plot(p3[:, 0], p3[:, 1], p3[:, 2], color="k", lw=1.8, zorder=100)
    ax.set_xlim(0, nx); ax.set_ylim(0, ny); ax.set_zlim(0, nz)
    ax.view_init(elev=VELEV, azim=VAZIM); ax.set_box_aspect((nx, ny, nz)); ax.set_axis_off()
    out = os.path.join(SL, "representations/4_sdf/versions", f"v9_{tag}.png")
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0, transparent=True); plt.close(fig); print("wrote", out)


render(CMAPS["rdbu"], False, True,  "rdbu_outline")
render(CMAPS["rdbu"], False, False, "rdbu_nooutline")
