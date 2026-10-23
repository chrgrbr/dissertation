"""SDF v4 -- optimized 3D cutaway. Solid organ (v1-style mesh) with a tilted slice through it:
near half removed, opaque cut face shows the SDF (halo of field around the cross-section) + bold
zero-level set. Slice normal is tilted only PARTWAY toward the camera so we see organ AND slice.
Emits several tilt candidates. Run from repo root:
python sections/02_background/figures/shape_landscape/shared/gen_sdf_v4_cut.py
"""
import os
import numpy as np, nibabel as nib
from scipy import ndimage as ndi
from skimage import measure
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

SEG = "/Users/chris/Documents/CONVEA/MWE_demonstrator/data/cases/abdomen_mrct_0006/fixed_seg.nii.gz"
ELEV, AZIM = 15, -120
FACE, EDGE = "#bcd0e6", "#5f86ac"     # v1-style organ render
SL = "sections/02_background/figures/shape_landscape"
BAND = 10.0                            # voxels of exterior field shown around the cross-section

img = nib.as_closest_canonical(nib.load(SEG))
d = np.asarray(img.dataobj); liver = (d == 1)
xs, ys, zs = np.where(liver)
sl = tuple(slice(max(c.min() - 2, 0), c.max() + 3) for c in (xs, ys, zs))
liver = liver[sl].astype(float)
nx, ny, nz = liver.shape
b = liver > 0.5
verts, faces, _, _ = measure.marching_cubes(ndi.gaussian_filter(liver, 1.0), level=0.5, step_size=2)
tri = verts[faces]; cents = tri.mean(axis=1)

sdf = ndi.distance_transform_edt(~b) - ndi.distance_transform_edt(b)
P = np.argwhere(b).astype(float); cen = P.mean(0)

er, ar = np.radians(ELEV), np.radians(AZIM)
cam = np.array([np.cos(er) * np.cos(ar), np.cos(er) * np.sin(ar), np.sin(er)])
up = np.array([0, 0, 1.0])
perpH = np.cross(cam, up); perpH /= np.linalg.norm(perpH)   # horizontal in-view tilt axis
perpV = np.cross(cam, perpH); perpV /= np.linalg.norm(perpV)  # vertical in-view tilt axis


def render(theta_deg, perp, tag):
    th = np.radians(theta_deg)
    n = np.cos(th) * cam + np.sin(th) * perp; n /= np.linalg.norm(n)  # tilt partway from camera
    u = np.cross(n, up); u /= np.linalg.norm(u)
    v = np.cross(n, u)

    pu, pv = (P - cen) @ u, (P - cen) @ v
    mg = BAND + 4
    s = np.linspace(pu.min() - mg, pu.max() + mg, 200); t = np.linspace(pv.min() - mg, pv.max() + mg, 200)
    S, T = np.meshgrid(s, t, indexing="ij")
    coords = cen[None, None, :] + S[..., None] * u + T[..., None] * v
    C = ndi.map_coordinates(sdf, [coords[..., 0], coords[..., 1], coords[..., 2]],
                            order=1, mode="constant", cval=sdf.max())
    vmax = max(-C.min(), 1.0) * 1.15; norm = colors.Normalize(-vmax, vmax)
    rgba = cm.RdBu(norm(C)); rgba[..., 3] = np.where(C <= BAND, 1.0, 0.0)  # halo mask

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    keep = (cents - cen) @ n <= 0                       # drop near (camera-side) half
    m = Poly3DCollection(tri[keep], alpha=1.0)
    m.set_facecolor(FACE); m.set_edgecolor(EDGE); m.set_linewidth(0.12)
    ax.add_collection3d(m)

    ax.plot_surface(coords[..., 0], coords[..., 1], coords[..., 2], facecolors=rgba,
                    shade=False, rstride=1, cstride=1, antialiased=False)
    ds, dt = s[1] - s[0], t[1] - t[0]; eps = 1.0 * n
    for cn in measure.find_contours(C, 0.0):
        pc3 = cen + (s[0] + cn[:, 0] * ds)[:, None] * u + (t[0] + cn[:, 1] * dt)[:, None] * v + eps
        ax.plot(pc3[:, 0], pc3[:, 1], pc3[:, 2], color="k", lw=2.2, zorder=100)

    ax.set_xlim(0, nx); ax.set_ylim(0, ny); ax.set_zlim(0, nz)
    ax.view_init(elev=ELEV, azim=AZIM); ax.set_box_aspect((nx, ny, nz)); ax.set_axis_off()
    out = os.path.join(SL, "representations/4_sdf/versions", f"v4_{tag}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig); print("wrote", out)


render(45, perpH, "H45")
render(55, perpH, "H55")
render(45, perpV, "V45")
