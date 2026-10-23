"""SDF v7 -- 3D cutaway: same liver (locked pose) as the other renders, rendered as a FLAT
LAVENDER solid (no visible facets). A cutting plane slices it; the near cap is removed; the cut
face is an opaque, black-bordered SDF sheet (warm inside / lavender outside) with the black
zero-level cross-section. Scans several cut angles / positions.
Run from repo root:
python sections/02_background/figures/shape_landscape/shared/gen_sdf_v7_cut.py
"""
import os
import numpy as np, nibabel as nib
from scipy import ndimage as ndi
from skimage import measure
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

SEG = "/Users/chris/Documents/CONVEA/MWE_demonstrator/data/cases/abdomen_mrct_0006/fixed_seg.nii.gz"
ELEV, AZIM = 15, -120
BASE = np.array([0.737, 0.816, 0.902])            # #bcd0e6 lavender organ tone (matches mesh render)
SL = "sections/02_background/figures/shape_landscape"
CMAP = colors.LinearSegmentedColormap.from_list(
    "sdf", ["#c98b8b", "#f0d9d9", "#f4f4fb", "#dfe3f5", "#b9c2e8"])   # warm inside -> lavender outside
MARGIN = 7.0                                        # how far the slice sheet extends past the organ

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
perpH = np.cross(cam, up); perpH /= np.linalg.norm(perpH)
perpV = np.cross(cam, perpH); perpV /= np.linalg.norm(perpV)
light = cam * 0.4 + up * 0.55 + perpH * 0.35; light /= np.linalg.norm(light)

# outward per-face normals for gentle form shading (kept LIGHT so it reads as flat lavender)
e1 = tri[:, 1] - tri[:, 0]; e2 = tri[:, 2] - tri[:, 0]
fn = np.cross(e1, e2); fn /= (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-9)
fn *= np.sign(((cents - cen) * fn).sum(1))[:, None]


def render(theta_deg, delta, perp, halo, tag):
    th = np.radians(theta_deg)
    n = np.cos(th) * cam + np.sin(th) * perp; n /= np.linalg.norm(n)
    u = np.cross(n, up); u /= np.linalg.norm(u); v = np.cross(n, u)
    origin = cen + delta * n

    pu, pv = (P - cen) @ u, (P - cen) @ v
    s = np.linspace(pu.min() - MARGIN, pu.max() + MARGIN, 240)
    t = np.linspace(pv.min() - MARGIN, pv.max() + MARGIN, 240)
    S, T = np.meshgrid(s, t, indexing="ij")
    coords = origin[None, None, :] + S[..., None] * u + T[..., None] * v
    C = ndi.map_coordinates(sdf, [coords[..., 0], coords[..., 1], coords[..., 2]],
                            order=1, mode="constant", cval=sdf.max())
    vmax = max(-C.min(), 1.0) * 1.1; norm = colors.Normalize(-vmax, vmax)
    rgba = CMAP(norm(C))
    rgba[..., 3] = (C <= halo).astype(float)              # slice = cross-section (+ optional halo band)

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    # remaining organ (front slab removed) = flat lavender solid
    keep = (cents - origin) @ n <= 0
    inten = 0.86 + 0.14 * np.clip((fn[keep] * light).sum(1), 0, 1)   # light -> stays lavender
    fc = np.clip(BASE[None, :] * inten[:, None], 0, 1)
    ax.add_collection3d(Poly3DCollection(tri[keep], facecolors=fc, edgecolors="none", alpha=1.0))

    # SDF slice, clipped to the cut face
    ax.plot_surface(coords[..., 0], coords[..., 1], coords[..., 2], facecolors=rgba,
                    shade=False, rstride=1, cstride=1, antialiased=False, zorder=50)

    ds, dt = s[1] - s[0], t[1] - t[0]; eps = 0.6 * n
    # black slice border(s): outer edge of the shown sheet, then the zero-level cross-section
    borders = [(halo, 1.6)] if halo > 0 else []
    borders.append((0.0, 2.4))
    for lvl, lw in borders:
        for cn in measure.find_contours(C, lvl):
            p3 = origin + (s[0] + cn[:, 0] * ds)[:, None] * u + (t[0] + cn[:, 1] * dt)[:, None] * v + eps
            ax.plot(p3[:, 0], p3[:, 1], p3[:, 2], color="k", lw=lw, zorder=100)

    ax.set_xlim(0, nx); ax.set_ylim(0, ny); ax.set_zlim(0, nz)
    ax.view_init(elev=ELEV, azim=AZIM); ax.set_box_aspect((nx, ny, nz)); ax.set_axis_off()
    out = os.path.join(SL, "representations/4_sdf/versions", f"v7_{tag}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig); print("wrote", out)


# large delta = remove only a front slab -> recognizable liver dominates; halo=0 -> clean cut face
render(30, 14, perpH, 0.0, "H30_d14")
render(30, 14, perpH, 6.0, "H30_d14_halo")
render(45, 12, perpV, 0.0, "V45_d12")
render(45, 12, perpV, 6.0, "V45_d12_halo")
