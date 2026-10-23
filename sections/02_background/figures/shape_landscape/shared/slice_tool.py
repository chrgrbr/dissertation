"""Interactive slice picker for the SDF cutaway icon.

Shows the liver (locked pose) as a flat lavender solid; a full RECTANGULAR SDF slice sheet
(interior AND exterior distances, opaque) cuts through it, black-bordered, with the black
zero-level cross-section. Sliders move the plane; 'Save' writes a PNG + the exact parameters.

Run from repo root:
    python sections/02_background/figures/shape_landscape/shared/slice_tool.py

Sliders:
    az / el  -- plane-normal orientation (world azimuth / elevation, degrees)
    pos      -- plane offset along its normal (voxels; + pushes toward camera -> keeps more organ)
    alpha    -- slice-sheet opacity (1 = opaque; lower it to see the organ through the sheet)
Buttons:
    Save     -> writes representations/4_sdf/versions/v8_custom.png + slice_params.txt
    Reset view -> returns the 3D camera to the locked pose (elev=15, azim=-120)
"""
import os
import numpy as np, nibabel as nib
from scipy import ndimage as ndi
from skimage import measure
import matplotlib
for _bk in ("MacOSX", "QtAgg", "Qt5Agg", "TkAgg"):
    try:
        matplotlib.use(_bk); break
    except Exception:
        continue
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

SEG = "/Users/chris/Documents/CONVEA/MWE_demonstrator/data/cases/abdomen_mrct_0006/fixed_seg.nii.gz"
ELEV, AZIM = 15, -120
BASE = np.array([0.737, 0.816, 0.902])            # #bcd0e6 lavender organ tone
SL = "sections/02_background/figures/shape_landscape"
CMAP = colors.LinearSegmentedColormap.from_list(
    "sdf", ["#c98b8b", "#f0d9d9", "#f4f4fb", "#dfe3f5", "#b9c2e8"])   # warm inside -> lavender outside
MARGIN = 8.0
GRID = 150
PAD = 18          # pad with background so the exterior SDF is REAL out to the sheet edge

# --- load + crop liver, then pad so exterior distances extend beyond the organ ---
img = nib.as_closest_canonical(nib.load(SEG))
d = np.asarray(img.dataobj); liver = (d == 1)
xs, ys, zs = np.where(liver)
sl = tuple(slice(max(c.min() - 2, 0), c.max() + 3) for c in (xs, ys, zs))
liver = np.pad(liver[sl].astype(float), PAD)
nx, ny, nz = liver.shape
b = liver > 0.5
verts, faces, _, _ = measure.marching_cubes(ndi.gaussian_filter(liver, 1.4), level=0.5, step_size=3)
tri = verts[faces]; cents = tri.mean(axis=1)
sdf = ndi.distance_transform_edt(~b) - ndi.distance_transform_edt(b)
P = np.argwhere(b).astype(float); cen = P.mean(0)

er, ar = np.radians(ELEV), np.radians(AZIM)
cam = np.array([np.cos(er) * np.cos(ar), np.cos(er) * np.sin(ar), np.sin(er)])
up = np.array([0, 0, 1.0])
perpH = np.cross(cam, up); perpH /= np.linalg.norm(perpH)
light = cam * 0.4 + up * 0.55 + perpH * 0.35; light /= np.linalg.norm(light)
e1 = tri[:, 1] - tri[:, 0]; e2 = tri[:, 2] - tri[:, 0]
fn = np.cross(e1, e2); fn /= (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-9)
fn *= np.sign(((cents - cen) * fn).sum(1))[:, None]


def plane_basis(az_deg, el_deg):
    a, e = np.radians(az_deg), np.radians(el_deg)
    n = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    n /= np.linalg.norm(n)
    ref = up if abs(n[2]) < 0.95 else perpH
    u = np.cross(ref, n); u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return n, u, v


def draw(ax, az, el, pos, alpha, elev=ELEV, azim=AZIM):
    ax.clear()
    n, u, v = plane_basis(az, el)
    origin = cen + pos * n

    pu, pv = (P - cen) @ u, (P - cen) @ v
    s = np.linspace(pu.min() - MARGIN, pu.max() + MARGIN, GRID)
    t = np.linspace(pv.min() - MARGIN, pv.max() + MARGIN, GRID)
    S, T = np.meshgrid(s, t, indexing="ij")
    coords = origin[None, None, :] + S[..., None] * u + T[..., None] * v
    C = ndi.map_coordinates(sdf, [coords[..., 0], coords[..., 1], coords[..., 2]],
                            order=1, mode="constant", cval=sdf.max())
    vmax = max(-C.min(), 1.0) * 1.3; norm = colors.Normalize(-vmax, vmax)
    rgba = CMAP(norm(C)); rgba[..., 3] = alpha

    keep = (cents - origin) @ n <= 0
    inten = 0.86 + 0.14 * np.clip((fn[keep] * light).sum(1), 0, 1)
    fc = np.clip(BASE[None, :] * inten[:, None], 0, 1)
    ax.add_collection3d(Poly3DCollection(tri[keep], facecolors=fc, edgecolors="none"))
    ax.plot_surface(coords[..., 0], coords[..., 1], coords[..., 2], facecolors=rgba,
                    shade=False, rstride=1, cstride=1, antialiased=False, zorder=50)

    ds, dt = s[1] - s[0], t[1] - t[0]; eps = 0.6 * n
    edge = np.array([origin + su * u + sv * v + eps for su, sv in
                     [(s[0], t[0]), (s[-1], t[0]), (s[-1], t[-1]), (s[0], t[-1]), (s[0], t[0])]])
    ax.plot(edge[:, 0], edge[:, 1], edge[:, 2], color="k", lw=1.6, zorder=90)
    for cn in measure.find_contours(C, 0.0):
        p3 = origin + (s[0] + cn[:, 0] * ds)[:, None] * u + (t[0] + cn[:, 1] * dt)[:, None] * v + eps
        ax.plot(p3[:, 0], p3[:, 1], p3[:, 2], color="k", lw=2.4, zorder=100)

    ax.set_xlim(0, nx); ax.set_ylim(0, ny); ax.set_zlim(0, nz)
    ax.view_init(elev=elev, azim=azim); ax.set_box_aspect((nx, ny, nz)); ax.set_axis_off()


fig = plt.figure(figsize=(7, 8))
ax = fig.add_axes([0.02, 0.24, 0.96, 0.74], projection="3d")
state = dict(az=AZIM, el=ELEV, pos=10.0, alpha=1.0)


def refresh(_=None):
    keep_view = (ax.elev, ax.azim)
    draw(ax, state["az"], state["el"], state["pos"], state["alpha"], *keep_view)
    fig.canvas.draw_idle()


sa = Slider(fig.add_axes([0.12, 0.17, 0.78, 0.025]), "az",    -180, 180, valinit=AZIM, valstep=2)
se = Slider(fig.add_axes([0.12, 0.13, 0.78, 0.025]), "el",     -90,  90, valinit=ELEV, valstep=2)
sp = Slider(fig.add_axes([0.12, 0.09, 0.78, 0.025]), "pos",    -20,  30, valinit=10.0, valstep=1)
sl_ = Slider(fig.add_axes([0.12, 0.05, 0.78, 0.025]), "alpha",   0,   1, valinit=1.0, valstep=0.05)
sa.on_changed(lambda x: (state.update(az=x), refresh()))
se.on_changed(lambda x: (state.update(el=x), refresh()))
sp.on_changed(lambda x: (state.update(pos=x), refresh()))
sl_.on_changed(lambda x: (state.update(alpha=x), refresh()))


def do_save(_):
    elev, azim = ax.elev, ax.azim                 # capture the CURRENT camera orbit
    out = os.path.join(SL, "representations/4_sdf/versions/v8_custom.png")
    fig2 = plt.figure(figsize=(4, 4))             # clean export figure (no sliders)
    ax2 = fig2.add_subplot(1, 1, 1, projection="3d")
    draw(ax2, state["az"], state["el"], state["pos"], state["alpha"], elev, azim)
    fig2.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig2)
    with open(os.path.join(SL, "shared/slice_params.txt"), "w") as f:
        f.write("az={az} el={el} pos={pos} alpha={alpha}".format(**state)
                + " view_elev={:.1f} view_azim={:.1f}\n".format(elev, azim))
    print("SAVED", out, "| plane:", state, "| view elev/azim: %.1f/%.1f" % (elev, azim))


def do_reset(_):
    ax.view_init(elev=ELEV, azim=AZIM); refresh()


bs = Button(fig.add_axes([0.60, 0.005, 0.18, 0.035]), "Save PNG+params")
br = Button(fig.add_axes([0.80, 0.005, 0.15, 0.035]), "Reset view")
bs.on_clicked(do_save); br.on_clicked(do_reset)

refresh()
print("Interactive slice tool ready. Drag sliders; drag on the 3D view to orbit; 'Save' to export.")
plt.show()
