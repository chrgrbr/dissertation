"""Interactive pose picker for the SDF-slice / camera of the shape-landscape figure.
Run interactively (NOT headless):  python sections/02_background/figures/pose_picker.py
- Drag with the mouse to set the CAMERA (rotate the 3D view).
- Sliders set the SDF cutting-plane: normal azimuth, normal elevation, offset along normal.
On window close it prints the chosen camera + plane parameters to paste back.
"""
import numpy as np, nibabel as nib
from scipy import ndimage as ndi
from skimage import measure
import matplotlib.pyplot as plt          # default (interactive) backend on macOS
from matplotlib import cm, colors
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

SEG = "/Users/chris/Documents/CONVEA/MWE_demonstrator/data/cases/abdomen_mrct_0006/fixed_seg.nii.gz"
ELEV0, AZIM0 = 15, -120

# --- data ---
img = nib.as_closest_canonical(nib.load(SEG)); d = np.asarray(img.dataobj); liver = (d == 1)
xs, ys, zs = np.where(liver); sl = tuple(slice(max(c.min() - 2, 0), c.max() + 3) for c in (xs, ys, zs))
liver = liver[sl].astype(float); nx, ny, nz = liver.shape; b = liver > 0.5
sdf = ndi.distance_transform_edt(~b) - ndi.distance_transform_edt(b)
verts, faces, _, _ = measure.marching_cubes(ndi.gaussian_filter(liver, 1.0), level=0.5, step_size=2)
tri = verts[faces]
P = np.argwhere(b).astype(float); cen = P.mean(0)
w, V = np.linalg.eigh(np.cov((P - cen).T)); order = w.argsort()[::-1]
v1 = V[:, order[0]]                                   # liver longest axis (= initial plane normal? no -> normal is v3)
n0 = V[:, order[2]]                                   # PCA plane normal (thin direction) -> plane spans the 2 long axes
paz0 = np.degrees(np.arctan2(n0[1], n0[0])); pel0 = np.degrees(np.arcsin(np.clip(n0[2], -1, 1)))
half = 0.5 * np.linalg.norm([nx, ny, nz])

cmap = cm.RdBu

def basis(n):
    n = n / (np.linalg.norm(n) + 1e-9)
    a = np.array([0, 0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0, 0])
    e1 = np.cross(n, a); e1 /= np.linalg.norm(e1); e2 = np.cross(n, e1)
    return n, e1, e2

def slice_coords(paz, pel, off):
    n = np.array([np.cos(np.radians(pel)) * np.cos(np.radians(paz)),
                  np.cos(np.radians(pel)) * np.sin(np.radians(paz)),
                  np.sin(np.radians(pel))])
    n, e1, e2 = basis(n)
    g = np.linspace(-half, half, 120); U, Vv = np.meshgrid(g, g, indexing="ij")
    C = cen + off * n
    coords = C[None, None, :] + U[..., None] * e1 + Vv[..., None] * e2
    vals = ndi.map_coordinates(sdf, [coords[..., 0], coords[..., 1], coords[..., 2]],
                               order=1, mode="constant", cval=sdf.max())
    return coords, vals

fig = plt.figure(figsize=(8, 8))
ax = fig.add_axes([0.0, 0.18, 1.0, 0.80], projection="3d")
mesh = Poly3DCollection(tri, alpha=0.12); mesh.set_facecolor("#999"); mesh.set_edgecolor("none")
ax.add_collection3d(mesh)
ax.set_xlim(0, nx); ax.set_ylim(0, ny); ax.set_zlim(0, nz)
ax.set_box_aspect((nx, ny, nz)); ax.view_init(elev=ELEV0, azim=AZIM0); ax.set_axis_off()
surf = [None]

def redraw(_=None):
    if surf[0] is not None:
        surf[0].remove()
    coords, vals = slice_coords(s_az.val, s_el.val, s_off.val)
    vmax = max(-vals.min(), 1.0) * 1.2
    fc = cmap(colors.Normalize(-vmax, vmax)(vals))
    surf[0] = ax.plot_surface(coords[..., 0], coords[..., 1], coords[..., 2],
                              facecolors=fc, shade=False, rstride=2, cstride=2, antialiased=False)
    fig.canvas.draw_idle()

ax_az = fig.add_axes([0.15, 0.10, 0.70, 0.03]); ax_el = fig.add_axes([0.15, 0.06, 0.70, 0.03])
ax_of = fig.add_axes([0.15, 0.02, 0.70, 0.03])
s_az = Slider(ax_az, "plane azim", -180, 180, valinit=paz0)
s_el = Slider(ax_el, "plane elev", -90, 90, valinit=pel0)
s_off = Slider(ax_of, "offset", -half, half, valinit=0.0)
for s in (s_az, s_el, s_off):
    s.on_changed(redraw)
redraw()

print("Rotate for the camera; use sliders for the SDF plane. Close the window when happy.")
plt.show()

# after close -> report
n = np.array([np.cos(np.radians(s_el.val)) * np.cos(np.radians(s_az.val)),
              np.cos(np.radians(s_el.val)) * np.sin(np.radians(s_az.val)),
              np.sin(np.radians(s_el.val))])
print("\n================ CHOSEN POSE ================")
print(f"CAMERA   elev={ax.elev:.1f}  azim={ax.azim:.1f}")
print(f"SLICE    plane_azim={s_az.val:.1f}  plane_elev={s_el.val:.1f}  offset={s_off.val:.1f}")
print(f"         plane normal = [{n[0]:.3f}, {n[1]:.3f}, {n[2]:.3f}]")
print("============================================")
