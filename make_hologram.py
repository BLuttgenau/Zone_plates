#!/usr/bin/env python3
"""
make_hologram.py – Three‑focus / multi‑OAM hologram demo (clean layout)
========================================================================
This revision fixes the **huge white‑space / squished‑plots** problem by:

* Dropping *constrained_layout* and using `plt.tight_layout()` instead.
* Placing **compact horizontal colour‑bars** *inside* each panel via
  `inset_axes`, so they no longer push the axes sideways.
* Keeping physical scale info with axis labels + scale bars (the colour‑bar
  itself no longer needs to reserve margin space).

Run the script – all four images should now be evenly sized in a neat 2 × 2
grid.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.font_manager as fm
import importlib

# -----------------------------------------------------------------------------
# 0.  Physical & design parameters  –– edit here
# -----------------------------------------------------------------------------
LAM     = 1.75e-9          # wavelength [m]
F       = 17.5e-3        # propagation distance / focal length [m]
D_ZP    = 330e-6         # zone‑plate / aperture diameter [m]

IMAGE_SIZE = 330e-6       # design‑plane side length [m]
N_IN       = 5096        # design grid (px)
N_OUT      = 10192        # detector grid (px)

FOCI       = [(-2e-6, 0.0), (0.0, 0.0), (2e-6, 0.0)]   # detector offsets [m]
L_CHARGE   = [ +1, 0, -1 ]                              # OAM per focus
REL_AMP    = [  1, 1,  1 ]                              # relative amplitude

# Scale‑bar lengths
SB_DES_LEN = 50e-6   # pupil plane
SB_DET_LEN = 5e-6    # detector plane

VERBOSE    = False   # set True to see min/max diagnostics
SAVE_FIG   = False

# -----------------------------------------------------------------------------
# 1.  Derived parameters
# -----------------------------------------------------------------------------
PITCH      = IMAGE_SIZE / N_IN
DET_PITCH  = LAM * F / IMAGE_SIZE
R_MAX      = IMAGE_SIZE / 2
P_MAX      = IMAGE_SIZE / 2 / (LAM * F)
CC         = 2 * np.pi / (LAM * F)

# -----------------------------------------------------------------------------
# 2.  Reload patched zernike_tools and clear cache
# -----------------------------------------------------------------------------
import zernike_tools as zt
importlib.reload(zt)
zt._sdft_cache.clear()

# -----------------------------------------------------------------------------
# 3.  Build design‑plane field
# -----------------------------------------------------------------------------
coords = (np.arange(N_IN) - (N_IN-1)/2) * PITCH
X, Y   = np.meshgrid(coords, coords, indexing="xy")
R      = np.hypot(X, Y)
THETA  = np.arctan2(Y, X)
MASK   = (R <= D_ZP/2).astype(float)

fx = np.array([x0 / (LAM * F) for x0, _ in FOCI])
fy = np.array([y0 / (LAM * F) for _, y0 in FOCI])

field_sum = np.zeros_like(X, dtype=np.complex128)
for fx_i, fy_i, l_i, a_i in zip(fx, fy, L_CHARGE, REL_AMP):
    phase = 2*np.pi*(fx_i*X + fy_i*Y) + l_i*THETA
    field_sum += a_i * np.exp(1j*phase)

PHASE  = np.angle(field_sum)
PUPIL  = MASK * np.exp(1j*PHASE)

# -----------------------------------------------------------------------------
# 4. Forward Fresnel propagation
# -----------------------------------------------------------------------------
U   = zt.sdft2d(PUPIL, +1, N_OUT, R_MAX, P_MAX, CC, double=True)
MAG = np.abs(U)
MAX_MAG = MAG.max()
AMP_NORM = MAG / MAX_MAG if MAX_MAG > 0 else np.zeros_like(MAG)

# Zoom box around brightest spot
peak_y, peak_x = np.unravel_index(np.argmax(MAG), MAG.shape)
BOX = 300
ys, ye = peak_y-BOX, peak_y+BOX
xs, xe = peak_x-BOX, peak_x+BOX
CROP   = MAG[ys:ye, xs:xe]

# -----------------------------------------------------------------------------
# 5.  Plotting
# -----------------------------------------------------------------------------
fontprops = fm.FontProperties(size=8)
fig, axs = plt.subplots(2, 2, figsize=(8, 8))

# -- helper to add horizontal colour‑bar inside panel -------------------------
def add_h_cbar(ax, im):
    cax = inset_axes(ax, width="60%", height="4%", loc="lower left",
                     bbox_to_anchor=(0.02, -0.12, 1, 1), bbox_transform=ax.transAxes,
                     borderpad=0)
    plt.colorbar(im, cax=cax, orientation="horizontal")

# (a) pupil phase -------------------------------------------------------------
extent_design = np.array([-IMAGE_SIZE/2, IMAGE_SIZE/2, -IMAGE_SIZE/2, IMAGE_SIZE/2])*1e6
im0 = axs[0,0].imshow(PHASE, cmap="twilight", vmin=-np.pi, vmax=np.pi, extent=extent_design)
axs[0,0].set_title("Pupil phase (rad)")
axs[0,0].set_xlabel("x [µm]"); axs[0,0].set_ylabel("y [µm]")
add_h_cbar(axs[0,0], im0)

sb0 = AnchoredSizeBar(axs[0,0].transData, SB_DES_LEN*1e6, "50 µm", 'lower right',
                      pad=0.35, color='white', frameon=False,
                      size_vertical=SB_DES_LEN*1e6*0.05, fontproperties=fontprops)
axs[0,0].add_artist(sb0)

# (b) detector amplitude ------------------------------------------------------
extent_det = (np.array([-N_OUT/2, N_OUT/2, -N_OUT/2, N_OUT/2]) * DET_PITCH * 1e6)
im1 = axs[0,1].imshow(MAG, cmap="inferno", norm=LogNorm(vmin=MAX_MAG*1e-6, vmax=MAX_MAG), extent=extent_det)
axs[0,1].set_title("|U| (log)"); axs[0,1].set_xlabel("x [µm]"); axs[0,1].set_ylabel("y [µm]")
add_h_cbar(axs[0,1], im1)

sb1 = AnchoredSizeBar(axs[0,1].transData, SB_DET_LEN*1e6, "5 µm", 'lower right',
                      pad=0.35, color='white', frameon=False,
                      size_vertical=SB_DET_LEN*1e6*0.05, fontproperties=fontprops)
axs[0,1].add_artist(sb1)

# (c) zoom --------------------------------------------------------------------
extent_crop = np.array([xs, xe, ys, ye])*DET_PITCH*1e6
im2 = axs[1,0].imshow(CROP, cmap="inferno", norm=LogNorm(vmin=CROP.max()*1e-6, vmax=CROP.max()), extent=extent_crop)
axs[1,0].set_title("|U| zoom"); axs[1,0].set_xlabel("x [µm]"); axs[1,0].set_ylabel("y [µm]")
add_h_cbar(axs[1,0], im2)

sb2 = AnchoredSizeBar(axs[1,0].transData, SB_DET_LEN*1e6, "5 µm", 'lower right',
                      pad=0.35, color='white', frameon=False,
                      size_vertical=SB_DET_LEN*1e6*0.05, fontproperties=fontprops)
axs[1,0].add_artist(sb2)

# (d) phase at detector -------------------------------------------------------
extent_phase = extent_det
im3 = axs[1,1].imshow(np.angle(U), cmap="twilight", vmin=-np.pi, vmax=np.pi, extent=extent_phase)
axs[1,1].set_title("Phase (rad)"); axs[1,1].set_xlabel("x [µm]"); axs[1,1].set_ylabel("y [µm]")
add_h_cbar(axs[1,1], im3)

sb3 = AnchoredSizeBar(axs[1,1].transData, SB_DET_LEN*1e6, "5 µm", 'lower right',
                      pad=0.35, color='white', frameon=False,
                      size_vertical=SB_DET_LEN*1e6*0.05, fontproperties=fontprops)
axs[1,1].add_artist(sb3)

plt.tight_layout()

if SAVE_FIG:
    fig.savefig("hologram_results.png", dpi=300, bbox_inches="tight", pad_inches=0.05)

plt.show()
