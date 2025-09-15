#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
triple_focus_zone_plate_cached_dynamic_zoom_optimized.py

Compute a 330 µm Fresnel zone plate mask that produces three foci in an
equilateral triangle. This optimized script features:
  • Parameter-aware caching with progress messages
  • FFT-based Fresnel propagation for 10× speedup on full-field propagation
  • Separate design vs. simulation propagation distances
  • Phase-zoom computed at the design focal length via inverse SDFT
  • Focal-zoom panels propagated to an arbitrary simulation plane
  • Optional additional curvature term for out-of-focus simulation
  • Runtime timer to display total execution time

Author: Bernhard Luttgenau (optimized)
"""

import os
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from zernike_tools import sdft2d

# ----------------------- Helper Functions -----------------------

def fresnel_propagate_fft(A, dx, z, wavelength):
    """
    Fast Fresnel propagation using FFT convolution.
    A: complex field at aperture plane
    dx: spatial sampling pitch [m]
    z: propagation distance [m]
    wavelength: wavelength [m]
    """
    N = A.shape[0]
    fx = np.fft.fftfreq(N, d=dx)
    FX, FY = np.meshgrid(fx, fx)
    H = np.exp(-1j * np.pi * wavelength * z * (FX**2 + FY**2))
    U1 = np.fft.fft2(A)
    U2 = U1 * H
    U = np.fft.ifft2(U2)
    return U

# Start runtime timer
t0 = time.time()

# ----------------------- User options -----------------------
diameter            = 330e-6
resolution_ap       = 8192
resolution_fp       = 512
zoom_resolution     = 4096
dpi_fig             = 500

focal_length        = 17.5e-3
z_distance          = 17.5e-3

defocus_delta        = 0e-3
defocus_focal_length = None

phase_zoom_size     = 10e-6
phase_zoom_cx       = -50e-6
phase_zoom_cy       = 100e-6

wavelength          = 1.75e-9
target_fwhm         = 175e-9
rng_radius          = 217e-6      # desired ring radius in focal plane
helicities          = [0, 1, -1]

spot_zoom_size      = 2e-5

# ---------------- Parameter-aware caching ----------------
cache_dir = "cache"
os.makedirs(cache_dir, exist_ok=True)

def cache_filename(prefix, params):
    parts = [f"{k}{v}" for k, v in params.items()]
    return os.path.join(cache_dir, f"{prefix}_" + "_".join(parts) + ".npy")

cache_A_params = {
    f"f{focal_length*1e3:.1f}mm": "",
    f"resAp{resolution_ap}": "",
    f"wl{wavelength*1e9:.2f}nm": "",
}
cache_U_params = {
    f"z{z_distance*1e3:.1f}mm": "",
    f"resFp{resolution_fp}": "",
}
if defocus_delta != 0:
    defocus_focal_length = -focal_length * (focal_length + defocus_delta) / defocus_delta
    cache_U_params[f"defl{defocus_focal_length*1e3:.1f}mm"] = ""

cache_Afile = cache_filename("A_complex", cache_A_params)
cache_Ufile = cache_filename("U_fp_full", cache_U_params)

# ---------------- Derived grid & constants ----------------
radius_ap = diameter / 2
dx_ap = diameter / resolution_ap
coords_ap = (np.arange(resolution_ap) - resolution_ap//2) * dx_ap
X_ap, Y_ap = np.meshgrid(coords_ap, coords_ap)
ap_mask = (X_ap**2 + Y_ap**2) <= radius_ap**2

# ----- analytic build parameters (used below and for zoom) -----
k       = 2 * np.pi / wavelength
theta   = 10.8e-3                      # tilt angle [rad]
f_theta = focal_length * theta        # nominal ring from tilt
r_sub   = f_theta - rng_radius        # sub-plate center offset in aperture
angles  = np.deg2rad([0, 120, 240])   # three directions

# ---------------- Build or load aperture field ----------------
if os.path.exists(cache_Afile):
    print(f"[Cache] Loading aperture field from {cache_Afile}")
    A_complex = np.load(cache_Afile)
else:
    print(f"[Cache] Computing aperture field and saving to {cache_Afile}")
    A_complex = np.zeros_like(X_ap, dtype=np.complex128)

    for phi_dir, ℓ in zip(angles, helicities):
        # 1) tilt phase to steer focus by θ in direction phi_dir
        θx, θy    = theta*np.cos(phi_dir), theta*np.sin(phi_dir)
        tilt_phase = np.exp(-1j * k * (θx*X_ap + θy*Y_ap))

        # 2) shift local lens center by r_sub along phi_dir
        x_off, y_off = r_sub*np.cos(phi_dir), r_sub*np.sin(phi_dir)
        Xs, Ys       = X_ap - x_off, Y_ap - y_off

        # 3) quadratic lens phase about the shifted center
        lens_phase   = np.exp(-1j * k/(2*focal_length) * (Xs**2 + Ys**2))

        # 4) vortex term of charge ℓ about the shifted center
        vortex_phase = np.exp(1j * ℓ * np.arctan2(Ys, Xs))

        Ai = lens_phase * tilt_phase * vortex_phase
        Ai *= ap_mask
        Ai /= np.max(np.abs(Ai))
        A_complex += Ai

    A_complex = A_complex.astype(np.complex64)
    np.save(cache_Afile, A_complex)

# ---------------- Optional defocus ----------------
if defocus_focal_length:
    phi_defocus = np.exp(
        -1j * 2*np.pi / wavelength *
        (X_ap**2 + Y_ap**2) / (2 * defocus_focal_length)
    )
    A_complex_defocused = A_complex * phi_defocus
else:
    A_complex_defocused = A_complex

# ---------------- Full-field Fresnel propagation (optimized) ----------------
if os.path.exists(cache_Ufile):
    print(f"[Cache] Loading propagated field from {cache_Ufile}")
    U_fp_full = np.load(cache_Ufile)
else:
    print("[Cache] Performing FFT-based Fresnel propagation for full field (10× faster)")
    U_fp_full = fresnel_propagate_fft(
        A_complex_defocused, dx_ap, z_distance, wavelength
    )
    np.save(cache_Ufile, U_fp_full)
I_fp_full = np.abs(U_fp_full)**2

# ---------------- Prepare for plotting ----------------
# define true focal-plane ring positions
x_f = rng_radius * np.cos(angles)
y_f = rng_radius * np.sin(angles)
x_z = x_f * (z_distance / focal_length)
y_z = y_f * (z_distance / focal_length)

fig, axs = plt.subplots(3, 3, figsize=(15, 10), dpi=dpi_fig)
ext_ap_um = radius_ap * 1e6

# ---- Row 1: Full-aperture plots (unchanged) ----
phase_ap = np.mod(np.angle(A_complex), 2*np.pi)
phase_ap = np.where(ap_mask, phase_ap, np.nan)
im1 = axs[0,0].imshow(phase_ap, origin='lower',
    extent=[-ext_ap_um, ext_ap_um, -ext_ap_um, ext_ap_um], cmap='hsv')
axs[0,0].set(title='Full Zone Plate \n Full Phase', xlabel='x [µm]', ylabel='y [µm]')
plt.colorbar(im1, ax=axs[0,0])

phase_bin = np.where(phase_ap < np.pi*3/4, 0, 1)
phase_bin = np.where(np.isfinite(phase_ap), phase_bin, np.nan)
im2 = axs[0,1].imshow(phase_bin, origin='lower',
    extent=[-ext_ap_um, ext_ap_um, -ext_ap_um, ext_ap_um], cmap='gray')
axs[0,1].set(title='Full Zone Plate \n Binary Phase', xlabel='x [µm]', ylabel='y [µm]')
plt.colorbar(im2, ax=axs[0,1])

amp_ap = np.abs(A_complex)
amp_ap = np.where(ap_mask, amp_ap, np.nan)
im3 = axs[0,2].imshow(amp_ap, origin='lower',
    extent=[-ext_ap_um, ext_ap_um, -ext_ap_um, ext_ap_um], cmap='inferno')
axs[0,2].set(title='Full Zone Plate \n Electric Field Amplitude', xlabel='x [µm]', ylabel='y [µm]')
plt.colorbar(im3, ax=axs[0,2])

# ---- Row 2: True analytic zoom of aperture-plane patch ----
half_az = phase_zoom_size / 2
dx_zoom = phase_zoom_size / zoom_resolution
coords_zoom = (np.arange(zoom_resolution) - zoom_resolution//2) * dx_zoom
X_zoom, Y_zoom = np.meshgrid(coords_zoom + phase_zoom_cx,
                             coords_zoom + phase_zoom_cy)

# (1) apply **global aperture mask**, not a patch-sized circle
mask_zoom = (X_zoom**2 + Y_zoom**2) <= radius_ap**2

A_zoom = np.zeros_like(X_zoom, dtype=complex)
for phi_dir, ℓ in zip(angles, helicities):
    θx, θy = theta*np.cos(phi_dir), theta*np.sin(phi_dir)
    tilt_loc = np.exp(-1j * k * (θx*X_zoom + θy*Y_zoom))

    Xs_loc = X_zoom - r_sub*np.cos(phi_dir)
    Ys_loc = Y_zoom - r_sub*np.sin(phi_dir)
    lens_loc = np.exp(-1j * k/(2*focal_length) * (Xs_loc**2 + Ys_loc**2))
    vort_loc = np.exp(1j * ℓ * np.arctan2(Ys_loc, Xs_loc))

    A_zoom += lens_loc * tilt_loc * vort_loc

A_zoom *= mask_zoom  # now zeros only where outside the real aperture

# a) continuously wrapped phase
phase_zoom = np.mod(np.angle(A_zoom), 2*np.pi)
im4 = axs[1,0].imshow(phase_zoom, origin='lower',
    extent=[(phase_zoom_cx-half_az)*1e6, (phase_zoom_cx+half_az)*1e6,
            (phase_zoom_cy-half_az)*1e6, (phase_zoom_cy+half_az)*1e6],
    cmap='hsv')
axs[1,0].set(title='Zone Plate Zoom-in \n Full Phase', xlabel='x [µm]', ylabel='y [µm]')
plt.colorbar(im4, ax=axs[1,0])

# b) binary phase
phase_zoom_bin = np.where(phase_zoom < np.pi*3/4, 0, 1)
im5 = axs[1,1].imshow(phase_zoom_bin, origin='lower',
    extent=[(phase_zoom_cx-half_az)*1e6, (phase_zoom_cx+half_az)*1e6,
            (phase_zoom_cy-half_az)*1e6, (phase_zoom_cy+half_az)*1e6],
    cmap='gray')
axs[1,1].set(title='Zone Plate Zoom-in \n Binary Phase', xlabel='x [µm]', ylabel='y [µm]')
plt.colorbar(im5, ax=axs[1,1])

# c) amplitude
amp_zoom = np.abs(A_zoom)
im6 = axs[1,2].imshow(amp_zoom, origin='lower',
    extent=[(phase_zoom_cx-half_az)*1e6, (phase_zoom_cx+half_az)*1e6,
            (phase_zoom_cy-half_az)*1e6, (phase_zoom_cy+half_az)*1e6],
    cmap='inferno')
axs[1,2].set(title='Zone Plate Zoom-in \n Electric Field Amplitude', xlabel='x [µm]', ylabel='y [µm]')
plt.colorbar(im6, ax=axs[1,2])

# ---- Row 3: Focus zoom via forward SDFT (unchanged) ----
half_s = spot_zoom_size / 2
for i, (xi, yi) in enumerate(zip(x_z, y_z)):
    ax = axs[2,i]
    tilt = np.exp(-1j * 2*np.pi/(wavelength*z_distance) * (xi*X_ap + yi*Y_ap))
    Uz = sdft2d(A_complex_defocused * tilt, dir=+1,
                Nout=resolution_fp, rmax=radius_ap,
                pmax=half_s, cc=2*np.pi/(wavelength*z_distance))
    Iz = np.abs(Uz)**2
    he = half_s * 1e6
    
    # --- FWHM measurement snippet ---
    # assume Uz and Iz are defined, zoom_resolution and spot_zoom_size are set

    # physical x‐coordinates for the zoom patch
    x = np.linspace(-half_s, half_s, resolution_fp) * 1e6  # in µm

    # take a 1D cut through the center row of Iz
    center_idx = resolution_fp // 2
    I1d = Iz[center_idx, :]

    # find half‐max crossings
    half = I1d.max() / 2
    # indices where signal ≥ half
    inds = np.where(I1d >= half)[0]
    if len(inds) > 1:
        left, right = inds[0], inds[-1]
        fwhm_um = x[right] - x[left]
        print(f"Measured FWHM: {fwhm_um:.2f} µm")
    else:
        print("FWHM measurement failed (no half-max crossing).")
            # --- end snippet ---
    imz = ax.imshow(Iz, origin='lower',
        extent=[xi*1e6-he, xi*1e6+he, yi*1e6-he, yi*1e6+he])
    ax.set(title=f'Focus {i+1} Zoom', xlabel='x [µm]', ylabel='y [µm]')
    plt.colorbar(imz, ax=ax)

plt.tight_layout()
timestamp = time.strftime("%m-%d-%Y_%H-%M-%S", time.localtime())

plt.savefig(f"/Users/BJLuttgenau/Documents/ALS_Research Scientist/Simulation/IDL code/images/threefoci_oam_overview_dynamic_optimized_{timestamp}.png", dpi=dpi_fig,
            bbox_inches="tight", pad_inches=0.05)
plt.show()


# =================== ZP → detector propagation with scaled Fresnel (filled, linear, memory-safe) ===================

import numpy as np
import matplotlib.pyplot as plt

# ---------------- Controls ----------------
# Use a lower propagation grid to keep RAM in check. Must divide resolution_ap.
prop_res = 4096 if (resolution_ap % 4096 == 0) else 2048   # fallback if 4096 not divisible
panel_res = 1024    # display cap per side (downsample only for plotting)
dpi_panels = 250
z_panels_mm = np.array([1, 5, 10, 17.5, 25, 50, 75, 100, 300], dtype=float)
z_panels = z_panels_mm * 1e-3  # meters

# -------------- Prep: optionally downsample aperture field for propagation --------------
ds = resolution_ap // prop_res
if ds < 1:
    ds = 1
A_prop = A_complex_defocused[::ds, ::ds].astype(np.complex64, copy=False)
dx_prop = dx_ap * ds                              # input pixel size for propagation [m]
Nprop   = A_prop.shape[0]
L_in    = Nprop * dx_prop                         # physical side length at input [m]; equals your diameter-grid span

# -------------- Scaled Fresnel (chirp-FFT-chirp) --------------
def fresnel_scaled(A, dx, z, wavelength):
    """
    Unitary scaled Fresnel (chirp-FFT-chirp).
    Returns:
        Uz      : complex field at distance z [same N×N as A]
        dx_out  : output sampling [m]
    Notes:
        dx_out = λ|z| / (N * dx); FOV_out = N * dx_out
    """
    A = A.astype(np.complex64, copy=False)
    N = A.shape[0]
    if np.isclose(z, 0.0):
        return A.copy(), dx

    k = 2*np.pi / wavelength

    # Coordinates (float64 for phase accuracy; fields remain complex64)
    x = (np.arange(N, dtype=np.float64) - N//2) * dx
    X, Y = np.meshgrid(x, x, indexing='xy')

    a = np.pi / (wavelength * z)   # chirp coefficient

    # Pre-chirp
    quad_in = np.exp(1j * a * (X**2 + Y**2)).astype(np.complex64)
    U_in = (A * quad_in).astype(np.complex64)

    # Unitary FFT
    U_F = (1/np.sqrt(N*N)) * np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(U_in)))

    # Output sampling and coords
    dx_out = wavelength * abs(z) / (N * dx)
    x_out = (np.arange(N, dtype=np.float64) - N//2) * dx_out
    Xo, Yo = np.meshgrid(x_out, x_out, indexing='xy')

    # Post-chirp + propagation phase
    quad_out = np.exp(1j * a * (Xo**2 + Yo**2)).astype(np.complex64)
    Uz = (np.exp(1j*k*z) * quad_out * U_F).astype(np.complex64)

    return Uz, float(dx_out)

# -------------- Helper: display decimator --------------
def decimate_to(A, target_side):
    """Downsample A by integer stride so side ≤ target_side (for plotting only)."""
    n = A.shape[0]
    if n <= target_side:
        return A
    s = int(np.ceil(n / target_side))
    return A[::s, ::s]

# -------------- Ring centers (divergence geometry) --------------
def ring_centers(z):
    Rz = rng_radius * (z / focal_length)  # meters
    return [
        (Rz*np.cos(angles[0]), Rz*np.sin(angles[0])),
        (Rz*np.cos(angles[1]), Rz*np.sin(angles[1])),
        (Rz*np.cos(angles[2]), Rz*np.sin(angles[2])),
    ], Rz

# -------------- Make the 3×3 detector panels --------------
fig, axs = plt.subplots(3, 3, figsize=(13, 13), dpi=dpi_panels)
axs = axs.ravel()

for i, z in enumerate(z_panels):
    # Physics: direct ZP → plane z, with scaled Fresnel (this captures focusing at f and defocus beyond f)
    Uz, dx_out = fresnel_scaled(A_prop, dx_prop, z, wavelength)
    I = np.abs(Uz)**2
    I = I.astype(np.float32, copy=False)

    # Plot on linear scale; cap vmax by percentile to keep bright cores from washing out blur
    I_disp = decimate_to(I, panel_res)
    if I_disp.max() > 0:
        vmax = float(np.percentile(I_disp, 99.9))
        if vmax <= 0:
            vmax = float(I_disp.max())
    else:
        vmax = 1.0

    # Physical extent (fully filled): the scaled Fresnel gives us the detector-plane pixel size
    Ndisp = I_disp.shape[0]
    L_out = Ndisp * dx_out  # meters
    extent_mm = [-L_out/2*1e3, L_out/2*1e3, -L_out/2*1e3, L_out/2*1e3]

    ax = axs[i]
    im = ax.imshow(I_disp, origin='lower', extent=extent_mm, cmap='inferno',
                   vmin=0.0, vmax=vmax, interpolation='nearest')
    ax.set_aspect('equal')
    ax.set_title(f"Detector @ {z_panels_mm[i]:.1f} mm")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")

    # Optional: annotate expected ring radius at this z to confirm divergence geometry
    (centers, Rz) = ring_centers(z)
    ax.add_patch(plt.Circle((0,0), Rz*1e3, fill=False, lw=0.8, ls='--', color='w', alpha=0.6))

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Intensity [a.u.] (linear)")

plt.tight_layout()
timestamp = time.strftime("%m-%d-%Y_%H-%M-%S", time.localtime())
outpath = f"/Users/BJLuttgenau/Documents/ALS_Research Scientist/Simulation/IDL code/images/threefoci_scaledFresnel_linear_{timestamp}.png"
plt.savefig(outpath, dpi=dpi_panels, bbox_inches="tight", pad_inches=0.05)
plt.show()
print(f"[Scaled-Fresnel panels saved] {outpath}")

# -------------- Quick sanity print: expected post-focus blur scale --------------
# Treat focused spot as a Gaussian waist near z=f with w0 from your target FWHM.
w0 = (target_fwhm / np.sqrt(2*np.log(2)))  # 1/e^2 radius at focus [m]
zR = np.pi * w0**2 / wavelength            # Rayleigh range [m]
for z in [0.025, 0.050, 0.100, 0.300]:     # meters
    w = w0 * np.sqrt(1.0 + ((z - focal_length)/zR)**2)
    print(f"z={z*1e3:6.1f} mm: expected 1/e^2 radius ≈ {w*1e3:6.2f} mm (FWHM ≈ {w*np.sqrt(2*np.log(2))*1e3:6.2f} mm)")


# Display runtime
run_time = time.time() - t0
print(f"Total runtime: {run_time:.2f} seconds")
