#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_zone_plate_focus.py – Demo of on-axis beam focusing using SDFT,
with adjustable physical focal lengths (primary and additional).
Linear-scale intensity view, zoomed around the focal spot.
"""

import numpy as np
import matplotlib.pyplot as plt
from zernike_tools import sdft2d

# ---------------------------
# Physical parameters
# ---------------------------
wavelength = 500e-9         # Wavelength (m)
ap_radius = 0.5e-3            # Aperture radius (m)

# Focal-length parameters (edit these):
f_main = 0.15                # Primary focal length (m)
f_add = 0.2              # Additional curvature focal length (m)

# Simulation grids
N = 256                     # Aperture sampling points
Nout = 512                 # Focal-plane sampling points

# Derived quantities
dx = 10 * ap_radius / (N - 1)        # Aperture-plane pixel size (m)
rmax = ap_radius                    # Aperture half-width (m)
pmax = wavelength * f_main / dx     # Focal-plane half-width (m) based on primary focus
cc = 2 * np.pi / wavelength         # Propagation constant (1/m)

# ---------------------------
# Build aperture field
# ---------------------------
x = np.linspace(-ap_radius, ap_radius, N)
X, Y = np.meshgrid(x, x)
mask = (X**2 + Y**2 <= ap_radius**2).astype(np.float32)

# ---------------------------
# Apply lens phase
# ---------------------------
# Quadratic phase phi = (k/(2f)) * r^2
phi_main = (cc / (2 * f_main)) * (X**2 + Y**2)
phi_add  = (cc / (2 * f_add))  * (X**2 + Y**2)

# Field at aperture: converging lens phase exp(+i*phi)
E_main    = mask * np.exp(1j * phi_main)
E_defocus = mask * np.exp(1j * (phi_main + phi_add))

# ---------------------------
# Propagate to focal plane via SDFT (inverse transform)
# ---------------------------
# Use inverse (dir=-1) for Fresnel propagation
F_main    = sdft2d(E_main,    dir=-1, Nout=Nout, rmax=rmax, pmax=pmax, cc=cc)
F_defocus = sdft2d(E_defocus, dir=-1, Nout=Nout, rmax=rmax, pmax=pmax, cc=cc)

# Compute intensities
I_main    = np.abs(F_main)**2
I_defocus = np.abs(F_defocus)**2

# ---------------------------
# Zoom around focal spot
# ---------------------------
c = Nout // 2
zoom_px = 50
I0 = I_main[c-zoom_px:c+zoom_px, c-zoom_px:c+zoom_px]
I1 = I_defocus[c-zoom_px:c+zoom_px, c-zoom_px:c+zoom_px]

# Coordinates in focal plane (m)
x_out = np.linspace(-pmax, pmax, Nout)
x_zoom = x_out[c-zoom_px:c+zoom_px]
extent = (x_zoom[0]*1e3, x_zoom[-1]*1e3, x_zoom[0]*1e3, x_zoom[-1]*1e3)  # in mm

# ---------------------------
# Plot results
# ---------------------------
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Main focus
im0 = axs[0].imshow(I0, origin='lower', extent=extent, cmap='viridis')
axs[0].set_title(f'Focused: f = {f_main*1e3:.1f} mm')
axs[0].set_xlabel('x (mm)')
axs[0].set_ylabel('y (mm)')
fig.colorbar(im0, ax=axs[0], label='Intensity (a.u.)')

# Defocused
im1 = axs[1].imshow(I1, origin='lower', extent=extent, cmap='viridis')
axs[1].set_title(f'Defocused: f_add = {f_add*1e3:.1f} mm')
axs[1].set_xlabel('x (mm)')
axs[1].set_ylabel('')
fig.colorbar(im1, ax=axs[1], label='Intensity (a.u.)')

plt.tight_layout()
plt.show()