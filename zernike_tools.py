#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""zernike_tools.py – Optics helper library (Python/NumPy)
========================================================
Updated July 2025
-----------------


This file is a translation of several IDL utilities for
wave‑optics prototyping:

• **kzrt**          Create a centred polar grid.
• **ZZ / zsetup_py**Build individual or stacked Zernike polynomials.
• **sdft2d**        Scaled 2‑D discrete Fourier transform with
                    arbitrary output size and physical scaling.
• **sdft_na2**      Same idea, but axes mapped to numerical‑aperture angles.


"""

from __future__ import annotations

from math import factorial, pi
from typing import Dict, List, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt

Array  = np.ndarray
Shape2 = Tuple[int, int]
Vec2   = Tuple[float, float]

# ---------------------------------------------------------------------------
# 0. CACHE DICTS (mimic IDL COMMON blocks)
# ---------------------------------------------------------------------------
_zzz_cache:     Dict[Tuple[int, int, bool, bool], List[dict]]  = {}
_ZZ_cache:      Dict[Tuple[int, Shape2, bool, int], Array]     = {}
_sdft_cache:    Dict[Tuple[int, int, float, float, float, Vec2, bool], Tuple[Array, Array]] = {}
_sdftna_cache:  Dict[Tuple[int, int, float, float, float, bool], Tuple[Array, Array, Array]] = {}

# ---------------------------------------------------------------------------
# 1. POLAR GRID (kzrt)
# ---------------------------------------------------------------------------

def kzrt(N: int, rot1: int = 0, *, double: bool = False,
         nocrop: bool = False, annular: float | None = None) -> Tuple[Array, Array, Array]:
    """Create a centred `(r, θ)` grid covering −1 … +1.

    Parameters
    ----------
    N        : side length (outputs are N×N).
    rot1     : rotate θ‑origin by –π/2 if 1 (IDL compatibility).
    double   : return float64 instead of float32.
    nocrop   : if False, zero‑out `r` outside the unit circle.
    annular  : inner radius for an annular aperture (0–1).
    """
    dtype = np.float64 if double else np.float32
    coord = np.linspace(-(N - 1) / 2, (N - 1) / 2, N, dtype=dtype) * (2 / (N - 1))
    y = np.tile(coord, (N, 1))
    x = y.T

    r = np.sqrt(x**2 + y**2, dtype=dtype)
    if N % 2 == 0:  # even grid – centre lies between 4 pixels
        r /= r[:, 0].min()  # scale so corner radius = 1

    theta = np.arctan2(y, x, dtype=dtype)
    theta = np.where(theta < 0, theta + 2 * np.pi, theta) - rot1 * np.pi / 2

    mask = (r <= 1).astype(dtype)
    if annular is not None:
        mask *= r >= float(annular)
    if not nocrop:
        r *= mask
    return r, theta, mask

# ---------------------------------------------------------------------------
# 2. ZERNIKE POLYNOMIALS
# ---------------------------------------------------------------------------

def _radial(n: int, m: int, r: Array) -> Array:
    """Radial component Rₙᵐ(r) (Born & Wolf eq. 9‑12)."""
    m = abs(m)
    R = np.zeros_like(r, dtype=np.float64)
    for k in range((n - m) // 2 + 1):
        coeff = (-1)**k * factorial(n - k) / (
            factorial(k) * factorial((n + m)//2 - k) * factorial((n - m)//2 - k))
        R += coeff * r**(n - 2 * k)
    return R


def _nm_sequence(NZ: int, mahajan: bool) -> List[Tuple[int, int]]:
    seq: List[Tuple[int, int]] = []
    n = 0
    while len(seq) < NZ:
        m_vals = list(range(-n, n + 1, 2))
        if mahajan:
            seq.extend((n, m) for m in m_vals)
        else:
            if 0 in m_vals:
                seq.append((n, 0)); m_vals.remove(0)
            for m in m_vals:
                if m > 0:
                    seq.extend([(n, m), (n, -m)])
        n += 1
    return seq[:NZ]


def ZZ(*, idx: int, r: Array, theta: Array, mask: Array, mahajan: bool = False) -> Array:
    key = (idx, r.shape, mahajan, id(r))
    if key in _ZZ_cache:
        return _ZZ_cache[key]
    n, m = _nm_sequence(idx + 1, mahajan)[idx]
    Rnm  = _radial(n, m, r)
    Z = Rnm if m == 0 else Rnm * (np.cos(m * theta) if m > 0 else np.sin(-m * theta))
    Z *= mask
    _ZZ_cache[key] = Z
    return Z


def zsetup_py(N: int, *, NZ: int | None = None, double: bool = False, M: bool = False) -> List[dict]:
    if NZ is None:
        NZ = 37 + int(M)
    key = (N, NZ, double, M)
    if key in _zzz_cache:
        return _zzz_cache[key]
    r, theta, mask = kzrt(N, nocrop=True, double=double)
    dtype = np.float64 if double else np.float32
    stack = [{"a": ZZ(idx=i, r=r, theta=theta, mask=mask, mahajan=M).astype(dtype, copy=False)}
             for i in range(NZ)]
    _zzz_cache[key] = stack
    return stack

# ---------------------------------------------------------------------------
# 3. SCALED / PRUNED DFT (sdft2d)
# ---------------------------------------------------------------------------

def _sdft_phase_matrices(N: int, Nout: int, rmax: float, pmax: float, cc: float,
                          offset: Vec2, *, double: bool) -> Tuple[Array, Array]:
    key = (N, Nout, rmax, pmax, cc, offset, double)
    if key in _sdft_cache:
        return _sdft_cache[key]

    dtype  = np.float64 if double else np.float32
    cdtype = np.complex128 if double else np.complex64

    k_in  = (np.arange(N,    dtype=dtype) - (N-1)/2)   / ((N-1)/2)          # len N
    k_out = (np.arange(Nout, dtype=dtype) - (Nout-1)/2)/ ((Nout-1)/2)       # len Nout

    fx_out = k_out + offset[0] / pmax
    fy_out = k_out + offset[1] / pmax

    coef = 1j * cc * rmax * pmax * (Nout / float(N))
    trmx = np.exp(coef * np.outer(fx_out, k_in)).astype(cdtype, copy=False)  # (Nout,N)
    my   = np.exp(coef * np.outer(k_in,  fy_out)).astype(cdtype, copy=False) # (N,Nout)

    _sdft_cache[key] = (trmx, my)
    return trmx, my


def sdft2d(A: Union[Array, List[List[float]]], dir: int, Nout: int,
           rmax: float, pmax: float, cc: float,
           *, offset: Vec2 = (0.0, 0.0), double: bool = False, **_ignored) -> Array:
    """Scaled, pruned 2‑D DFT (forward if `dir ≥ 0`, inverse if `dir < 0`)."""
    A = np.asarray(A, dtype=(np.complex128 if double else np.complex64)
                        if np.iscomplexobj(A) else
                        (np.float64 if double else np.float32))
    assert A.ndim == 2 and A.shape[0] == A.shape[1], "A must be a square 2‑D array."
    N = A.shape[0]
    dir = 1 if dir >= 0 else -1

    trmx, my = _sdft_phase_matrices(N, Nout, rmax, pmax, cc, offset, double=double)

    # --- Runtime guard: rebuild if stale transposed matrices sneak in ---
    if trmx.shape != (Nout, N) or my.shape != (N, Nout):
        _sdft_cache.pop((N, Nout, rmax, pmax, cc, offset, double), None)
        trmx, my = _sdft_phase_matrices(N, Nout, rmax, pmax, cc, offset, double=double)

    out = trmx @ (A @ my)
    if dir < 0:
        out /= N**2
    return out

# ---------------------------------------------------------------------------
# 4. NA‑AWARE TRANSFORM (sdft_na2)
# ---------------------------------------------------------------------------

def _sdftna_matrices(N: int, Nout: int, rmax: float, NA: float, lam: float,
                     *, double: bool) -> Tuple[Array, Array, Array]:
    key = (N, Nout, rmax, NA, lam, double)
    if key in _sdftna_cache:
        return _sdftna_cache[key]

    dtype  = np.float64 if double else np.float32
    cdtype = np.complex128 if double else np.complex64

    k_in  = (np.arange(N,    dtype=dtype) - (N-1)/2)   / ((N-1)/2)
    k_out = (np.arange(Nout, dtype=dtype) - (Nout-1)/2)/ ((Nout-1)/2)

    coef = 2j * pi * rmax * NA / lam * (Nout / float(N))
    mx = np.exp(coef * np.outer(k_out, k_in)).astype(cdtype, copy=False)  # (Nout,N)
    my = np.exp(coef * np.outer(k_in,  k_out)).astype(cdtype, copy=False) # (N,Nout)

    theta = np.arcsin(np.clip(NA * k_out, -1, 1))  # angle per output column
    _sdftna_cache[key] = (mx, my, theta)
    return mx, my, theta


def sdft_na2(A: Union[Array, List[List[float]]], dir: int, Nout: int,
             rmax: float, NA: float, lam: float,
             *, return_theta: bool = False, double: bool = False) -> Union[Array, Tuple[Array, Array]]:
    A = np.asarray(A, dtype=(np.complex128 if double else np.complex64)
                        if np.iscomplexobj(A) else
                        (np.float64 if double else np.float32))
    assert A.ndim == 2 and A.shape[0] == A.shape[1], "A must be a square 2‑D array."
    N = A.shape[0]
    dir = 1 if dir >= 0 else -1

    mx, my, theta = _sdftna_matrices(N, Nout, rmax, NA, lam, double=double)

    if mx.shape != (Nout, N):
        _sdftna_cache.pop((N, Nout, rmax, NA, lam, double), None)
        mx, my, theta = _sdftna_matrices(N, Nout, rmax, NA, lam, double=double)

    out = mx @ (A @ my)
    if dir < 0:
        out /= N**2
    return (out, theta) if return_theta else out

# ---------------------------------------------------------------------------
# 5. DEMOS & CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "zernike_demo"

    if cmd == "sdft_demo":
        img = np.random.rand(64, 64)
        F   = sdft2d(img, +1, 128, 1, 1, 1)
        print("sdft2d output", F.shape)

    elif cmd == "sdftna_demo":
        img = np.random.rand(64, 64)
        F, th = sdft_na2(img, +1, 128, 1, 0.25, 1, return_theta=True)
        print("sdft_na2 output", F.shape, "theta length", len(th))

    else:  # default: quick Mahajan Zernike gallery
        zzz = zsetup_py(512, NZ=6, double=True, M=True)
        fig, axs = plt.subplots(2, 3, figsize=(8, 5))
        for k, ax in enumerate(axs.flat):
            im = ax.imshow(zzz[k]["a"], cmap="RdBu", origin="lower")
            ax.axis("off"); ax.set_title(f"Mode {k}")
        plt.tight_layout(); plt.show()