"""
Gaussian beam with a half-plane π phase step:
- Build a Gaussian field at the lens (pupil).
- Impose a π phase step for x < 0.
- Compute the Fraunhofer (focal-plane) field via FFT.
- Visualize focal intensity with and without the phase step, plus a line cut.

Tweak: lam, f, wL, N, L.
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------- Parameters ----------
lam = 632.8e-9          # wavelength [m]
f   = 150e-3            # focal length [m]
wL  = 1.5e-3            # 1/e field radius at the lens [m]
N   = 1024              # grid samples (power of 2 is convenient)
L   = 4e-2              # total span at the lens plane [m] (choose >> wL to avoid clipping)

# ---------- Coordinates ----------
dx = L / N
x = (np.arange(N) - N//2) * dx
X, Y = np.meshgrid(x, x, indexing="xy")

# Spatial-frequency grid
fx = (np.arange(N) - N//2) / L       # cycles/m
FX, FY = np.meshgrid(fx, fx, indexing="xy")

# Focal-plane coordinates (Fourier scaling)
xf = lam * f * FX   # [m]
yf = lam * f * FY   # [m]

# ---------- Pupil fields ----------
# Gaussian field amplitude (real, positive)
E_pupil = np.exp(-(X**2 + Y**2) / wL**2)

# Half-plane π phase step across x = 0
mask_pi = np.where(X >= 0, 1.0, -1.0)   # multiply field by ±1 (π phase flip)
E_pupil_pi = E_pupil * mask_pi

# Soft edge taper to reduce FFT wrap-around (optional but helpful)
edge_taper = np.exp(-((X**2 + Y**2)/(0.45*L)**2)**8)
E_pupil    = E_pupil    * edge_taper
E_pupil_pi = E_pupil_pi * edge_taper

# ---------- Fraunhofer propagation (field at focus) ----------
def focal_field(Ep):
    # Fraunhofer: field ∝ Fourier{ pupil field }
    Ef = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(Ep)))
    # Scale factor: absolute scale not critical for visualization
    Ef *= (dx * dx) / (1j * lam * f)
    return Ef

Ef0  = focal_field(E_pupil)
Efpi = focal_field(E_pupil_pi)

I0  = np.abs(Ef0)**2
Ipi = np.abs(Efpi)**2

# Normalize for plotting
I0  /= I0.max()
Ipi /= Ipi.max()

# ---------- Plots ----------
# (1) Pupil amplitude and phase mask
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(E_pupil, extent=[x[0]*1e3, x[-1]*1e3, x[0]*1e3, x[-1]*1e3],
           origin='lower', aspect='equal')
plt.xlabel('x at lens [mm]'); plt.ylabel('y at lens [mm]')
plt.title('Gaussian field amplitude (pupil)')
plt.colorbar(label='|E| (arb)')

plt.subplot(1, 2, 2)
plt.imshow(mask_pi, extent=[x[0]*1e3, x[-1]*1e3, x[0]*1e3, x[-1]*1e3],
           origin='lower', aspect='equal')
plt.xlabel('x at lens [mm]'); plt.ylabel('y at lens [mm]')
plt.title('Half-plane π phase step')
plt.colorbar(label='phase multiplier')
plt.tight_layout()
plt.show()

# (2) Focal-plane intensity: no mask vs π-step mask
# Show a central crop for readability
crop = 256
ix = slice(N//2 - crop, N//2 + crop)
iy = ix
xfx = xf[0, ix]      # x' axis for crop
yfy = yf[iy, 0]      # y' axis for crop

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(I0[iy, ix], extent=[xfx[0]*1e6, xfx[-1]*1e6, yfy[0]*1e6, yfy[-1]*1e6],
           origin='lower', aspect='equal')
plt.xlabel("x' at focus [µm]"); plt.ylabel("y' at focus [µm]")
plt.title('Focal intensity (no phase mask)')
plt.colorbar(label='normalized intensity')
plt.xlim(-150, 150)   # µm
plt.ylim(-150, 150)   # µm

plt.subplot(1, 2, 2)
plt.imshow(Ipi[iy, ix], extent=[xfx[0]*1e6, xfx[-1]*1e6, yfy[0]*1e6, yfy[-1]*1e6],
           origin='lower', aspect='equal')
plt.xlabel("x' at focus [µm]"); plt.ylabel("y' at focus [µm]")
plt.title('Focal intensity (half-plane π step)')
plt.colorbar(label='normalized intensity')
plt.xlim(-150, 150)   # µm
plt.ylim(-150, 150)   # µm
plt.tight_layout()
plt.show()

# (3) Line cut through y' = 0 in the focal plane
mid = N // 2
I0_line  = I0[mid, :]
Ipi_line = Ipi[mid, :]

plt.figure(figsize=(7, 4))
plt.plot(xf[0, :]*1e6, I0_line,  label='No phase mask')
plt.plot(xf[0, :]*1e6, Ipi_line, label='π step across x=0')
plt.xlim(-200, 200)
plt.xlabel("x' at focus [µm]")
plt.ylabel('Normalized intensity')
plt.title("Focal-plane line cut (y' = 0)")
plt.xlim(-150, 150)   # µm
plt.legend()
plt.tight_layout()
plt.show()
