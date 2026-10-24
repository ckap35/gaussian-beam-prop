"""Compare a focused Gaussian beam with and without a half-plane pi phase step."""

import argparse
from pathlib import Path

import numpy as np


WAVELENGTH = 632.8e-9  # m
FOCAL_LENGTH = 150e-3  # m
BEAM_RADIUS = 1.5e-3  # 1/e field radius at the lens, m
GRID_SIZE = 1024
GRID_SPAN = 40e-3  # m


def focal_field(pupil, dx, wavelength, focal_length):
    """Fraunhofer field, including pixel area and the 1/(i*lambda*f) factor."""
    field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil)))
    return field * dx**2 / (1j * wavelength * focal_length)


def make_figures():
    """Return figures for the pupil, focal intensities, and central line cut."""
    import matplotlib.pyplot as plt

    dx = GRID_SPAN / GRID_SIZE
    x = (np.arange(GRID_SIZE) - GRID_SIZE // 2) * dx
    X, Y = np.meshgrid(x, x, indexing="xy")
    focal_x = WAVELENGTH * FOCAL_LENGTH * np.fft.fftshift(
        np.fft.fftfreq(GRID_SIZE, d=dx)
    )

    pupil = np.exp(-(X**2 + Y**2) / BEAM_RADIUS**2)
    phase_mask = np.where(X >= 0, 1.0, -1.0)
    # The taper is negligible over the beam and damps the grid edges.
    taper = np.exp(-((X**2 + Y**2) / (0.45 * GRID_SPAN)**2)**8)
    pupil *= taper
    masked_pupil = pupil * phase_mask

    intensity = np.abs(focal_field(pupil, dx, WAVELENGTH, FOCAL_LENGTH))**2
    masked_intensity = np.abs(
        focal_field(masked_pupil, dx, WAVELENGTH, FOCAL_LENGTH)
    )**2
    # Each image uses its own peak; these plots compare shape, not peak power.
    intensity /= intensity.max()
    masked_intensity /= masked_intensity.max()

    def extent(axis, scale):
        half_pixel = (axis[1] - axis[0]) / 2
        return [(axis[0] - half_pixel) * scale,
                (axis[-1] + half_pixel) * scale] * 2

    pupil_figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, values, title, label in zip(
        axes, [pupil, phase_mask],
        ["Gaussian pupil amplitude", "Half-plane phase multiplier"],
        ["Field amplitude", "Field multiplier"],
    ):
        image = ax.imshow(values, extent=extent(x, 1e3), origin="lower")
        ax.set(xlabel="x at lens [mm]", ylabel="y at lens [mm]",
               title=title, xlim=(-5, 5), ylim=(-5, 5))
        pupil_figure.colorbar(image, ax=ax, label=label)
    pupil_figure.tight_layout()

    focus_figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    crop = slice(GRID_SIZE // 4, 3 * GRID_SIZE // 4)
    for ax, values, title in zip(
        axes, [intensity, masked_intensity],
        ["No phase mask", "Half-plane π phase step"],
    ):
        image = ax.imshow(
            values[crop, crop], extent=extent(focal_x[crop], 1e6),
            origin="lower", vmin=0, vmax=1,
        )
        ax.set(xlabel="x at focus [µm]", ylabel="y at focus [µm]",
               title=title, xlim=(-80, 80), ylim=(-80, 80))
        focus_figure.colorbar(image, ax=ax, label="Intensity / own peak")
    focus_figure.tight_layout()

    line_figure, ax = plt.subplots(figsize=(7, 4))
    mid = GRID_SIZE // 2
    ax.plot(focal_x * 1e6, intensity[mid], label="No phase mask")
    ax.plot(focal_x * 1e6, masked_intensity[mid], label="π phase step")
    ax.set(xlabel="x at focus [µm]", ylabel="Intensity / own peak",
           title="Focal-plane line cut at y = 0", xlim=(-150, 150))
    ax.legend()
    line_figure.tight_layout()
    return pupil_figure, focus_figure, line_figure


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Save three PNGs in this directory")
    args = parser.parse_args(argv)

    import matplotlib.pyplot as plt

    figures = make_figures()
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        for figure, name in zip(figures, ["pupil.png", "focus.png", "line_cut.png"]):
            figure.savefig(args.output / name, dpi=160)
            plt.close(figure)
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
