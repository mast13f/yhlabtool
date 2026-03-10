"""
imzML + ibd → PNG Ion Images Converter
----------------------------------------
Input:  path to a .imzML file (the .ibd file must be in the same folder)
Output: one PNG image per detected m/z value, saved to an output folder

Usage:
    python imzml_to_images.py path/to/file.imzML --output path/to/output_folder

Optional flags:
    --mz_min 400       only export ions above this m/z
    --mz_max 1800      only export ions below this m/z
    --top 200          only export the top N ions by total intensity (recommended)
    --colormap hot     matplotlib colormap (default: hot)
    --normalize        normalize each image to [0, 255] individually
"""

import os
import argparse
import numpy as np
from pathlib import Path

try:
    from pyimzml.ImzMLParser import ImzMLParser
except ImportError:
    raise ImportError("Run:  pip install pyimzml")

try:
    from PIL import Image
except ImportError:
    raise ImportError("Run:  pip install Pillow")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
except ImportError:
    raise ImportError("Run:  pip install matplotlib")


# ─────────────────────────────────────────────
# CORE CONVERSION
# ─────────────────────────────────────────────

def build_ion_image(parser, target_mz: float, mz_tol: float = 0.1) -> np.ndarray:
    """
    Reconstruct a 2D ion image for a single m/z value.

    For each pixel (x, y), sum intensities within [target_mz - tol, target_mz + tol].
    Returns a 2D float array shaped (height, width).
    """
    coords = parser.coordinates          # list of (x, y, z) or (x, y)
    x_coords = [c[0] for c in coords]
    y_coords = [c[1] for c in coords]

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    width  = x_max - x_min + 1
    height = y_max - y_min + 1

    image = np.zeros((height, width), dtype=np.float32)

    for idx, (x, y, *_) in enumerate(coords):
        mzs, intensities = parser.getspectrum(idx)
        mzs         = np.array(mzs)
        intensities = np.array(intensities)

        mask = (mzs >= target_mz - mz_tol) & (mzs <= target_mz + mz_tol)
        pixel_intensity = intensities[mask].sum()

        image[y - y_min, x - x_min] = pixel_intensity

    return image


def image_to_png(ion_image: np.ndarray, output_path: Path, colormap: str = "hot"):
    """Save a 2D float ion image as a coloured PNG."""
    vmax = ion_image.max()
    if vmax > 0:
        normalised = ion_image / vmax
    else:
        normalised = ion_image

    cmap   = cm.get_cmap(colormap)
    rgba   = cmap(normalised)                      # (H, W, 4) float in [0,1]
    rgb    = (rgba[:, :, :3] * 255).astype(np.uint8)
    Image.fromarray(rgb).save(output_path)


def get_all_mz_values(parser) -> np.ndarray:
    """Collect every unique m/z value across all pixels (can be slow for large files)."""
    print("  Scanning spectra to collect m/z values...")
    all_mzs = set()
    for idx in range(len(parser.coordinates)):
        mzs, _ = parser.getspectrum(idx)
        all_mzs.update(np.round(mzs, 2))          # round to 0.01 Da bins
    return np.array(sorted(all_mzs))


def get_top_mz_values(parser, top_n: int) -> np.ndarray:
    """
    Return the top N m/z values ranked by total intensity across all pixels.
    Much faster than exporting everything.
    """
    print(f"  Scanning spectra to find top {top_n} ions by total intensity...")
    mz_intensity: dict = {}

    for idx in range(len(parser.coordinates)):
        mzs, intensities = parser.getspectrum(idx)
        for mz, intensity in zip(mzs, intensities):
            mz_bin = round(float(mz), 2)
            mz_intensity[mz_bin] = mz_intensity.get(mz_bin, 0.0) + float(intensity)

    sorted_mzs = sorted(mz_intensity, key=mz_intensity.get, reverse=True)
    return np.array(sorted_mzs[:top_n])


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def convert(
    imzml_path: str,
    output_folder: str,
    mz_min: float   = None,
    mz_max: float   = None,
    top_n:  int     = None,
    colormap: str   = "hot",
    mz_tol: float   = 0.1,
):
    imzml_path    = Path(imzml_path)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading: {imzml_path.name}")
    parser = ImzMLParser(str(imzml_path))
    print(f"  Pixels found: {len(parser.coordinates)}")

    # ── Select which m/z values to export ──
    if top_n:
        mz_values = get_top_mz_values(parser, top_n)
    else:
        mz_values = get_all_mz_values(parser)

    # ── Apply m/z range filter ──
    if mz_min is not None:
        mz_values = mz_values[mz_values >= mz_min]
    if mz_max is not None:
        mz_values = mz_values[mz_values <= mz_max]

    print(f"  Exporting {len(mz_values)} ion images to: {output_folder}\n")

    for i, mz in enumerate(mz_values):
        ion_image   = build_ion_image(parser, mz, mz_tol)
        filename    = output_folder / f"ion_{mz:.2f}.png"
        image_to_png(ion_image, filename, colormap)

        if (i + 1) % 10 == 0 or (i + 1) == len(mz_values):
            print(f"  [{i+1}/{len(mz_values)}]  m/z {mz:.2f}  →  {filename.name}")

    print(f"\nDone. {len(mz_values)} images saved to: {output_folder}\n")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert imzML + ibd files into PNG ion images."
    )
    parser.add_argument("imzml",        type=str, help="Path to the .imzML file")
    parser.add_argument("--output",     type=str, default="ion_images",
                        help="Output folder for PNG images (default: ion_images/)")
    parser.add_argument("--mz_min",     type=float, default=None,
                        help="Minimum m/z to export")
    parser.add_argument("--mz_max",     type=float, default=None,
                        help="Maximum m/z to export")
    parser.add_argument("--top",        type=int, default=200,
                        help="Export only top N ions by total intensity (default: 200)")
    parser.add_argument("--colormap",   type=str, default="hot",
                        help="Matplotlib colormap (default: hot). Try: viridis, inferno, gray")
    parser.add_argument("--mz_tol",     type=float, default=0.1,
                        help="m/z tolerance window in Da (default: 0.1)")
    parser.add_argument("--all_mz",     action="store_true",
                        help="Export ALL m/z values (slow — use --top for large files)")

    args = parser.parse_args()

    convert(
        imzml_path    = args.imzml,
        output_folder = args.output,
        mz_min        = args.mz_min,
        mz_max        = args.mz_max,
        top_n         = None if args.all_mz else args.top,
        colormap      = args.colormap,
        mz_tol        = args.mz_tol,
    )