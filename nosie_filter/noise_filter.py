"""
MALDI Image Noise Filter
------------------------
Input:  a folder of ion images
Output: two folders
        - results/  → low-noise, high-quality images
        - noise/    → noisy, low-quality images
"""

import os
import shutil
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from scipy import ndimage
from skimage import filters, measure


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

def compute_autocorrelation(img: np.ndarray) -> float:
    """
    Spatial autocorrelation — measures how similar neighbouring pixels are.
    High value = structured biological signal.
    Near zero  = random noise.
    """
    flat      = img[:-1, :].flatten().astype(float)
    flat_next = img[1:,  :].flatten().astype(float)
    if flat.std() == 0 or flat_next.std() == 0:
        return 0.0
    corr = np.corrcoef(flat, flat_next)[0, 1]
    return float(corr)


def compute_cv(img: np.ndarray) -> float:
    """
    Coefficient of variation — catches flat/empty images (very low CV).
    """
    mean = img.mean()
    if mean == 0:
        return 0.0
    return float(img.std() / mean)


def compute_edge_density(img: np.ndarray) -> float:
    """
    Edge density via Sobel filter.
    Too low  = flat/empty image.
    Too high = pure salt-and-pepper noise.
    """
    edges = filters.sobel(img.astype(float))
    return float((edges > edges.mean()).sum() / edges.size)


def compute_entropy(img: np.ndarray) -> float:
    """
    Shannon entropy — catches both extremes:
    Too low  = uniform/empty.
    Too high = chaotic noise.
    """
    return float(measure.shannon_entropy(img))


def score_image(img: np.ndarray) -> dict:
    """Return all four metrics for one image."""
    return {
        "autocorrelation": compute_autocorrelation(img),
        "cv":              compute_cv(img),
        "edge_density":    compute_edge_density(img),
        "entropy":         compute_entropy(img),
    }


# ─────────────────────────────────────────────
# COMPOSITE SCORE  (0 = noisy, 1 = clean)
# ─────────────────────────────────────────────

def composite_score(metrics: dict) -> float:
    """
    Combine metrics into a single quality score.

    Logic:
    - Autocorrelation contributes positively (more structure = better).
    - CV must be in a mid-range; too low = empty, too high = noise.
    - Edge density must be in a mid-range (Goldilocks).
    - Entropy must be in a mid-range.

    Each sub-score is clipped to [0, 1] before weighting.
    """
    ac = metrics["autocorrelation"]
    cv = metrics["cv"]
    ed = metrics["edge_density"]
    en = metrics["entropy"]

    # Autocorrelation: directly maps to quality
    ac_score = np.clip(ac, 0, 1)

    # CV: penalise extremes; peak score around 0.3–0.8
    cv_score = np.clip(1 - abs(cv - 0.5) / 0.5, 0, 1)

    # Edge density: ideal range ~0.2–0.5
    ed_score = np.clip(1 - abs(ed - 0.35) / 0.35, 0, 1)

    # Entropy: ideal range ~3–6 bits (normalise assuming max ~8)
    en_norm  = en / 8.0
    en_score = np.clip(1 - abs(en_norm - 0.5) / 0.5, 0, 1)

    # Weighted sum  (autocorrelation is the most reliable signal)
    score = (
        0.50 * ac_score +
        0.20 * cv_score +
        0.15 * ed_score +
        0.15 * en_score
    )
    return float(score)


# ─────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def load_image_as_array(path: Path) -> np.ndarray:
    """Load image and convert to float grayscale numpy array."""
    img = Image.open(path).convert("L")          # grayscale
    arr = np.array(img, dtype=np.float32)
    # Normalise to [0, 1]
    if arr.max() > 0:
        arr = arr / arr.max()
    return arr


def run_pipeline(
    input_folder: str,
    threshold: float = 0.4,
    verbose: bool = True,
):
    input_path   = Path(input_folder)
    results_path = input_path.parent / "results"
    noise_path   = input_path.parent / "noise"

    results_path.mkdir(exist_ok=True)
    noise_path.mkdir(exist_ok=True)

    image_files = [
        f for f in input_path.iterdir()
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not image_files:
        print(f"No supported images found in {input_folder}")
        return

    print(f"\nFound {len(image_files)} images. Processing...\n")

    results_count = 0
    noise_count   = 0
    records       = []

    for img_path in sorted(image_files):
        try:
            arr     = load_image_as_array(img_path)
            metrics = score_image(arr)
            score   = composite_score(metrics)

            is_good = score >= threshold
            dest    = results_path if is_good else noise_path

            shutil.copy2(img_path, dest / img_path.name)

            label = "RESULT" if is_good else "NOISE "
            if verbose:
                print(
                    f"[{label}]  {img_path.name:<40}  "
                    f"score={score:.3f}  "
                    f"autocorr={metrics['autocorrelation']:.3f}  "
                    f"cv={metrics['cv']:.3f}  "
                    f"edge={metrics['edge_density']:.3f}  "
                    f"entropy={metrics['entropy']:.2f}"
                )

            if is_good:
                results_count += 1
            else:
                noise_count += 1

            records.append({"file": img_path.name, "score": score, "good": is_good, **metrics})

        except Exception as e:
            print(f"  [ERROR] Could not process {img_path.name}: {e}")

    # ── Summary ──
    print(f"\n{'─'*60}")
    print(f"  Total processed : {len(records)}")
    print(f"  Kept (results/) : {results_count}")
    print(f"  Filtered (noise/): {noise_count}")
    print(f"  Threshold used  : {threshold}")
    print(f"  Output folders  :")
    print(f"    {results_path}")
    print(f"    {noise_path}")
    print(f"{'─'*60}\n")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MALDI ion image noise filter — sorts images into results/ and noise/ folders."
    )
    parser.add_argument(
        "input_folder",
        type=str,
        help="Path to folder containing raw ion images",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="Quality score threshold (0–1). Images above this go to results/. Default: 0.4",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-image output",
    )

    args = parser.parse_args()

    run_pipeline(
        input_folder=args.input_folder,
        threshold=args.threshold,
        verbose=not args.quiet,
    )