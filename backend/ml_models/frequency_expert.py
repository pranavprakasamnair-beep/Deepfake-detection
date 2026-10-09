"""
Frequency Expert — FFT + DCT analysis for detecting AI generation artifacts.

AI-generated images have characteristic high-frequency artifacts from the
upsampling (transposed convolution) grid pattern used during GAN generation.
"""
import logging
from typing import Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def analyze_frequency(image: Image.Image) -> dict:
    """
    Analyze frequency domain characteristics of an image.
    Returns dict with score (0-1, higher = more fake), verdict, details.
    """
    try:
        img_gray = np.array(image.convert("L"), dtype=np.float32)

        fft_score = _fft_artifact_score(img_gray)
        dct_score = _dct_block_artifact_score(img_gray)
        ela_score = _ela_score(image)

        # Weighted combination
        final_score = 0.45 * fft_score + 0.35 * dct_score + 0.20 * ela_score

        # Clamp
        final_score = float(np.clip(final_score, 0.0, 1.0))

        return {
            "score": round(final_score, 4),
            "verdict": "FAKE" if final_score > 0.50 else "REAL",
            "details": {
                "fft_score": round(fft_score, 4),
                "dct_score": round(dct_score, 4),
                "ela_score": round(ela_score, 4),
            },
        }
    except Exception as e:
        logger.error(f"Frequency analysis error: {e}")
        return {"score": 0.5, "verdict": "UNKNOWN", "details": {"error": str(e)}}


def _fft_artifact_score(gray: np.ndarray) -> float:
    """
    Detect periodic grid artifacts in the frequency domain.
    Standardized to 512x512 so high-resolution phone cameras do not trigger false peaks.
    """
    if gray.shape[0] != 512 or gray.shape[1] != 512:
        gray_std = cv2.resize(gray, (512, 512), interpolation=cv2.INTER_AREA)
    else:
        gray_std = gray

    f = np.fft.fft2(gray_std)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1e-8)

    cy, cx = 256, 256

    # Center region (low frequency) — exclude it
    center_mask = np.zeros_like(magnitude, dtype=bool)
    center_mask[cy - 20:cy + 20, cx - 20:cx + 20] = True

    high_freq = magnitude[~center_mask]
    mean_hf = high_freq.mean()
    std_hf = high_freq.std()

    # Count peaks significantly above mean (GAN grid = periodic spikes)
    threshold = mean_hf + 3.5 * std_hf
    n_peaks = int((high_freq > threshold).sum())

    # Standardized: natural camera images ~20-80 peaks; GAN grid images: 200+ peaks
    score = min(1.0, n_peaks / 250.0)
    return float(score)


def _dct_block_artifact_score(gray: np.ndarray) -> float:
    """
    Detect block artifacts from JPEG compression mismatch or DCT patterns.
    Re-saved AI faces often show DCT inconsistencies at 8x8 block boundaries.
    """
    from scipy.fft import dct

    h, w = gray.shape
    h8 = (h // 8) * 8
    w8 = (w // 8) * 8
    gray_cropped = gray[:h8, :w8]

    # Compute horizontal and vertical block boundary differences
    if h8 < 16 or w8 < 16:
        return 0.5

    h_diffs = []
    for y in range(8, h8, 8):
        row_diff = np.abs(gray_cropped[y, :] - gray_cropped[y - 1, :]).mean()
        h_diffs.append(row_diff)

    v_diffs = []
    for x in range(8, w8, 8):
        col_diff = np.abs(gray_cropped[:, x] - gray_cropped[:, x - 1]).mean()
        v_diffs.append(col_diff)

    if not h_diffs or not v_diffs:
        return 0.5

    mean_boundary = (np.mean(h_diffs) + np.mean(v_diffs)) / 2

    # Compare to interior pixel differences
    interior_diff = np.abs(np.diff(gray_cropped, axis=0)).mean()

    if interior_diff < 1e-6:
        return 0.5

    ratio = mean_boundary / (interior_diff + 1e-6)

    # Ratio close to 1 → natural. Ratio significantly > 1 → block artifacts.
    # Clamp to [0,1]
    score = float(np.clip((ratio - 1.0) / 3.0, 0.0, 1.0))
    return score


def _ela_score(image: Image.Image) -> float:
    """
    Error Level Analysis (ELA) — detects regions at different compression levels.
    AI-generated or spliced faces often show ELA anomalies.
    """
    import io

    try:
        # Re-save at quality 85
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=85)
        buffer.seek(0)
        resaved = Image.open(buffer).convert("RGB")

        orig_array = np.array(image.convert("RGB"), dtype=np.float32)
        resaved_array = np.array(resaved, dtype=np.float32)

        ela = np.abs(orig_array - resaved_array)
        ela_mean = ela.mean()
        ela_std = ela.std()
        ela_max = ela.max()

        # ELA characteristics:
        # Authentic uniform-compression: low, even ELA
        # Spliced/AI regions: high ELA in specific areas
        # Score based on spatial variance of ELA (high variance → suspicious)
        
        if ela_max < 1e-6:
            return 0.4  # Pristine PNG — lean slightly real

        # Coefficient of variation
        cv = ela_std / (ela_mean + 1e-6)
        score = float(np.clip(cv / 5.0, 0.0, 1.0))
        return score
    except Exception:
        return 0.5
