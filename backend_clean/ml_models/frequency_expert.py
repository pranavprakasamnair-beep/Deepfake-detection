import cv2
import numpy as np
from scipy.fftpack import dct

def analyze_fft_spectrum(img_rgb: np.ndarray) -> dict:
    if img_rgb is None or img_rgb.size == 0:
        return {"fft_score": 0.5, "high_low_ratio": 0.0}

    img_u8 = np.clip(img_rgb, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(fshift))

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2

    low_radius = max(5, min(h, w) // 16)
    low_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(low_mask, (cx, cy), low_radius, 1, -1)
    low_freq = float(magnitude[low_mask > 0].mean()) if np.sum(low_mask) > 0 else 1.0

    high_radius = max(15, min(h, w) // 5)
    high_mask = np.ones((h, w), dtype=np.uint8)
    cv2.circle(high_mask, (cx, cy), high_radius, 0, -1)
    high_freq = float(magnitude[high_mask > 0].mean()) if np.sum(high_mask) > 0 else 0.0

    ratio = high_freq / (low_freq + 1e-5)
    naturalness = np.clip((ratio - 0.35) / 0.30, 0.0, 1.0)
    fft_fake_prob = float(1.0 - naturalness)

    return {
        "fft_score": round(fft_fake_prob, 4),
        "high_low_ratio": round(float(ratio), 4)
    }

def analyze_dct_spectrum(img_rgb: np.ndarray, block_size: int = 8) -> dict:
    if img_rgb is None or img_rgb.size == 0:
        return {"dct_score": 0.5, "high_freq_energy": 0.0}

    img_u8 = np.clip(img_rgb, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape
    h_blocks = h // block_size
    w_blocks = w // block_size

    if h_blocks == 0 or w_blocks == 0:
        return {"dct_score": 0.5, "high_freq_energy": 0.0}

    high_freq_energies = []
    for i in range(h_blocks):
        for j in range(w_blocks):
            block = gray[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            dct_block = dct(dct(block.T, norm="ortho").T, norm="ortho")
            high_band = np.abs(dct_block[block_size//2:, block_size//2:])
            high_freq_energies.append(np.mean(high_band))

    energy_std = float(np.std(high_freq_energies)) if high_freq_energies else 0.0
    energy_mean = float(np.mean(high_freq_energies)) if high_freq_energies else 0.0

    if energy_mean < 0.5:
        dct_fake_prob = 0.75
    else:
        dct_fake_prob = float(np.clip((energy_mean - energy_std) / (energy_mean + 1e-4) * 0.3, 0.0, 0.35))

    return {
        "dct_score": round(dct_fake_prob, 4),
        "high_freq_energy": round(energy_mean, 4)
    }

def analyze_high_frequency_noise(img_rgb: np.ndarray) -> dict:
    if img_rgb is None or img_rgb.size == 0:
        return {"noise_fake_score": 0.5, "noise_std": 0.0}

    img_u8 = np.clip(img_rgb, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    noise_residual = gray - blurred
    noise_std = float(np.std(noise_residual))

    if noise_std < 1.5:
        noise_fake_score = 1.0 - (noise_std / 1.5)
    elif noise_std > 6.0:
        noise_fake_score = min(1.0, (noise_std - 6.0) / 4.0)
    else:
        noise_fake_score = 0.1

    return {
        "noise_fake_score": round(float(noise_fake_score), 4),
        "noise_std": round(noise_std, 4)
    }

def full_frequency_analysis(img_rgb: np.ndarray) -> dict:
    fft_res = analyze_fft_spectrum(img_rgb)
    dct_res = analyze_dct_spectrum(img_rgb)
    noise_res = analyze_high_frequency_noise(img_rgb)

    combined = (0.40 * fft_res["fft_score"]) + (0.35 * dct_res["dct_score"]) + (0.25 * noise_res["noise_fake_score"])

    return {
        "frequency_fake_score": round(float(np.clip(combined, 0.0, 1.0)), 4),
        "fft_score": fft_res["fft_score"],
        "dct_score": dct_res["dct_score"],
        "noise_score": noise_res["noise_fake_score"],
        "high_low_ratio": fft_res["high_low_ratio"],
        "noise_std": noise_res["noise_std"]
    }
