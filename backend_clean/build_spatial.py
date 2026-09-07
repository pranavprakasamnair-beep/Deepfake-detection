import cv2
import numpy as np

# 2. SPATIAL CNN & HEURISTIC ARTIFACT EXPERT
spatial_code = '''import cv2
import numpy as np

def extract_boundary_seams(img_rgb: np.ndarray) -> dict:
    """
    Hunts for blending seams and edge discontinuity at face boundaries.
    Face swaps and GAN inpainting produce localized edge anomalies around
    the outer 15% perimeter of the crop.
    """
    if img_rgb is None or img_rgb.size == 0:
        return {"boundary_score": 0.0, "edge_density": 0.0}

    h, w, _ = img_rgb.shape
    if h < 20 or w < 20:
        return {"boundary_score": 0.0, "edge_density": 0.0}

    mask = np.zeros((h, w), dtype=np.float32)
    by = int(h * 0.15)
    bx = int(w * 0.15)
    mask[:by, :] = 1.0
    mask[-by:, :] = 1.0
    mask[:, :bx] = 1.0
    mask[:, -bx:] = 1.0

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    boundary_edges = edges * mask
    edge_density = float(boundary_edges.mean() / 255.0)
    boundary_fake_prob = float(min(edge_density / 0.10, 1.0))

    return {
        "boundary_score": round(boundary_fake_prob, 4),
        "edge_density": round(edge_density, 4)
    }

def extract_texture_smoothness(img_rgb: np.ndarray) -> dict:
    """
    Measures facial skin texture variance using CLAHE-normalized Laplacian operator.
    Real faces have pore and fine wrinkle micro-textures; GANs produce unnaturally smooth surfaces.
    """
    if img_rgb is None or img_rgb.size == 0:
        return {"texture_score": 0.5, "laplacian_var": 0.0}

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm_gray = clahe.apply(gray)
    var = float(cv2.Laplacian(norm_gray, cv2.CV_64F).var())
    
    # var > 100 is natural high detail; var < 25 is heavily smoothed
    natural_texture = float(min(var / 100.0, 1.0))
    texture_fake_prob = float(1.0 - natural_texture)

    return {
        "texture_score": round(texture_fake_prob, 4),
        "laplacian_var": round(var, 2)
    }

def extract_lighting_consistency(img_rgb: np.ndarray) -> dict:
    """
    Analyzes specular highlight and lighting distribution across bilateral facial halves.
    """
    if img_rgb is None or img_rgb.size == 0:
        return {"lighting_mismatch": 0.0}

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape
    mid = w // 2
    left_half = gray[:, :mid]
    right_half = cv2.flip(gray[:, mid:], 1)
    
    # Resize to align
    min_w = min(left_half.shape[1], right_half.shape[1])
    l_m = np.mean(left_half[:, :min_w])
    r_m = np.mean(right_half[:, :min_w])
    
    diff_ratio = abs(l_m - r_m) / (max(l_m, r_m) + 1e-5)
    lighting_anomaly = float(np.clip(diff_ratio / 0.40, 0.0, 1.0))

    return {
        "lighting_mismatch": round(lighting_anomaly, 4)
    }

def full_spatial_analysis(img_rgb: np.ndarray) -> dict:
    """
    Consolidated CNN Spatial Feature Extraction Module (Section III. Spatial Feature Extraction).
    """
    bound_res = extract_boundary_seams(img_rgb)
    text_res = extract_texture_smoothness(img_rgb)
    light_res = extract_lighting_consistency(img_rgb)

    spatial_fake_score = (0.50 * bound_res["boundary_score"]) + (0.35 * text_res["texture_score"]) + (0.15 * light_res["lighting_mismatch"])

    return {
        "spatial_fake_score": round(float(np.clip(spatial_fake_score, 0.0, 1.0)), 4),
        "boundary_score": bound_res["boundary_score"],
        "texture_score": text_res["texture_score"],
        "lighting_mismatch": light_res["lighting_mismatch"],
        "laplacian_var": text_res["laplacian_var"]
    }
'''

with open("ml_models/spatial_cnn_expert.py", "w", encoding="utf-8") as f:
    f.write(spatial_code)
print("Wrote ml_models/spatial_cnn_expert.py")
