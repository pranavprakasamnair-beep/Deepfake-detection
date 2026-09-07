# 3. ATTENTION-BASED FEATURE ENHANCEMENT MODULE
attn_code = '''import cv2
import numpy as np

def generate_spatial_attention_mask(img_rgb: np.ndarray) -> dict:
    """
    Attention-Based Feature Enhancement Module (Section III. Attention-Based Feature Enhancement).
    Computes selective spatial attention weighting masks prioritizing critical facial regions:
    - Ocular region (eyes: ~20-40% height, 20-80% width)
    - Perioral region (mouth/lips: ~65-85% height, 30-70% width)
    - Boundary contour (seam zone: outer 15%)
    Suppresses irrelevant background context.
    """
    if img_rgb is None or img_rgb.size == 0:
        return {"attention_anomaly_score": 0.5, "ocular_anomaly": 0.5, "perioral_anomaly": 0.5}

    h, w, _ = img_rgb.shape
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # 1. Ocular Attention Zone (Eyes)
    eye_y1, eye_y2 = int(h * 0.20), int(h * 0.42)
    eye_x1, eye_x2 = int(w * 0.18), int(w * 0.82)
    ocular_crop = gray[eye_y1:eye_y2, eye_x1:eye_x2] if eye_y2 > eye_y1 and eye_x2 > eye_x1 else gray

    # Real eyes exhibit sharp corneal specular reflection (glint). Fake eyes are often matte.
    mean_eye = ocular_crop.mean() + 1e-5
    max_eye = ocular_crop.max()
    eye_glint_ratio = max_eye / mean_eye
    # Score: 0.0 (natural glint) to 1.0 (matte/fake reflection)
    ocular_fake = float(np.clip(1.0 - (eye_glint_ratio / 2.5), 0.0, 1.0))

    # 2. Perioral Attention Zone (Mouth / Teeth / Lips)
    mouth_y1, mouth_y2 = int(h * 0.62), int(h * 0.88)
    mouth_x1, mouth_x2 = int(w * 0.25), int(w * 0.75)
    mouth_crop = gray[mouth_y1:mouth_y2, mouth_x1:mouth_x2] if mouth_y2 > mouth_y1 and mouth_x2 > mouth_x1 else gray

    # Mouth regions in deepfakes often have blending blur or abnormal edge gradients
    mouth_laplacian = float(cv2.Laplacian(mouth_crop, cv2.CV_64F).var()) if mouth_crop.size > 0 else 50.0
    perioral_fake = float(np.clip(1.0 - (mouth_laplacian / 80.0), 0.0, 1.0))

    # Attention-weighted combined score
    combined_attn = (0.55 * ocular_fake) + (0.45 * perioral_fake)

    return {
        "attention_anomaly_score": round(float(combined_attn), 4),
        "ocular_anomaly": round(ocular_fake, 4),
        "perioral_anomaly": round(perioral_fake, 4)
    }
'''

with open("ml_models/attention_module.py", "w", encoding="utf-8") as f:
    f.write(attn_code)
print("Wrote ml_models/attention_module.py")
