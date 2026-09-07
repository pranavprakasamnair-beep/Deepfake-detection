import cv2
import numpy as np

def generate_spatial_attention_mask(img_rgb: np.ndarray) -> dict:
    if img_rgb is None or img_rgb.size == 0:
        return {"attention_anomaly_score": 0.5, "ocular_anomaly": 0.5, "perioral_anomaly": 0.5}

    img_u8 = np.clip(img_rgb, 0, 255).astype(np.uint8)
    h, w, _ = img_u8.shape
    gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # 1. Ocular Attention Zone (Eyes)
    eye_y1, eye_y2 = int(h * 0.20), int(h * 0.42)
    eye_x1, eye_x2 = int(w * 0.18), int(w * 0.82)
    ocular_crop = gray[eye_y1:eye_y2, eye_x1:eye_x2] if eye_y2 > eye_y1 and eye_x2 > eye_x1 else gray

    mean_eye = ocular_crop.mean() + 1e-5
    max_eye = ocular_crop.max()
    eye_glint_ratio = max_eye / mean_eye
    ocular_fake = float(np.clip(1.0 - (eye_glint_ratio / 2.5), 0.0, 1.0))

    # 2. Perioral Attention Zone (Mouth / Teeth / Lips)
    mouth_y1, mouth_y2 = int(h * 0.62), int(h * 0.88)
    mouth_x1, mouth_x2 = int(w * 0.25), int(w * 0.75)
    mouth_crop = img_u8[mouth_y1:mouth_y2, mouth_x1:mouth_x2] if mouth_y2 > mouth_y1 and mouth_x2 > mouth_x1 else img_u8

    mouth_gray = cv2.cvtColor(mouth_crop, cv2.COLOR_RGB2GRAY) if len(mouth_crop.shape) == 3 else mouth_crop
    mouth_laplacian = float(cv2.Laplacian(mouth_gray, cv2.CV_32F).var()) if mouth_gray.size > 0 else 50.0
    perioral_fake = float(np.clip(1.0 - (mouth_laplacian / 80.0), 0.0, 1.0))

    combined_attn = (0.55 * ocular_fake) + (0.45 * perioral_fake)

    return {
        "attention_anomaly_score": round(float(combined_attn), 4),
        "ocular_anomaly": round(ocular_fake, 4),
        "perioral_anomaly": round(perioral_fake, 4)
    }
