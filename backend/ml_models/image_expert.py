"""
Image Expert — Face Extraction + Dual Vision Transformers (dima806 + Wvolf) + Custom EfficientNet.
"""
import logging
from typing import List, Optional

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# Polarity for HuggingFace pre-trained models:
# Index 0 = REAL / AUTHENTIC (dima806 / Wvolf id2label: {0: 'Real', 1: 'Fake'})
# Index 1 = FAKE / MANIPULATED
VIT_REAL_IDX = 0
VIT_FAKE_IDX = 1

def _get_fake_prob(logits: torch.Tensor, model_config) -> float:
    """Extract fake probability using model config id2label or standard polarity."""
    probs = torch.softmax(logits, dim=-1)[0]
    id2label = getattr(model_config, "id2label", {})
    if id2label:
        for idx, lbl in id2label.items():
            lbl_lower = str(lbl).lower()
            if any(term in lbl_lower for term in ["fake", "manipulat", "forger", "deepfake"]):
                return float(probs[int(idx)].cpu().item())
            elif any(term in lbl_lower for term in ["real", "authent", "genuine", "original"]):
                return 1.0 - float(probs[int(idx)].cpu().item())
    return float(probs[VIT_FAKE_IDX].cpu().item())

# Colab EfficientNet: trained with 0=Real, 1=Fake (standard)
EFFNET_FAKE_IDX = 1

_face_cascade = None

def _get_cascade():
    global _face_cascade
    if _face_cascade is None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(path)
    return _face_cascade


def extract_face_crops(image: Image.Image) -> List[Image.Image]:
    """Extract clean face crops from the input image. Returns full image if no face found."""
    cascade = _get_cascade()
    img_cv = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Stricter parameters to prevent false positive detections on clothes/furniture
    detected = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    if len(detected) == 0:
        return [image]

    # Sort faces by size (largest first)
    sorted_faces = sorted(detected, key=lambda f: f[2] * f[3], reverse=True)
    h, w, _ = img_cv.shape

    # If the primary face occupies >45% of the image, the image is already a face portrait.
    # Re-cropping truncates chin/forehead and degrades ViT attention.
    max_face_area = sorted_faces[0][2] * sorted_faces[0][3]
    if max_face_area / (w * h) >= 0.45:
        return [image]

    crops = [image]

    for x, y, fw, fh in sorted_faces[:4]:  # test up to 4 faces in group photos
        pad = 0.20
        x1 = max(0, int(x - fw * pad))
        y1 = max(0, int(y - fh * pad))
        x2 = min(w, int(x + fw * (1.0 + pad)))
        y2 = min(h, int(y + fh * (1.0 + pad)))
        face_bgr = img_cv[y1:y2, x1:x2]
        if face_bgr.shape[0] >= 40 and face_bgr.shape[1] >= 40:
            crops.append(Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)))

    return crops if crops else [image]


def _predict_face_prob(face_pil: Image.Image, registry) -> float:
    """Predict fake probability for a single face crop."""
    from ml_models.model_registry import DEVICE

    scores = []

    # 1. ViT Model 1 (dima806 — synthesis & GANs)
    if registry.vit_model1 is not None and registry.vit_processor1 is not None:
        try:
            inp = registry.vit_processor1(images=face_pil, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                logits1 = registry.vit_model1(**inp).logits
            scores.append(_get_fake_prob(logits1, registry.vit_model1.config))
        except Exception as e:
            logger.debug(f"ViT 1 inference error: {e}")

    # 2. ViT Model 2 (Wvolf — FaceSwap)
    if registry.vit_model2 is not None and registry.vit_processor2 is not None:
        try:
            inp = registry.vit_processor2(images=face_pil, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                logits2 = registry.vit_model2(**inp).logits
            scores.append(_get_fake_prob(logits2, registry.vit_model2.config))
        except Exception as e:
            logger.debug(f"ViT 2 inference error: {e}")

    # 3. Custom EfficientNet (if trained and provided by user from Colab)
    if registry.custom_efficientnet is not None and registry.efficientnet_transform is not None:
        try:
            tensor = registry.efficientnet_transform(face_pil.convert("RGB")).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                probs = torch.softmax(registry.custom_efficientnet(tensor), dim=1)[0]
            effnet_score = float(probs[EFFNET_FAKE_IDX].cpu().item())
            # When custom model is available, weight it strongly
            if scores:
                vit_avg = np.mean(scores)
                return float(0.60 * effnet_score + 0.40 * vit_avg)
            return effnet_score
        except Exception as e:
            logger.debug(f"Custom EfficientNet error: {e}")

    if not scores:
        return 0.50

    if len(scores) == 1:
        return float(scores[0])

    # Dual ViT fusion:
    s1, s2 = scores[0], scores[1]
    max_s = max(s1, s2)
    min_s = min(s1, s2)

    # If either expert detects clear manipulation, trust the positive finding
    if max_s > 0.65:
        return float(0.80 * max_s + 0.20 * min_s)
    elif max_s < 0.25:
        return float(min_s)
    else:
        return float(0.60 * max_s + 0.40 * min_s)


def analyze_image(image: Image.Image, registry) -> dict:
    """Run full image forensic analysis with whole-image anchoring and face validation."""
    # 1. Primary evaluation on full image context
    full_score = _predict_face_prob(image, registry)

    # 2. Extract detected face crops
    crops = extract_face_crops(image)
    valid_face_crops = [c for c in crops if c is not image]
    face_scores = [_predict_face_prob(c, registry) for c in valid_face_crops]

    if not face_scores:
        final_score = full_score
        all_scores = [full_score]
    else:
        all_scores = [full_score] + face_scores
        # If the holistic image is decisively authentic (<0.25), reject spurious background wall crops
        if full_score < 0.25:
            final_score = full_score
        elif full_score > 0.75:
            final_score = max(full_score, max(face_scores))
        else:
            final_score = 0.50 * full_score + 0.50 * float(np.mean(face_scores))

    verdict = "FAKE" if final_score > 0.50 else "REAL"

    return {
        "score": round(float(np.clip(final_score, 0.0, 1.0)), 4),
        "verdict": verdict,
        "details": {
            "faces_analyzed": max(1, len(valid_face_crops)),
            "face_scores": [round(s, 4) for s in all_scores],
            "custom_colab_active": registry.has_custom,
            "models_used": "Dual ViT (dima806 + Wvolf FaceSwap)" + (" + Colab EfficientNet-B4" if registry.has_custom else ""),
        }
    }
