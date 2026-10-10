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

# Softmax temperature. Fine-tuned ViTs are over-confident (probabilities pinned at
# 0.00 / 1.00), which throws away ranking information. T > 1 keeps scores graded so
# the learned cut-off from diagnose.py can separate images.
VIT_TEMPERATURE = 3.0

def _get_fake_prob(logits: torch.Tensor, model_config) -> float:
    """Extract fake probability using model config id2label or standard polarity."""
    probs = torch.softmax(logits / VIT_TEMPERATURE, dim=-1)[0]
    id2label = getattr(model_config, "id2label", {})
    if id2label:
        for idx, lbl in id2label.items():
            lbl_lower = str(lbl).lower()
            if any(term in lbl_lower for term in ["fake", "manipulat", "forger", "deepfake"]):
                return float(probs[int(idx)].cpu().item())
            elif any(term in lbl_lower for term in ["real", "authent", "genuine", "original"]):
                return 1.0 - float(probs[int(idx)].cpu().item())
    return float(probs[VIT_FAKE_IDX].cpu().item())

def vit_fake_prob(registry, which: int, img: Image.Image, apply_flip: bool = True) -> float:
    """
    Fake probability from ViT model `which` (1 or 2) for a PIL image.
    `registry.vitN_flip` is set at startup by polarity calibration against known
    real/fake reference images, so a wrong id2label in a config cannot invert verdicts.
    """
    from ml_models.model_registry import DEVICE

    model = registry.vit_model1 if which == 1 else registry.vit_model2
    processor = registry.vit_processor1 if which == 1 else registry.vit_processor2
    inp = processor(images=img.convert("RGB"), return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = model(**inp).logits
    p = _get_fake_prob(logits, model.config)
    if apply_flip and getattr(registry, f"vit{which}_flip", False):
        p = 1.0 - p
    return float(p)


# Colab EfficientNet (colab/train_deepfake_efficientnet.py) was trained with
# output 0 = FAKE, 1 = REAL. The registry also verifies this against reference images.
EFFNET_FAKE_IDX = 0


def effnet_fake_prob(registry, img: Image.Image, idx: Optional[int] = None) -> float:
    tensor = registry.efficientnet_transform(img.convert("RGB")).unsqueeze(0).to(
        next(registry.custom_efficientnet.parameters()).device
    )
    with torch.no_grad():
        probs = torch.softmax(registry.custom_efficientnet(tensor), dim=1)[0]
    if idx is None:
        idx = getattr(registry, "effnet_fake_idx", EFFNET_FAKE_IDX)
    return float(probs[idx].cpu().item())

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
        # 1. Tight face crop (focuses strictly on face region for deepfake neural anomalies)
        face_tight = img_cv[y:y+fh, x:x+fw]
        if face_tight.shape[0] >= 40 and face_tight.shape[1] >= 40:
            crops.append(Image.fromarray(cv2.cvtColor(face_tight, cv2.COLOR_BGR2RGB)))

        # 2. Padded face crop with boundary context
        pad = 0.15
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
            scores.append(vit_fake_prob(registry, 1, face_pil))
        except Exception as e:
            logger.warning(f"ViT 1 inference error: {e}")

    # 2. ViT Model 2 (Wvolf — FaceSwap)
    if registry.vit_model2 is not None and registry.vit_processor2 is not None:
        try:
            scores.append(vit_fake_prob(registry, 2, face_pil))
        except Exception as e:
            logger.warning(f"ViT 2 inference error: {e}")

    # 3. Custom EfficientNet (if trained and provided by user from Colab)
    if registry.custom_efficientnet is not None and registry.efficientnet_transform is not None:
        try:
            effnet_score = effnet_fake_prob(registry, face_pil)
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


def compute_face_ela_ratio(image: Image.Image, face_box: tuple) -> float:
    """Error Level Analysis ratio: face ELA vs background ELA."""
    try:
        import io
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=90)
        buf.seek(0)
        recompressed = Image.open(buf).convert("RGB")
        orig_arr = np.array(image.convert("RGB"), dtype=np.float32)
        recomp_arr = np.array(recompressed, dtype=np.float32)
        diff = np.abs(orig_arr - recomp_arr)
        x, y, fw, fh = face_box
        face_ela = float(diff[y:y+fh, x:x+fw].mean())
        bg_ela = float(diff.mean())
        return face_ela / (bg_ela + 1e-5)
    except Exception:
        return 1.0


def analyze_image(image: Image.Image, registry) -> dict:
    """Run full image forensic analysis with general face swap & multi-scale synthesis detection."""
    # 1. Primary evaluation on full image context
    full_score = _predict_face_prob(image, registry)

    # 2. Extract detected faces using Haar cascade
    cascade = _get_cascade()
    img_cv = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    h, w, _ = img_cv.shape
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    detected_type = "GLOBAL_ANALYSIS"
    face_scores = []

    if len(faces) == 0:
        final_score = full_score
        all_scores = [full_score]
    else:
        sorted_faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        detected_type = "NATURAL_PORTRAIT"
        is_faceswap = False

        for x, y, fw, fh in sorted_faces[:3]:
            face_tight = img_cv[y:y+fh, x:x+fw]
            if face_tight.shape[0] >= 40 and face_tight.shape[1] >= 40:
                tight_pil = Image.fromarray(cv2.cvtColor(face_tight, cv2.COLOR_BGR2RGB))
                t_score = _predict_face_prob(tight_pil, registry)
                face_scores.append(t_score)

                ela_ratio = compute_face_ela_ratio(image, (x, y, fw, fh))

                # FaceSwap detection: high neural fake probability on tight face + compression discrepancy
                if t_score > 0.70 and ela_ratio < 0.95:
                    is_faceswap = True
                    detected_type = "FACESWAP_SPLICING"
                    final_score = t_score
                    break

        if not is_faceswap:
            if full_score > 0.60:
                final_score = full_score
                detected_type = "SYNTHETIC_GENERATION"
            else:
                final_score = min(full_score, float(np.mean(face_scores)) if face_scores else full_score)

        all_scores = [full_score] + face_scores

    verdict = "FAKE" if final_score > 0.50 else "REAL"

    # Per-model raw fake probabilities on the whole image (for debugging / the report)
    model_probs = {}
    for name, which, mdl in (("vit1", 1, registry.vit_model1), ("vit2", 2, registry.vit_model2)):
        if mdl is not None:
            try:
                model_probs[name] = round(vit_fake_prob(registry, which, image), 4)
            except Exception as e:
                logger.warning(f"{name} breakdown failed: {e}")
    if registry.custom_efficientnet is not None:
        try:
            model_probs["effnet"] = round(effnet_fake_prob(registry, image), 4)
        except Exception as e:
            logger.warning(f"effnet breakdown failed: {e}")
    logger.info(f"Forensic type: {detected_type}, score: {final_score:.4f}, P(fake): {model_probs}")

    return {
        "score": round(float(np.clip(final_score, 0.0, 1.0)), 4),
        "verdict": verdict,
        "details": {
            "forensic_type": detected_type,
            "model_probs": model_probs,
            "vit1_flipped": bool(getattr(registry, "vit1_flip", False)),
            "vit2_flipped": bool(getattr(registry, "vit2_flip", False)),
            "faces_analyzed": max(1, len(face_scores)),
            "face_scores": [round(s, 4) for s in all_scores],
            "custom_colab_active": registry.has_custom,
            "models_used": "Dual ViT (dima806 + Wvolf FaceSwap)" + (" + Colab EfficientNet-B4" if registry.has_custom else ""),
        }
    }