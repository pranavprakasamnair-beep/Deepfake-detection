import os
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model 1: dima806 — strong on AI-generated faces
MODEL_1_PATH = os.path.join(BASE_DIR, "models", "vit_finetuned")
# Model 2: Wvolf — strong on face swaps
MODEL_2_PATH = os.path.join(BASE_DIR, "models", "vit_faceswap")

# Polarity for HuggingFace models (id2label: {0: 'Real', 1: 'Fake'})
MODEL_1_FAKE_IDX_OVERRIDE = 1
MODEL_2_FAKE_IDX_OVERRIDE = 1

# Hardware acceleration target
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"[INFO] Hardware acceleration target: {device}")

processor1 = None
model1 = None
fake_label_idx1 = None

processor2 = None
model2 = None
fake_label_idx2 = None


def _resolve_fake_index(model, model_name: str, override: int = None) -> int:
    """
    Determines which class index corresponds to 'fake' for a given model.
    Never silently guesses — if it can't confidently resolve, it raises a
    loud warning and falls back to a safe default while telling you exactly
    what to check.
    """
    if override is not None:
        print(f"[INFO] {model_name}: using MANUAL OVERRIDE fake index = {override}")
        return override

    if not (hasattr(model, "config") and hasattr(model.config, "id2label")):
        print(f"[WARNING] {model_name}: model has no id2label config at all. "
              f"Defaulting to index 1 — VERIFY THIS MANUALLY.")
        return 1

    id2label = model.config.id2label
    print(f"[INFO] {model_name} full id2label map: {id2label}")

    fake_idx = None
    real_idx = None
    for idx, label in id2label.items():
        label_lower = str(label).lower()
        if any(w in label_lower for w in ["fake", "synthesized", "spoof", "deepfake", "manipulated"]):
            fake_idx = int(idx)
        if any(w in label_lower for w in ["real", "authentic", "genuine", "pristine"]):
            real_idx = int(idx)

    if fake_idx is not None:
        print(f"[OK] {model_name}: resolved FAKE index = {fake_idx} (matched label text)")
        return fake_idx

    if real_idx is not None:
        # We found "real" but not "fake" by name -> fake is whichever index isn't real
        other_indices = [int(i) for i in id2label.keys() if int(i) != real_idx]
        if len(other_indices) == 1:
            inferred = other_indices[0]
            print(f"[OK] {model_name}: resolved FAKE index = {inferred} "
                  f"(inferred as 'not real', since REAL index {real_idx} was matched by name)")
            return inferred

    # Could not confidently resolve — this is the dangerous silent-failure case.
    print(f"[SEVERE WARNING] {model_name}: could NOT confidently resolve the FAKE "
          f"class index from id2label={id2label}. Defaulting to index 1, but this "
          f"is an UNVERIFIED GUESS. If your real/fake outputs look swapped, this is "
          f"almost certainly why. Set MODEL_{'1' if 'Model 1' in model_name else '2'}"
          f"_FAKE_IDX_OVERRIDE at the top of this file to the correct index (0 or 1) "
          f"based on the id2label map printed above.")
    return 1


# --- Load Model 1 ---
try:
    processor1 = AutoImageProcessor.from_pretrained(MODEL_1_PATH, local_files_only=True)
    model1 = AutoModelForImageClassification.from_pretrained(MODEL_1_PATH, local_files_only=True)
    model1.to(device)
    model1.eval()
    fake_label_idx1 = _resolve_fake_index(model1, "Model 1 (dima806 ViT)", MODEL_1_FAKE_IDX_OVERRIDE)
    print(f"[OK] Model 1 (dima806 ViT) loaded. FAKE label index: {fake_label_idx1}")
except Exception as e:
    print(f"[WARNING] Failed to load Model 1. Error: {e}")

# --- Load Model 2 ---
try:
    processor2 = AutoImageProcessor.from_pretrained(MODEL_2_PATH, local_files_only=True)
    model2 = AutoModelForImageClassification.from_pretrained(MODEL_2_PATH, local_files_only=True)
    model2.to(device)
    model2.eval()
    fake_label_idx2 = _resolve_fake_index(model2, "Model 2 (Wvolf FaceSwap ViT)", MODEL_2_FAKE_IDX_OVERRIDE)
    print(f"[OK] Model 2 (Wvolf FaceSwap ViT) loaded. FAKE label index: {fake_label_idx2}")
except Exception as e:
    print(f"[WARNING] Failed to load Model 2. Error: {e}")


def calibrate_probability(logits: torch.Tensor, temperature: float = 1.4) -> torch.Tensor:
    """Temperature scaling softening overconfident deep network predictions."""
    return F.softmax(logits / temperature, dim=-1)


def predict_vit(pil_image: Image.Image) -> float:
    """Single image inference — averaged across both ViT models."""
    scores = []
    if model1 is not None and processor1 is not None:
        try:
            inputs = processor1(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                probs = calibrate_probability(model1(**inputs).logits, temperature=1.4)[0]
            scores.append(float(probs[fake_label_idx1].cpu().item()))
        except Exception as e:
            pass

    if model2 is not None and processor2 is not None:
        try:
            inputs = processor2(images=pil_image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                probs = calibrate_probability(model2(**inputs).logits, temperature=1.4)[0]
            scores.append(float(probs[fake_label_idx2].cpu().item()))
        except Exception as e:
            pass

    if not scores:
        return 0.5
    if len(scores) == 1:
        return float(scores[0])
    
    # Dual-expert ensemble: Model 1 (GAN synthesis) + Model 2 (FaceSwap)
    # If both models agree it's real (< 0.25), output is firmly real.
    # If either specialized expert detects strong manipulation (> 0.75), output escalates.
    max_s = max(scores)
    min_s = min(scores)
    if max_s > 0.70:
        return float(0.75 * max_s + 0.25 * min_s)
    elif max_s < 0.30:
        return float(min_s)
    else:
        return float(0.60 * max_s + 0.40 * min_s)


def predict_vit_batch(pil_images: list) -> list:
    """Batch inference across video frame sequence."""
    if not pil_images:
        return []

    scores_all = [[] for _ in range(len(pil_images))]

    if model1 is not None and processor1 is not None:
        try:
            inputs = processor1(images=pil_images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                probs = calibrate_probability(model1(**inputs).logits, temperature=1.4)
            for i, p in enumerate(probs[:, fake_label_idx1].cpu().numpy().tolist()):
                scores_all[i].append(p)
        except Exception:
            pass

    if model2 is not None and processor2 is not None:
        try:
            inputs = processor2(images=pil_images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                probs = calibrate_probability(model2(**inputs).logits, temperature=1.4)
            for i, p in enumerate(probs[:, fake_label_idx2].cpu().numpy().tolist()):
                scores_all[i].append(p)
        except Exception:
            pass

    final_scores = []
    for s in scores_all:
        if not s:
            final_scores.append(0.5)
        elif len(s) == 1:
            final_scores.append(float(s[0]))
        else:
            max_s = max(s)
            min_s = min(s)
            if max_s > 0.70:
                final_scores.append(float(0.75 * max_s + 0.25 * min_s))
            elif max_s < 0.30:
                final_scores.append(float(min_s))
            else:
                final_scores.append(float(0.60 * max_s + 0.40 * min_s))

    return final_scores