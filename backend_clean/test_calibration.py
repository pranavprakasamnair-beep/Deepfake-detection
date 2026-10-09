"""
Ground-truth calibration check — bypasses id2label text entirely and tests
what each model actually outputs against images with KNOWN correct answers.
Also prints raw pre-softmax logits to check if the classifier head has any
real signal in it at all.
"""
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_1_PATH = os.path.join(BASE_DIR, "models", "vit_finetuned")
MODEL_2_PATH = os.path.join(BASE_DIR, "models", "vit_faceswap")

device = torch.device("cpu")

def test_model(model_path, model_name, real_img_path, fake_img_path):
    print(f"\n{'='*60}\n{model_name}\n{'='*60}")
    processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForImageClassification.from_pretrained(model_path, local_files_only=True)
    model.to(device).eval()
    print(f"id2label metadata says: {model.config.id2label}")
    print(f"num_labels: {model.config.num_labels}")

    for label_name, img_path in [("KNOWN REAL", real_img_path), ("KNOWN FAKE", fake_img_path)]:
        img = Image.open(img_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = F.softmax(logits, dim=-1)[0]
        print(f"\n  Testing a {label_name} image ({img_path}):")
        print(f"    raw logits: {logits.tolist()}")
        for idx, p in enumerate(probs.tolist()):
            label = model.config.id2label.get(idx, f"idx_{idx}")
            print(f"    index {idx} ('{label}'): {p:.4f}")

if __name__ == "__main__":
    REAL_IMAGE = "test_real.jpg"
    FAKE_IMAGE = "test_fake.jpg"

    # --- File hash check: are the two "different" model weight files identical? ---
    import hashlib
    def file_hash(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()

    hash1 = file_hash(os.path.join(MODEL_1_PATH, "model.safetensors"))
    hash2 = file_hash(os.path.join(MODEL_2_PATH, "model.safetensors"))
    print(f"Model 1 weights SHA256: {hash1}")
    print(f"Model 2 weights SHA256: {hash2}")
    print(f"IDENTICAL FILES: {hash1 == hash2}")

    # --- Config contents ---
    print(f"\n{'='*60}\nModel 1 config.json\n{'='*60}")
    with open(os.path.join(MODEL_1_PATH, "config.json")) as f:
        print(f.read())

    print(f"\n{'='*60}\nModel 2 config.json\n{'='*60}")
    with open(os.path.join(MODEL_2_PATH, "config.json")) as f:
        print(f.read())

    # --- Actual model tests ---
    test_model(MODEL_1_PATH, "Model 1 (dima806 ViT)", REAL_IMAGE, FAKE_IMAGE)
    test_model(MODEL_2_PATH, "Model 2 (Wvolf FaceSwap ViT)", REAL_IMAGE, FAKE_IMAGE)