"""
Loads the ACTUAL models from Hugging Face Hub (bypassing local folders)
to check if local weights are the problem.
"""
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

device = torch.device("cpu")

def test_hub_model(hub_id, real_img_path, fake_img_path):
    print(f"\n{'='*60}\n{hub_id}\n{'='*60}")
    processor = AutoImageProcessor.from_pretrained(hub_id)
    model = AutoModelForImageClassification.from_pretrained(hub_id)
    model.to(device).eval()
    print(f"id2label: {model.config.id2label}")

    for label_name, img_path in [("KNOWN REAL", real_img_path), ("KNOWN FAKE", fake_img_path)]:
        img = Image.open(img_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = F.softmax(logits, dim=-1)[0]
        print(f"\n  {label_name} ({img_path}):")
        print(f"    raw logits: {logits.tolist()}")
        for idx, p in enumerate(probs.tolist()):
            print(f"    index {idx} ('{model.config.id2label[idx]}'): {p:.4f}")

if __name__ == "__main__":
    # Confirm this is the correct hub ID for the model you intended to use
    test_hub_model("dima806/deepfake_vs_real_image_detection", "test_real.jpg", "test_fake.jpg")