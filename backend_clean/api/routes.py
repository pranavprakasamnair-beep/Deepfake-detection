"""
Data-driven calibration. Run ONCE from the backend folder:

    python diagnose.py <folder_of_REAL_images> <folder_of_FAKE_images>

It scores every image with the full pipeline, prints a table, then learns from YOUR
labelled images (a) whether the final score is inverted and (b) the best REAL/FAKE
cut-off, and saves them to calibration/decision.json. fusion.py reads that file.
Restart the backend afterwards. Delete decision.json to go back to defaults.
"""
import json
import sys
from pathlib import Path

import numpy as np

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
OUT = Path(__file__).parent / "calibration" / "decision.json"


def auc(real, fake):
    """P(random fake scores higher than random real); 0.5 = no signal, <0.5 = inverted."""
    wins = sum((f > r) + 0.5 * (f == r) for f in fake for r in real)
    return wins / (len(real) * len(fake))


def fit_decision(real_scores, fake_scores):
    real = np.asarray(real_scores, float)
    fake = np.asarray(fake_scores, float)
    a = auc(real, fake)
    invert = bool(a < 0.5)
    if invert:
        real, fake = 1.0 - real, 1.0 - fake
    allv = np.unique(np.concatenate([real, fake]))
    cands = (allv[:-1] + allv[1:]) / 2.0 if len(allv) > 1 else np.array([0.5])
    best_t, best_acc, best_gap = 0.5, -1.0, 9.0
    mid = (real.mean() + fake.mean()) / 2.0
    for t in cands:
        acc = 0.5 * ((real <= t).mean() + (fake > t).mean())  # balanced accuracy
        gap = abs(t - mid)
        if acc > best_acc + 1e-9 or (abs(acc - best_acc) <= 1e-9 and gap < best_gap):
            best_t, best_acc, best_gap = float(t), float(acc), gap
    best_t = float(np.clip(best_t, 0.05, 0.95))
    return {"invert": invert, "threshold": round(best_t, 4),
            "auc": round(float(max(a, 1 - a)), 4), "balanced_accuracy": round(best_acc, 4)}


def score_folder(folder, registry):
    from PIL import Image
    from ml_models.image_expert import analyze_image
    from ml_models.frequency_expert import analyze_frequency
    rows = []
    for p in sorted(Path(folder).iterdir()):
        if p.suffix.lower() not in EXTS:
            continue
        try:
            img = Image.open(p).convert("RGB")
            ir = analyze_image(img, registry)
            fr = analyze_frequency(img)
            w_i, w_f = 0.88, 0.12
            s = (ir["score"] * w_i + fr["score"] * w_f) / (w_i + w_f)
            rows.append((p.name, ir["score"], fr["score"], s, ir["details"].get("model_probs", {})))
        except Exception as e:
            print(f"  skipped {p.name}: {e}")
    return rows


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    from ml_models.model_registry import ModelRegistry
    reg = ModelRegistry.get_instance()
    reg.load_all()
    # Score with raw (un-inverted) pipeline so the fit sees true behaviour.
    if OUT.exists():
        OUT.unlink()
    real_rows = score_folder(sys.argv[1], reg)
    fake_rows = score_folder(sys.argv[2], reg)
    if len(real_rows) < 2 or len(fake_rows) < 2:
        print("Need at least 2 real and 2 fake images (5+ each is much better).")
        sys.exit(1)
    for title, rows in (("REAL", real_rows), ("FAKE", fake_rows)):
        print(f"\n--- {title} images (name | image_expert | frequency | combined) ---")
        for n, i, f, s, mp in rows:
            print(f"{n[:40]:40s} {i:6.3f} {f:6.3f} {s:6.3f}   models={mp}")
    res = fit_decision([r[3] for r in real_rows], [r[3] for r in fake_rows])
    print("\nLearned:", res)
    if res["auc"] < 0.6:
        print("WARNING: scores barely separate your real and fake images (AUC < 0.6).")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(f"Saved {OUT}. Restart the backend.")


if __name__ == "__main__":
    main()