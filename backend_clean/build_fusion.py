# 5. FEATURE FUSION & DECISION MODULE WITH RULE-BASED FALLBACK
fusion_code = '''import numpy as np

def apply_rule_based_fallback(
    preliminary_score: float,
    spatial_data: dict,
    frequency_data: dict,
    attention_data: dict,
    vit_prob: float
) -> tuple[float, list[str]]:
    """
    Fallback and Validation Mechanism (Survey Paper Section III).
    Applies deterministic forensic rules in edge cases where deep neural network
    predictions are borderline, highly discordant, or vulnerable to adversarial smoothing.
    """
    fallback_flags = []
    adjusted_score = preliminary_score

    bound_score = spatial_data.get("boundary_score", 0.0)
    fft_score = frequency_data.get("fft_score", 0.0)
    noise_std = frequency_data.get("noise_std", 2.5)
    ocular_anomaly = attention_data.get("ocular_anomaly", 0.0)

    # Rule 1: Clear boundary seam detected (strong geometric face-swap artifact)
    if bound_score > 0.65:
        adjusted_score = max(adjusted_score, 0.75)
        fallback_flags.append("Definite boundary seam distortion detected")

    # Rule 2: Unnatural lack of camera sensor shot noise (over-smoothed GAN/diffusion)
    if noise_std < 0.60:
        adjusted_score = max(adjusted_score, 0.68)
        fallback_flags.append("Abnormal sensor noise absence (Synthetic smoothing signature)")

    # Rule 3: Severe frequency spectrum imbalance
    if fft_score > 0.80:
        adjusted_score = max(adjusted_score, 0.72)
        fallback_flags.append("Abnormal high-frequency power spectrum drop-off")

    # Rule 4: Deep neural model confident REAL but multiple physical forensics flag anomalies
    if vit_prob < 0.20 and (bound_score > 0.40 or fft_score > 0.60 or ocular_anomaly > 0.70):
        adjusted_score = max(adjusted_score, 0.52)
        fallback_flags.append("ViT-Forensic discordance override triggered")

    return float(np.clip(adjusted_score, 0.0, 1.0)), fallback_flags


def fuse_image_multi_branch(
    vit_prob: float,
    spatial_data: dict,
    frequency_data: dict,
    attention_data: dict
) -> dict:
    """
    Comprehensive Multi-Branch Image Fusion (Section III).
    Branches:
    1. Vision Transformer (ViT) representation (0.45)
    2. Frequency Domain (FFT + DCT + Noise) (0.25)
    3. CNN Spatial Features (Texture + Seams + Lighting) (0.18)
    4. Attention-Enhanced Critical Regions (0.12)
    """
    spatial_score = spatial_data.get("spatial_fake_score", 0.5)
    freq_score = frequency_data.get("frequency_fake_score", 0.5)
    attn_score = attention_data.get("attention_anomaly_score", 0.5)

    base_fused = (
        0.45 * vit_prob +
        0.25 * freq_score +
        0.18 * spatial_score +
        0.12 * attn_score
    )

    final_score, fallback_notes = apply_rule_based_fallback(
        base_fused, spatial_data, frequency_data, attention_data, vit_prob
    )

    return {
        "final_score": round(final_score, 4),
        "vit_prob": round(vit_prob, 4),
        "spatial_score": round(spatial_score, 4),
        "frequency_score": round(freq_score, 4),
        "attention_score": round(attn_score, 4),
        "fallback_triggered": len(fallback_notes) > 0,
        "fallback_notes": fallback_notes
    }


def fuse_video_multi_branch(
    temporal_data: dict,
    spatial_data: dict,
    frequency_data: dict,
    attention_data: dict,
    audio_score: float = None
) -> dict:
    """
    Comprehensive Multi-Branch Video Fusion (Section III & MINTIME).
    Branches:
    1. Spatio-Temporal Inter-frame Glitch & ViT Dynamics (0.50)
    2. Frequency Domain (FFT + DCT + Noise) (0.20)
    3. Spatial Artifacts & Seams (0.18)
    4. Attention-Enhanced Anomaly (0.12)
    + Optional Multimodal Audio Integration (30% audio, 70% video if audio exists)
    """
    temp_score = temporal_data.get("temporal_score", 0.5)
    spatial_score = spatial_data.get("spatial_fake_score", 0.5)
    freq_score = frequency_data.get("frequency_fake_score", 0.5)
    attn_score = attention_data.get("attention_anomaly_score", 0.5)

    video_visual_score = (
        0.50 * temp_score +
        0.20 * freq_score +
        0.18 * spatial_score +
        0.12 * attn_score
    )

    if audio_score is not None:
        overall_score = (0.70 * video_visual_score) + (0.30 * audio_score)
    else:
        overall_score = video_visual_score

    overall_score = float(np.clip(overall_score, 0.0, 1.0))

    return {
        "final_score": round(overall_score, 4),
        "visual_score": round(video_visual_score, 4),
        "temporal_score": round(temp_score, 4),
        "spatial_score": round(spatial_score, 4),
        "frequency_score": round(freq_score, 4),
        "attention_score": round(attn_score, 4),
        "audio_score": round(audio_score, 4) if audio_score is not None else None,
        "suspicious_frames": temporal_data.get("suspicious_frames", [])
    }
'''

with open("ml_models/fusion_model.py", "w", encoding="utf-8") as f:
    f.write(fusion_code)
print("Wrote ml_models/fusion_model.py")
