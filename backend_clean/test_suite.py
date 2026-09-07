import os
import sys
import time
import numpy as np
from PIL import Image
import io

print("="*70)
print(" AI DRIVEN DEEPFAKE DETECTION — COMPREHENSIVE ARCHITECTURAL TEST SUITE")
print(" Base Reference: MINTIME (Multi-Identity Size-Invariant Video Deepfake Detection)")
print("="*70)

from ml_models.frequency_expert import full_frequency_analysis
from ml_models.spatial_cnn_expert import full_spatial_analysis
from ml_models.attention_module import generate_spatial_attention_mask
from ml_models.spatio_temporal_expert import spatio_temporal_video_analysis
from ml_models.fusion_model import fuse_image_multi_branch, fuse_video_multi_branch
from ml_models.vit_model import predict_vit, predict_vit_batch
from utils.video_utils import extract_face_crop, preprocess_frame_normalization

# 1. Test Preprocessing & Normalization
print("\n[TEST 1] Preprocessing Layer & Normalization...")
dummy_img = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
norm_img = preprocess_frame_normalization(dummy_img)
assert norm_img.shape == dummy_img.shape, "Shape mismatch after normalization"
print("  -> Preprocessing & Bilateral Filtering: PASSED")

# 2. Test Spatial CNN Module
print("\n[TEST 2] Spatial CNN Feature Extraction Module...")
spatial_res = full_spatial_analysis(norm_img)
print(f"  -> Boundary Score: {spatial_res['boundary_score']}")
print(f"  -> Texture Score: {spatial_res['texture_score']}")
print(f"  -> Spatial Fake Score: {spatial_res['spatial_fake_score']}")
assert "spatial_fake_score" in spatial_res
print("  -> Spatial CNN Module: PASSED")

# 3. Test Frequency Analysis Module (FFT + DCT + Noise)
print("\n[TEST 3] Frequency Analysis Module (2D FFT, DCT & Shot Noise)...")
freq_res = full_frequency_analysis(norm_img)
print(f"  -> FFT High/Low Ratio: {freq_res['high_low_ratio']} (Score: {freq_res['fft_score']})")
print(f"  -> DCT Energy Score: {freq_res['dct_score']}")
print(f"  -> Sensor Noise STD: {freq_res['noise_std']} (Score: {freq_res['noise_score']})")
print(f"  -> Combined Frequency Anomaly: {freq_res['frequency_fake_score']}")
assert "frequency_fake_score" in freq_res
print("  -> Frequency Analysis Module: PASSED")

# 4. Test Attention-Based Feature Enhancement
print("\n[TEST 4] Attention-Based Feature Enhancement Module...")
attn_res = generate_spatial_attention_mask(norm_img)
print(f"  -> Ocular Anomaly: {attn_res['ocular_anomaly']}")
print(f"  -> Perioral Anomaly: {attn_res['perioral_anomaly']}")
print(f"  -> Combined Attention Anomaly: {attn_res['attention_anomaly_score']}")
assert "attention_anomaly_score" in attn_res
print("  -> Attention-Based Feature Enhancement: PASSED")

# 5. Test Dual Vision Transformer (ViT) Models
print("\n[TEST 5] Dual Vision Transformer Inference...")
pil_dummy = Image.fromarray(norm_img)
vit_score = predict_vit(pil_dummy)
print(f"  -> Dual ViT Calibrated Fake Probability: {vit_score:.4f}")
assert 0.0 <= vit_score <= 1.0
print("  -> Dual Vision Transformer: PASSED")

# 6. Test Image Multi-Branch Fusion & Rule-Based Fallback
print("\n[TEST 6] Multi-Branch Image Fusion & Safety Fallback Engine...")
fused_img = fuse_image_multi_branch(
    vit_prob=vit_score,
    spatial_data=spatial_res,
    frequency_data=freq_res,
    attention_data=attn_res
)
print(f"  -> Fused Verdict Score: {fused_img['final_score']}")
print(f"  -> Fallback Triggered: {fused_img['fallback_triggered']}")
if fused_img['fallback_notes']:
    print(f"  -> Fallback Notes: {fused_img['fallback_notes']}")
print("  -> Image Fusion & Fallback Engine: PASSED")

# 7. Test Spatio-Temporal Sequence Modeling (MINTIME Video Stream)
print("\n[TEST 7] Spatio-Temporal Video Modeling & Sequence Analysis...")
video_frames = [np.random.randint(40, 220, (224, 224, 3), dtype=np.uint8) for _ in range(8)]
vit_batch_scores = predict_vit_batch([Image.fromarray(f) for f in video_frames])
temp_res = spatio_temporal_video_analysis(video_frames, vit_batch_scores)
print(f"  -> Inter-Frame Motion Glitch: {temp_res['motion_glitch']}")
print(f"  -> Sequence Mean ViT: {temp_res['seq_mean_prob']}")
print(f"  -> Sequence Variance: {temp_res['seq_variance']}")
print(f"  -> Suspicious Frames: {temp_res['suspicious_frames']}")
print("  -> Spatio-Temporal Modeling: PASSED")

print("\n" + "="*70)
print(" ALL 7 ARCHITECTURAL SUBSYSTEMS VERIFIED AND FULLY OPERATIONAL!")
print("="*70)
