import os
import shutil
import subprocess
import io
from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
import numpy as np

from utils.video_utils import extract_faces_from_video, extract_face_crop, preprocess_frame_normalization
from ml_models.vit_model import predict_vit, predict_vit_batch
from ml_models.spatial_cnn_expert import full_spatial_analysis
from ml_models.frequency_expert import full_frequency_analysis
from ml_models.attention_module import generate_spatial_attention_mask
from ml_models.spatio_temporal_expert import spatio_temporal_video_analysis
from ml_models.fusion_model import fuse_image_multi_branch, fuse_video_multi_branch
from ml_models.audio_expert import predict_audio

router = APIRouter()

TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)


@router.post("/analyze")
async def auto_route_analysis(file: UploadFile = File(...)):
    """
    Polymorphic Dynamic Orchestration Gateway (Survey Paper Fig 3.1).
    Distributes incoming multimedia data to specialized analytical pipelines.
    """
    filename_lower = file.filename.lower()
    
    if filename_lower.endswith((".png", ".jpg", ".jpeg", ".webp")):
        return await analyze_image(file)
    elif filename_lower.endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a")):
        return await analyze_audio(file)
    elif filename_lower.endswith((".mp4", ".mov", ".avi", ".mkv", ".webm")):
        return await analyze_video(file)
    else:
        raise HTTPException(
            status_code=400, 
            detail="Unsupported media format. Please upload an image, audio, or video asset."
        )


@router.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    """
    Image Deepfake Detection Pipeline:
    Preprocessing -> Dual ViT + Spatial CNN + Frequency (FFT/DCT) + Attention Enhancement -> Feature Fusion
    """
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        raise HTTPException(status_code=400, detail="Must be an image file.")

    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_array = np.array(pil_image)

        # 1. Preprocessing Layer
        face_crop = extract_face_crop(img_array)
        analysis_input = face_crop if face_crop is not None else img_array
        normalized_input = preprocess_frame_normalization(analysis_input)
        analysis_pil = Image.fromarray(normalized_input)

        # 2. Parallel Analytical Streams
        vit_prob = predict_vit(analysis_pil)
        spatial_data = full_spatial_analysis(normalized_input)
        frequency_data = full_frequency_analysis(normalized_input)
        attention_data = generate_spatial_attention_mask(normalized_input)

        # 3. Feature Fusion & Fallback Validation
        fusion_result = fuse_image_multi_branch(
            vit_prob=vit_prob,
            spatial_data=spatial_data,
            frequency_data=frequency_data,
            attention_data=attention_data
        )

        final_score = fusion_result["final_score"]
        is_fake = final_score > 0.50

        # Calibrated Confidence: how confident we are in the verdict
        confidence = round((final_score if is_fake else (1.0 - final_score)) * 100, 2)

        return {
            "filename": file.filename,
            "file_type": "image",
            "label": "FAKE" if is_fake else "REAL",
            "confidence": confidence,
            "probability": final_score,
            "vit_score": fusion_result["vit_prob"],
            "spatial_score": fusion_result["spatial_score"],
            "frequency_score": fusion_result["frequency_score"],
            "attention_score": fusion_result["attention_score"],
            "fallback_triggered": fusion_result["fallback_triggered"],
            "fallback_notes": fusion_result["fallback_notes"],
            "forensic_breakdown": {
                "boundary_seam": spatial_data["boundary_score"],
                "texture_smoothness": spatial_data["texture_score"],
                "fft_ratio": frequency_data["high_low_ratio"],
                "fft_score": frequency_data["fft_score"],
                "dct_score": frequency_data["dct_score"],
                "noise_score": frequency_data["noise_score"],
                "ocular_anomaly": attention_data["ocular_anomaly"],
                "perioral_anomaly": attention_data["perioral_anomaly"]
            },
            "suspicious_frames": []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...)):
    """
    Video Deepfake Detection Pipeline:
    MINTIME Multi-Face Tracking -> Spatio-Temporal Sequence Modeling -> Multi-Modal Fusion
    """
    if not file.filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm")):
        raise HTTPException(status_code=400, detail="Must be a video file.")

    video_path = os.path.join(TEMP_DIR, f"temp_{file.filename}")
    audio_path = os.path.join(TEMP_DIR, f"temp_{file.filename}.wav")

    try:
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extract audio stream if present for multimodal verification
        has_audio = False
        try:
            subprocess.run([
                "ffmpeg", "-i", video_path, "-q:a", "0", "-map", "a",
                audio_path, "-y"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            has_audio = os.path.exists(audio_path) and os.path.getsize(audio_path) > 1024
        except:
            has_audio = False

        # 1. Preprocessing & MINTIME Multi-Face Extraction
        face_arrays = extract_faces_from_video(video_path, max_frames=35)
        
        if not face_arrays:
            return {
                "filename": file.filename,
                "file_type": "video",
                "label": "REAL",
                "confidence": 95.0,
                "probability": 0.05,
                "suspicious_frames": [],
                "forensic_breakdown": {}
            }

        # 2. Spatio-Temporal Modeling & ViT Sequence
        pil_faces = [Image.fromarray(f) for f in face_arrays]
        vit_probs = predict_vit_batch(pil_faces)
        
        temporal_data = spatio_temporal_video_analysis(face_arrays, vit_probs)

        # 3. Spatial, Frequency & Attention Aggregations
        sampled_indices = np.linspace(0, len(face_arrays) - 1, min(8, len(face_arrays)), dtype=int)
        sampled_faces = [face_arrays[i] for i in sampled_indices]

        spatial_scores = [full_spatial_analysis(f)["spatial_fake_score"] for f in sampled_faces]
        freq_scores = [full_frequency_analysis(f)["frequency_fake_score"] for f in sampled_faces]
        attn_scores = [generate_spatial_attention_mask(f)["attention_anomaly_score"] for f in sampled_faces]

        avg_spatial = {"spatial_fake_score": float(np.mean(spatial_scores))}
        avg_freq = {"frequency_fake_score": float(np.mean(freq_scores))}
        avg_attn = {"attention_anomaly_score": float(np.mean(attn_scores))}

        # 4. Multimodal Audio Analysis
        audio_score = None
        if has_audio:
            try:
                audio_score = predict_audio(audio_path)
            except:
                audio_score = None

        # 5. Multi-Branch Video Fusion
        fusion_result = fuse_video_multi_branch(
            temporal_data=temporal_data,
            spatial_data=avg_spatial,
            frequency_data=avg_freq,
            attention_data=avg_attn,
            audio_score=audio_score
        )

        final_score = fusion_result["final_score"]
        is_fake = final_score > 0.50
        confidence = round((final_score if is_fake else (1.0 - final_score)) * 100, 2)

        return {
            "filename": file.filename,
            "file_type": "video",
            "label": "FAKE" if is_fake else "REAL",
            "confidence": confidence,
            "probability": final_score,
            "temporal_score": fusion_result["temporal_score"],
            "spatial_score": fusion_result["spatial_score"],
            "frequency_score": fusion_result["frequency_score"],
            "attention_score": fusion_result["attention_score"],
            "audio_score": fusion_result["audio_score"],
            "suspicious_frames": fusion_result["suspicious_frames"],
            "forensic_breakdown": {
                "motion_glitch": temporal_data["motion_glitch"],
                "seq_mean_vit": temporal_data["seq_mean_prob"],
                "seq_variance": temporal_data["seq_variance"],
                "sampled_spatial": avg_spatial["spatial_fake_score"],
                "sampled_frequency": avg_freq["frequency_fake_score"],
                "sampled_attention": avg_attn["attention_anomaly_score"]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
            except:
                pass
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except:
                pass


@router.post("/analyze-audio")
async def analyze_audio(file: UploadFile = File(...)):
    """
    Audio Deepfake Detection Pipeline (Spectral & Waveform Anomaly Analysis).
    """
    if not file.filename.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a")):
        raise HTTPException(status_code=400, detail="Must be an audio file.")

    audio_path = os.path.join(TEMP_DIR, f"temp_{file.filename}")
    converted_path = os.path.join(TEMP_DIR, f"temp_{file.filename}.wav")

    try:
        with open(audio_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if not file.filename.lower().endswith(".wav"):
            subprocess.run([
                "ffmpeg", "-i", audio_path, "-ar", "16000", "-ac", "1",
                converted_path, "-y"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            analysis_path = converted_path
        else:
            analysis_path = audio_path

        fake_probability = predict_audio(analysis_path)
        is_fake = fake_probability > 0.50
        confidence = round((fake_probability if is_fake else (1.0 - fake_probability)) * 100, 2)

        return {
            "filename": file.filename,
            "file_type": "audio",
            "label": "FAKE" if is_fake else "REAL",
            "confidence": confidence,
            "probability": fake_probability,
            "audio_score": fake_probability,
            "suspicious_frames": []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except:
                pass
        if os.path.exists(converted_path):
            try:
                os.remove(converted_path)
            except:
                pass
