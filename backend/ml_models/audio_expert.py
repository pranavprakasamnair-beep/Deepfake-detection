"""
Audio Expert — 107 Acoustic Features + Logistic Regression & Pitch Dynamics.
Extracts MFCCs, Delta, Chroma, Spectral Flatness, Jitter, Shimmer, Pitch Dynamics.
"""
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import joblib
import librosa
import numpy as np

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "models"
MODEL_PATH = MODELS_DIR / "audio_logistic_model.pkl"

_audio_clf = None

def get_audio_model():
    global _audio_clf
    if _audio_clf is None and MODEL_PATH.exists():
        try:
            _audio_clf = joblib.load(MODEL_PATH)
            logger.info("✅ Audio logistic model loaded successfully")
        except Exception as e:
            logger.warning(f"Audio model load error: {e}")
            _audio_clf = None
    return _audio_clf


def extract_audio_from_video(video_path: str) -> Optional[str]:
    """Extract audio track from video file using ffmpeg."""
    try:
        out_path = tempfile.mktemp(suffix=".wav")
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            out_path
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        if result.returncode == 0 and Path(out_path).exists() and Path(out_path).stat().st_size > 1000:
            return out_path
        return None
    except Exception as e:
        logger.warning(f"Audio extraction error: {e}")
        return None


def extract_audio_features(file_path: str) -> np.ndarray:
    """Extract 76-107 acoustic features matching trained forensic pipeline."""
    y, sr = librosa.load(file_path, sr=16000)
    min_samples = sr
    if len(y) < min_samples:
        y = np.pad(y, (0, min_samples - len(y)))

    # MFCC (13) + Delta (13) + Delta2 (13)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfcc_delta = librosa.feature.delta(mfccs)
    mfcc_delta_mean = np.mean(mfcc_delta, axis=1)
    mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
    mfcc_delta2_mean = np.mean(mfcc_delta2, axis=1)

    # Chroma (12)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    # Spectral features
    cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    roll = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    flat = np.mean(librosa.feature.spectral_flatness(y=y))
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = np.mean(contrast, axis=1)
    spec = np.abs(librosa.stft(y))
    flux = np.mean(np.diff(spec, axis=1) ** 2)

    # Prosodic features
    rms = librosa.feature.rms(y=y)[0]
    rms_mean = np.mean(rms)
    rms_std = np.std(rms)

    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        f0_voiced = f0[voiced_flag] if voiced_flag is not None else np.array([])
        pitch_std = np.std(f0_voiced) if len(f0_voiced) > 0 else 0.0
        voiced_ratio = np.sum(voiced_flag) / len(voiced_flag) if voiced_flag is not None and len(voiced_flag) > 0 else 0.5
    except Exception:
        pitch_std = 0.0
        voiced_ratio = 0.5

    zcr_mean = np.mean(librosa.feature.zero_crossing_rate(y))

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_mean = np.mean(mel)
    mel_std = np.std(mel)
    mel_skew = float(np.mean(((mel - mel_mean) / (mel_std + 1e-8)) ** 3))
    mel_band_std = np.mean(np.std(mel, axis=1))

    frame_length = 512
    hop_length = 128
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=0))
    shimmer = np.mean(np.abs(np.diff(frame_rms)) / (frame_rms[:-1] + 1e-8))
    frame_zcr = np.mean(np.abs(np.diff(np.sign(frames), axis=0)), axis=0) / frame_length
    jitter = np.std(frame_zcr)
    energy_entropy = -np.sum(
        (frame_rms / (np.sum(frame_rms) + 1e-8)) *
        np.log2(frame_rms / (np.sum(frame_rms) + 1e-8) + 1e-8)
    )
    silence_ratio = np.sum(frame_rms < 0.01) / (len(frame_rms) + 1e-8)

    features = np.hstack((
        mfccs_mean,
        mfcc_delta_mean,
        mfcc_delta2_mean,
        chroma_mean,
        cent,
        bw,
        roll,
        flat,
        contrast_mean,
        flux,
        rms_mean,
        rms_std,
        pitch_std,
        voiced_ratio,
        zcr_mean,
        mel_mean,
        mel_std,
        mel_skew,
        mel_band_std,
        shimmer,
        jitter,
        energy_entropy,
        silence_ratio,
    ))
    return features


def analyze_audio(file_path: str, registry=None) -> dict:
    """Analyze audio and return structured forensic score."""
    clf = get_audio_model()
    try:
        features = extract_audio_features(file_path)
        # pitch_std is at index 65 (13+13+13+12+1+1+1+1+7+1+1+1 = 65)
        pitch_std = features[65] if len(features) > 65 else 0.0

        if clf is not None:
            try:
                features_reshaped = features.reshape(1, -1)
                prob = float(clf.predict_proba(features_reshaped)[0][1])
            except Exception as e:
                logger.warning(f"Classifier inference failed: {e}")
                prob = 0.5
        else:
            prob = 0.5

        # Heuristic acoustic rules for synthetic robotic voices vs natural human speech
        if pitch_std < 1.0:
            prob = max(prob, 0.88)
        elif pitch_std > 8.0:
            prob = min(prob, 0.12)

        verdict = "FAKE" if prob > 0.50 else "REAL"
        return {
            "score": round(float(np.clip(prob, 0.0, 1.0)), 4),
            "verdict": verdict,
            "details": {
                "pitch_std": round(float(pitch_std), 4),
                "model_used": "LogisticRegression + Acoustic Dynamics" if clf is not None else "Acoustic Dynamics Heuristic",
            }
        }
    except Exception as e:
        logger.error(f"Audio analysis error: {e}")
        return {"score": 0.5, "verdict": "UNKNOWN", "details": {"error": str(e)}}
