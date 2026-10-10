"""
Model Registry — Singleton holding all vision, audio, and custom neural networks.
Loads Dual Vision Transformers (dima806 + Wvolf FaceSwap), trained Audio model,
and custom EfficientNet-B4 (from Google Colab training).
"""
import logging
import os
import pickle
from pathlib import Path

import joblib
import numpy as np
import timm
import torch
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelRegistry:
    _instance = None

    def __init__(self):
        # Dual Vision Transformers
        self.vit_model1 = None          # dima806 (GAN / AI generation)
        self.vit_processor1 = None
        self.vit_model2 = None          # Wvolf (FaceSwap forensics)
        self.vit_processor2 = None

        # Custom-trained model from Google Colab
        self.custom_efficientnet = None
        self.efficientnet_transform = None
        self.has_custom = False

        # Audio Forensic Model
        self.audio_model = None

        self._loaded = False

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_all(self):
        if self._loaded:
            return
        self._load_vit_models()
        self._load_custom_efficientnet()
        self._load_learned_flips()
        self._load_audio_model()
        self._loaded = True
        logger.info("=== Deepfake Detection Engine v2.0 Ready ===")

    # ── Dual Vision Transformers ──────────────────────────────────────────────
    def _load_vit_models(self):
        # Model 1: vit_finetuned (dima806)
        path1 = MODELS_DIR / "vit_finetuned"
        if not path1.exists():
            path1 = MODELS_DIR / "vit_deepfake"

        try:
            if path1.exists():
                logger.info(f"Loading ViT Model 1 (Synthesis) from {path1}")
                self.vit_processor1 = AutoImageProcessor.from_pretrained(str(path1))
                self.vit_model1 = AutoModelForImageClassification.from_pretrained(str(path1))
            else:
                logger.info("Downloading ViT Model 1 from HuggingFace...")
                repo = "dima806/deepfake_vs_real_image_detection"
                self.vit_processor1 = AutoImageProcessor.from_pretrained(repo)
                self.vit_model1 = AutoModelForImageClassification.from_pretrained(repo)
                self.vit_processor1.save_pretrained(str(path1))
                self.vit_model1.save_pretrained(str(path1))
            self.vit_model1.eval()
            self.vit_model1.to(DEVICE)
            logger.info("✅ ViT Model 1 (dima806) loaded")
        except Exception as e:
            logger.error(f"Failed to load ViT Model 1: {e}")

        # Model 2: vit_faceswap (Wvolf)
        path2 = MODELS_DIR / "vit_faceswap"
        try:
            if path2.exists():
                logger.info(f"Loading ViT Model 2 (FaceSwap) from {path2}")
                self.vit_processor2 = AutoImageProcessor.from_pretrained(str(path2))
                self.vit_model2 = AutoModelForImageClassification.from_pretrained(str(path2))
            else:
                logger.info("Downloading ViT Model 2 (Wvolf) from HuggingFace...")
                repo = "Wvolf/ViT_Face_Deepfake_Detection"
                self.vit_processor2 = AutoImageProcessor.from_pretrained(repo)
                self.vit_model2 = AutoModelForImageClassification.from_pretrained(repo)
                self.vit_processor2.save_pretrained(str(path2))
                self.vit_model2.save_pretrained(str(path2))
            self.vit_model2.eval()
            self.vit_model2.to(DEVICE)
            logger.info("✅ ViT Model 2 (Wvolf FaceSwap) loaded")
        except Exception as e:
            logger.warning(f"ViT Model 2 not available: {e} — continuing with Model 1")

    # ── Learned polarity (written by `python diagnose.py`, many labelled images) ──
    def _load_learned_flips(self):
        """
        Per-model direction is NOT guessed from one or two images any more. It is
        learned by diagnose.py from your labelled real/fake folders and stored in
        calibration/decision.json. With no file, nothing is flipped.
        """
        import json
        self.vit1_flip = False
        self.vit2_flip = False
        self.effnet_fake_idx = 0  # Colab script trains 0 = FAKE, 1 = REAL
        p = Path(__file__).parent.parent / "calibration" / "decision.json"
        try:
            flips = json.loads(p.read_text()).get("flips", {})
            self.vit1_flip = bool(flips.get("vit1", False))
            self.vit2_flip = bool(flips.get("vit2", False))
            self.effnet_fake_idx = 1 if flips.get("effnet", False) else 0
            logger.info(f"Learned flips loaded: vit1={self.vit1_flip} vit2={self.vit2_flip} effnet_idx={self.effnet_fake_idx}")
        except Exception:
            logger.info("No learned flips (run diagnose.py to learn them) - using model labels as-is")

    # ── Custom Trained EfficientNet (from Google Colab) ──────────────────────
    def _load_custom_efficientnet(self):
        model_path = MODELS_DIR / "efficientnet_deepfake.pt"
        if model_path.exists():
            try:
                logger.info(f"Loading custom EfficientNet-B4 from {model_path}...")
                model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=2)
                state_dict = torch.load(model_path, map_location=DEVICE)
                if "model_state_dict" in state_dict:
                    state_dict = state_dict["model_state_dict"]
                model.load_state_dict(state_dict)
                model.eval()
                model.to(DEVICE)
                self.custom_efficientnet = model
                self.has_custom = True
                logger.info("✅ Custom EfficientNet-B4 loaded (Google Colab trained)")
            except Exception as e:
                logger.warning(f"Failed to load custom EfficientNet: {e}")
                self.custom_efficientnet = None
                self.has_custom = False
        else:
            logger.info("ℹ️  Custom EfficientNet not present yet. Ready to accept 'models/efficientnet_deepfake.pt' from Colab.")
            self.custom_efficientnet = None
            self.has_custom = False

        self.efficientnet_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    # ── Audio Logistic Forensic Model ─────────────────────────────────────────
    def _load_audio_model(self):
        mpath = MODELS_DIR / "audio_logistic_model.pkl"
        if mpath.exists():
            try:
                self.audio_model = joblib.load(mpath)
                logger.info("✅ Audio logistic model loaded")
            except Exception as e:
                logger.warning(f"Audio model loading error: {e}")
                self.audio_model = None
        else:
            self.audio_model = None