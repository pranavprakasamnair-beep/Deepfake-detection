main_code = '''from fastapi import FastAPI 
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router

app = FastAPI(
    title="AI-Driven Deepfake Image & Video Detection API",
    description="Multi-Branch Deepfake Detection Framework (MINTIME spatio-temporal tracking, dual ViT, FFT/DCT frequency analysis, and attention enhancement)",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_headers=["*"],
    allow_methods=["*"]
)

@app.get("/")
async def root():
    return {
        "status": "online", 
        "system": "AI-Driven Deepfake Image and Video Detection Framework",
        "reference_base": "MINTIME Multi-Identity Size-Invariant Video Deepfake Detection",
        "active_modules": [
            "MINTIME Multi-Identity Size-Invariant Tracking & Preprocessing",
            "Dual Vision Transformer (dima806 + Wvolf FaceForensics++)",
            "CNN Spatial Feature Extraction (Boundary Seams, Texture Smoothness, Lighting)",
            "Frequency Analysis Module (2D FFT Radial Spectrum + DCT + Shot Noise)",
            "Attention-Based Feature Enhancement (Ocular & Perioral Weighting)",
            "Spatio-Temporal Sequence Modeling & Glitch Dynamics",
            "Multi-Modal Feature Fusion & Rule-Based Fallback Engine",
            "Audio Spectral Waveform Forensics"
        ]
    }

app.include_router(router)
'''

with open("main.py", "w", encoding="utf-8") as f:
    f.write(main_code)
print("Wrote main.py")
