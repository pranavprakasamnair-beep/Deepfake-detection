# =============================================================================
# STAGE 1 — Build Frontend (React + Vite + Tailwind)
# =============================================================================
FROM node:20-slim AS frontend-builder
WORKDIR /build

COPY frontend/package*.json ./
RUN npm install --legacy-peer-deps

COPY frontend/ ./
RUN npm run build

# =============================================================================
# STAGE 2 — Production Python Backend with AI Engine & Static Frontend
# =============================================================================
FROM python:3.11-slim AS production

# Prevent Python from writing .pyc files and enable live stdout flushing
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Install essential system libraries required by OpenCV, Librosa, and FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Install CPU-optimized PyTorch (drastically reduces image size from 8GB to ~1.8GB)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu

# 2. Install application dependencies
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy backend source code & pre-trained models
COPY backend/ ./

# 4. Copy production frontend build from Stage 1 into backend static directory
COPY --from=frontend-builder /build/dist /app/static

# 5. Create storage directories for sqlite database & temp processing
RUN mkdir -p /app/database /app/models

EXPOSE 8000

# Run FastAPI engine via Uvicorn (binds to PORT environment variable for cloud hosts)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
