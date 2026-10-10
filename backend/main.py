"""
Deepfake Detection v2.0 — Main FastAPI Application
"""
import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from ml_models.model_registry import ModelRegistry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models at startup."""
    logger.info("=== Deepfake Detection v2.0 starting up ===")
    registry = ModelRegistry.get_instance()
    registry.load_all()
    logger.info("=== All models loaded. Ready. ===")
    yield
    logger.info("=== Shutting down ===")


app = FastAPI(
    title="Deepfake Detection API v2.0",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from pathlib import Path
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app.include_router(router, prefix="")
app.include_router(router, prefix="/api")


@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0"}


media_dir = Path(__file__).parent / "database" / "media"
media_dir.mkdir(parents=True, exist_ok=True)
app.mount("/media", StaticFiles(directory=str(media_dir)), name="media")

# Mount static production build for single-port unified deployment
dist_path = Path(__file__).parent.parent / "frontend" / "dist"
if not dist_path.exists():
    dist_path = Path(__file__).parent / "static"

if dist_path.exists() and (dist_path / "index.html").exists():
    assets_dir = dist_path / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/")
    def serve_index():
        return FileResponse(str(dist_path / "index.html"))
else:
    @app.get("/")
    def root():
        return {"status": "ok", "service": "Deepfake Detection Engine v2.0", "version": "2.0.0"}

