"""SPANDANA FastAPI application.

Run from the backend/ folder:

    python -m uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import CORS_ORIGINS, VERSION
from app.services import model_status

app = FastAPI(
    title="SPANDANA — AI Cardiac Intelligence API",
    description="Research screening API for LVH: Sokolow-Lyon / Cornell "
                "criteria + real Perceptron and CNN inference on 12-lead ECGs.",
    version=VERSION,
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
def report_status() -> None:
    st = model_status.artifact_status()
    ready = [m for m, info in st["models"].items() if info["available"]]
    print(f"[SPANDANA] v{VERSION} — models with artefacts: "
          f"{ready or 'NONE (screening will answer 503)'}")
    if not ready:
        print("[SPANDANA] train them with: python training/train_perceptron.py "
              "&& python training/train_cnn.py && python training/evaluate_cnn.py")
