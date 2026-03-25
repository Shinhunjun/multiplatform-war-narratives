"""
FastAPI backend for Venezuela-US Narrative Analysis Dashboard.
Serves Reddit analysis data (sentiment, topics, clusters).
Designed for Google Cloud Run deployment.
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import chat, clusters, overview, reports, sentiment, tiktok, topics
from .services.data_service import download_from_gcs

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Download data from GCS on startup (no-op if GCS_BUCKET not set)
    download_from_gcs()
    yield


app = FastAPI(
    title="Venezuela-US Narrative Analysis API",
    description="Multi-platform discourse analysis: Reddit, TikTok, GDELT",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS for React frontend (Vercel)
_extra_origins = os.environ.get("CORS_ORIGINS", "").split(",")
_origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:5175",
    "http://localhost:5176",
    "http://localhost:5177",
] + [o.strip() for o in _extra_origins if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(overview.router)
app.include_router(sentiment.router)
app.include_router(topics.router)
app.include_router(clusters.router)
app.include_router(tiktok.router)
app.include_router(reports.router)
app.include_router(chat.router)


@app.get("/")
def root():
    return {
        "name": "Venezuela-US Narrative Analysis API",
        "version": "0.1.0",
        "docs": "/docs",
        "platforms": ["reddit", "news", "tiktok"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}
