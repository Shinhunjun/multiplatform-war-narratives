"""
FastAPI backend for Venezuela-US Narrative Analysis Dashboard.
Serves Reddit analysis data (sentiment, topics, clusters).
Designed for Google Cloud Run deployment.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import clusters, overview, sentiment, topics

app = FastAPI(
    title="Venezuela-US Narrative Analysis API",
    description="Multi-platform discourse analysis: Reddit, TikTok, GDELT",
    version="0.1.0",
)

# CORS for React frontend (Vercel)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "https://*.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(overview.router)
app.include_router(sentiment.router)
app.include_router(topics.router)
app.include_router(clusters.router)


@app.get("/")
def root():
    return {
        "name": "Venezuela-US Narrative Analysis API",
        "version": "0.1.0",
        "docs": "/docs",
        "platforms": ["reddit"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}
