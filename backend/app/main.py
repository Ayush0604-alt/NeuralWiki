from dotenv import load_dotenv

load_dotenv()

import logging
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.upload import router as upload_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)

app = FastAPI(
    title="NeuralWiki API",
    description="RAG-powered document knowledge API",
    version="2.0.0",
)

# CORS configuration - support local dev and deployed environments
cors_origins = [
    "http://localhost:5173",   # Vite dev server
    "http://127.0.0.1:5173",
    "http://localhost:5174",   # Vite dev server (alternate port)
    "http://127.0.0.1:5174",
    "http://localhost:4173",   # Vite preview
    "http://127.0.0.1:4173",
]

# Add frontend URL from environment (set on Render)
frontend_url = os.getenv("FRONTEND_URL")
if frontend_url:
    cors_origins.append(frontend_url)
    # Also allow with/without trailing slash variants
    cors_origins.append(frontend_url.rstrip("/"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router)


@app.get("/")
def home():
    return {"message": "NeuralWiki Backend Running", "version": "2.0.0"}