# main.py
import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse

from .websocket_handler import websocket_endpoint
from .workers import start_workers
from .model import load_asr_model  # Import our loading orchestrator
from .vad_model import load_vad_model

INDEX_HTML = Path(__file__).with_name("index.html")
TEST_HTML = Path(__file__).with_name("stress-test.html")
ASR_WORKERS = int(
    os.getenv(
        "ASR_WORKERS",
        "1",
    )
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start model loading in the background so startup does not block on
    # large Hugging Face downloads or first-time cache population.
    asyncio.create_task(asyncio.to_thread(load_asr_model))
    asyncio.create_task(asyncio.to_thread(load_vad_model))

    # Kick off asyncio background worker loops.
    await start_workers(num_workers=ASR_WORKERS)

    yield
    # Any teardown logic (if needed) goes here when the app closes

# Register the lifespan loop directly into your app
app = FastAPI(lifespan=lifespan)

@app.get("/ping")
async def health_check():
    return {"status": "healthy"}

@app.get("/")
async def index():
    return FileResponse(INDEX_HTML)

@app.get("/stress-test")
async def index():
    return FileResponse(TEST_HTML)

@app.websocket("/ws")
async def websocket_route(websocket: WebSocket):
    await websocket_endpoint(websocket)
