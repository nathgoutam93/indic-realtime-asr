# main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket
import uvicorn

from .websocket_handler import websocket_endpoint
from .workers import start_workers
from .model import load_asr_model  # Import our loading orchestrator

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. This runs EXACTLY ONCE when the actual live worker process boots up
    load_asr_model()
    
    # 2. Kick off your asyncio background worker loops
    await start_workers(num_workers=2)
    
    yield
    # Any teardown logic (if needed) goes here when the app closes

# Register the lifespan loop directly into your app
app = FastAPI(lifespan=lifespan)


@app.websocket("/ws")
async def websocket_route(websocket: WebSocket):
    await websocket_endpoint(websocket)
