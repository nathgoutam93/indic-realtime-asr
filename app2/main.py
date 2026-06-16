# main.py
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse

from .transcribe_handler import transcribe_endpoint
from .workers import start_workers
from .model import load_asr_model 
from .vad_model import load_vad_model

INDEX_HTML = Path(__file__).with_name("index.html")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. This runs EXACTLY ONCE when the actual live worker process boots up
    load_asr_model()
    load_vad_model()
    
    # 2. Kick off your asyncio background worker loops
    await start_workers(num_workers=4)
    
    yield
    # Any teardown logic (if needed) goes here when the app closes

# Register the lifespan loop directly into your app
app = FastAPI(lifespan=lifespan)


@app.get("/")
async def index():
    return FileResponse(INDEX_HTML)


@app.post("/transcribe")
async def transcribe_route(
    file: UploadFile = File(...),
    language: str = Form("as"),
):
    return await transcribe_endpoint(
        file=file,
        language=language,
    )
