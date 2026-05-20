# model.py
import os
import torch
from dotenv import load_dotenv
from transformers import AutoModel

load_dotenv()

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

HF_TOKEN = os.getenv("HF_TOKEN")

# This starts as None so importing the file is instant and lightweight
model = None

def load_asr_model():
    """Explicitly initializes the global model once called."""
    global model
    if model is not None:
        return model

    print("Loading ASR model...")
    model = AutoModel.from_pretrained(
        "ai4bharat/indic-conformer-600m-multilingual",
        trust_remote_code=True,
        token=HF_TOKEN,
        cache_dir="./hf_cache"
    )
    model.eval()
    print("ASR model loaded")
    return model


def transcribe(wav, language="as"):
    global model
    if model is None:
        raise RuntimeError("Model has not been initialized yet! Call load_asr_model() first.")
        
    with torch.inference_mode():
        return model(wav, language, "rnnt")
