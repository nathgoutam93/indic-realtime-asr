# model.py
import os
import threading
import torch
from dotenv import load_dotenv
from transformers import AutoModel

load_dotenv()

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

HF_TOKEN = os.getenv("HF_TOKEN")
ENABLE_ASR_INFERENCE_LOCK = (
    os.getenv(
        "ENABLE_ASR_INFERENCE_LOCK",
        "1",
    ).lower()
    in {"1", "true", "yes", "on"}
)

# This starts as None so importing the file is instant and lightweight
model = None
_model_load_lock = threading.Lock()
_model_infer_lock = threading.Lock()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_asr_model():
    global model
    if model is not None:
        return model

    with _model_load_lock:
        if model is not None:
            return model

        print(f"Loading ASR model on {DEVICE}...")

        model = AutoModel.from_pretrained(
            "ai4bharat/indic-conformer-600m-multilingual",
            trust_remote_code=True,
            token=HF_TOKEN,
            cache_dir="./hf_cache"
        )

        model = model.to(DEVICE)
        model.eval()

        print(f"ASR model loaded on {next(model.parameters()).device}")

    return model

def transcribe(wav, language="as"):
    global model
    if model is None:
        raise RuntimeError("Model has not been initialized yet! Call load_asr_model() first.")

    if not ENABLE_ASR_INFERENCE_LOCK:
        with torch.inference_mode():
            return model(wav, language, "rnnt")

    # The remote-code model mixes TorchScript and ONNXRuntime components.
    # Serializing entry avoids concurrent native execution crashes under load.
    with _model_infer_lock:
        with torch.inference_mode():
            return model(wav, language, "rnnt") here how to make sure that model loaded on cuda 
