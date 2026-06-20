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
HF_CACHE = os.getenv("HF_CACHE")
REQUIRE_CUDA = (
    os.getenv(
        "REQUIRE_CUDA",
        "1",
    ).lower()
    in {"1", "true", "yes", "on"}
)
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

def _assert_cuda_available():
    if REQUIRE_CUDA and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required but not available. Check NVIDIA drivers, CUDA runtime, "
            "and that PyTorch was installed with CUDA support. Set REQUIRE_CUDA=0 "
            "only if CPU fallback is intentional."
        )


def _assert_model_on_device(loaded_model):
    devices = {parameter.device.type for parameter in loaded_model.parameters()}
    devices.update(buffer.device.type for buffer in loaded_model.buffers())

    if devices and DEVICE.type not in devices:
        raise RuntimeError(f"ASR model was not moved to {DEVICE}; found devices: {sorted(devices)}")


def _move_to_device(value):
    if torch.is_tensor(value):
        return value.to(DEVICE, non_blocking=True)
    return value


def _describe_model_device(loaded_model):
    for parameter in loaded_model.parameters():
        return parameter.device

    for buffer in loaded_model.buffers():
        return buffer.device

    return DEVICE


def load_asr_model():
    global model
    if model is not None:
        return model

    with _model_load_lock:
        if model is not None:
            return model

        _assert_cuda_available()
        print(f"Loading ASR model on {DEVICE}...")
        if DEVICE.type == "cuda":
            print(f"CUDA device: {torch.cuda.get_device_name(DEVICE)}")

        model = AutoModel.from_pretrained(
            "ai4bharat/indic-conformer-600m-multilingual",
            trust_remote_code=True,
            token=HF_TOKEN,
            cache_dir=HF_CACHE
        )

        model = model.to(DEVICE)
        _assert_model_on_device(model)
        model.eval()

        print(f"ASR model loaded on {_describe_model_device(model)}")

    return model

def transcribe(wav, language="as"):
    global model
    if model is None:
        raise RuntimeError("Model has not been initialized yet! Call load_asr_model() first.")

    wav = _move_to_device(wav)

    if not ENABLE_ASR_INFERENCE_LOCK:
        with torch.inference_mode():
            return model(wav, language, "rnnt")

    # The remote-code model mixes TorchScript and ONNXRuntime components.
    # Serializing entry avoids concurrent native execution crashes under load.
    with _model_infer_lock:
        with torch.inference_mode():
            return model(wav, language, "rnnt")
