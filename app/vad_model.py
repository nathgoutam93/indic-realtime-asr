import os
import threading

from silero_vad import load_silero_vad


ENABLE_VAD_INFERENCE_LOCK = (
    os.getenv(
        "ENABLE_VAD_INFERENCE_LOCK",
        "1",
    ).lower()
    in {"1", "true", "yes", "on"}
)

vad_model = None
_vad_load_lock = threading.Lock()
_vad_infer_lock = threading.Lock()


def load_vad_model():
    """Initializes the shared Silero VAD model once."""
    global vad_model
    if vad_model is not None:
        return vad_model

    with _vad_load_lock:
        if vad_model is not None:
            return vad_model

        print("Loading Silero VAD model...")
        vad_model = load_silero_vad()
        print("Silero VAD model loaded")
    return vad_model


def get_vad_model():
    if vad_model is None:
        raise RuntimeError("VAD model has not been initialized yet! Call load_vad_model() first.")

    return vad_model


def run_vad(vad_iterator, pcm_tensor):
    if not ENABLE_VAD_INFERENCE_LOCK:
        return vad_iterator(pcm_tensor)

    # VADIterator keeps per-client state, but the underlying model is shared.
    # Serializing model entry avoids concurrent native Torch execution.
    with _vad_infer_lock:
        return vad_iterator(pcm_tensor)
