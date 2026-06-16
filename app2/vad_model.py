from silero_vad import load_silero_vad


vad_model = None


def load_vad_model():
    """Initializes the shared Silero VAD model once."""
    global vad_model
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

