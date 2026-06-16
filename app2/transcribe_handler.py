from __future__ import annotations

import asyncio
import audioop
import io
import wave
import uuid

import torch
import numpy as np
from fastapi import File, Form, HTTPException, UploadFile

from .connection_manager import manager
from .jobs import TranscriptionJob
from .queue_manager import transcription_queue


def _resample_mono(wav: torch.Tensor, sample_rate: int, target_rate: int = 16000) -> torch.Tensor:
    if sample_rate == target_rate:
        return wav

    if wav.numel() == 0:
        return wav

    # Linear interpolation is sufficient here and avoids pulling in torchaudio/ffmpeg.
    num_samples = int(round(wav.shape[-1] * target_rate / sample_rate))
    if num_samples <= 0:
        return wav.new_zeros((1, 0))

    wav_3d = wav.unsqueeze(0)
    resampled = torch.nn.functional.interpolate(
        wav_3d,
        size=num_samples,
        mode="linear",
        align_corners=False,
    )
    return resampled.squeeze(0)


def _decode_wav(upload_bytes: bytes) -> torch.Tensor:
    try:
        with wave.open(io.BytesIO(upload_bytes), "rb") as wav_file:
            num_channels = wav_file.getnchannels()
            sample_rate = wav_file.getframerate()
            sample_width = wav_file.getsampwidth()
            num_frames = wav_file.getnframes()
            raw_audio = wav_file.readframes(num_frames)
    except wave.Error as exc:
        raise ValueError(f"unsupported WAV container: {exc}") from exc

    if num_frames == 0:
        raise ValueError("WAV file contains no audio frames")

    if sample_width == 1:
        # 8-bit PCM WAV is unsigned.
        audio = (np.frombuffer(raw_audio, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(raw_audio, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 3:
        # Convert 24-bit PCM to 16-bit PCM before normalization.
        audio = np.frombuffer(audioop.lin2lin(raw_audio, 3, 2), dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(raw_audio, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported sample width: {sample_width * 8} bits")

    audio_tensor = torch.from_numpy(np.asarray(audio, dtype=np.float32))

    if num_channels > 1:
        frame_count = audio_tensor.numel() // num_channels
        audio_tensor = audio_tensor[: frame_count * num_channels].reshape(frame_count, num_channels).mean(dim=1)

    wav = audio_tensor.unsqueeze(0).contiguous()

    if sample_rate != 16000:
        wav = _resample_mono(wav, sample_rate, 16000)

    return wav


async def transcribe_endpoint(
    file: UploadFile = File(...),
    language: str = Form("as"),
):
    if file.content_type not in {"audio/wav", "audio/x-wav", "audio/wave", "application/octet-stream"}:
        raise HTTPException(
            status_code=415,
            detail="Only WAV uploads are supported",
        )

    upload_bytes = await file.read()
    if not upload_bytes:
        raise HTTPException(status_code=400, detail="Empty file upload")

    try:
        wav = _decode_wav(upload_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid WAV file: {exc}") from exc

    request_id = uuid.uuid4().hex
    client_id = request_id

    manager.create_pending(request_id)

    job = TranscriptionJob(
        client_id=client_id,
        request_id=request_id,
        chunk_id=0,
        wav=wav,
        language=language,
        is_partial=False,
    )

    try:
        transcription_queue.put_nowait(job)
    except Exception as exc:
        manager.reject(request_id, str(exc))
        manager.discard(request_id)
        raise HTTPException(status_code=429, detail="Transcription queue is full") from exc

    try:
        result = await manager.wait_for(request_id, timeout=600)
    except asyncio.TimeoutError as exc:
        manager.discard(request_id)
        raise HTTPException(status_code=504, detail="Transcription timed out") from exc
    except RuntimeError as exc:
        manager.discard(request_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "request_id": request_id,
        "transcript": result["text"],
    }
