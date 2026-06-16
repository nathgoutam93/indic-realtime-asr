import uuid
import threading
import io
import os
import numpy as np
import torch
import av
import asyncio

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

from silero_vad import VADIterator

from .queue_manager import enqueue_transcription_job
from .schemas import TranscriptionJob
from .connection_manager import manager
from .workers import cleanup_client
from .vad_model import get_vad_model, run_vad


# =========================================================
# Audio Configuration
# =========================================================

SAMPLE_RATE = 16000

# Silero VAD ONLY supports:
# 512 samples @ 16kHz
VAD_FRAME_SIZE = 512

# VAD tuning
VAD_THRESHOLD = 0.5
MIN_SILENCE_DURATION_MS = 500
SPEECH_PAD_MS = 200

# Fixed finalized speech chunks
TRANSCRIPTION_CHUNK_SECONDS = float(
    os.getenv(
        "TRANSCRIPTION_CHUNK_SECONDS",
        "2.0",
    )
)

TRANSCRIPTION_OVERLAP_SECONDS = float(
    os.getenv(
        "TRANSCRIPTION_OVERLAP_SECONDS",
        "0.5",
    )
)

MIN_FINAL_FLUSH_SECONDS = float(
    os.getenv(
        "MIN_FINAL_FLUSH_SECONDS",
        "0.6",
    )
)

CHUNK_SAMPLES = int(
    SAMPLE_RATE * TRANSCRIPTION_CHUNK_SECONDS
)

OVERLAP_SAMPLES = int(
    SAMPLE_RATE * TRANSCRIPTION_OVERLAP_SECONDS
)

MIN_FINAL_FLUSH_SAMPLES = int(
    SAMPLE_RATE * MIN_FINAL_FLUSH_SECONDS
)

if CHUNK_SAMPLES <= 0:
    raise ValueError(
        "TRANSCRIPTION_CHUNK_SECONDS must be greater than 0"
    )

if OVERLAP_SAMPLES >= CHUNK_SAMPLES:
    raise ValueError(
        "TRANSCRIPTION_OVERLAP_SECONDS must be less than "
        "TRANSCRIPTION_CHUNK_SECONDS"
    )

# =========================================================
# Blocking pipe: asyncio producer → PyAV consumer
# =========================================================

class WebMPipe(io.RawIOBase):

    def __init__(self):
        super().__init__()

        self._buf = bytearray()
        self._eof = False

        self._cond = threading.Condition(
            threading.Lock()
        )

    def feed(self, data: bytes) -> None:

        with self._cond:

            self._buf.extend(data)

            self._cond.notify_all()

    def close_pipe(self) -> None:

        with self._cond:

            self._eof = True

            self._cond.notify_all()

        super().close()

    def readable(self) -> bool:
        return True

    def readinto(self, b: bytearray) -> int:

        with self._cond:

            while not self._buf and not self._eof:
                self._cond.wait()

            if not self._buf:
                return 0

            n = min(len(b), len(self._buf))

            b[:n] = self._buf[:n]

            del self._buf[:n]

            return n


# =========================================================
# Client State
# =========================================================

client_states = {}


# =========================================================
# WebSocket Endpoint
# =========================================================

async def websocket_endpoint(websocket: WebSocket):

    client_id = str(uuid.uuid4())

    language = websocket.query_params.get(
        "language",
        "en"
    )

    await manager.connect(client_id, websocket)

    print(f"[{client_id}] Client connected")

    loop = asyncio.get_event_loop()

    pipe = WebMPipe()

    client_states[client_id] = {
        "chunk_counter": 0,
    }

    # Each client needs its own iterator state, but the model is shared.
    vad_model = get_vad_model()

    vad_iterator = VADIterator(
        vad_model,
        threshold=VAD_THRESHOLD,
        sampling_rate=SAMPLE_RATE,
        min_silence_duration_ms=MIN_SILENCE_DURATION_MS,
        speech_pad_ms=SPEECH_PAD_MS,
    )

    # =====================================================
    # Decode Thread
    # =====================================================

    def decode_thread():

        state = client_states[client_id]

        speech_active = False

        speech_buffer = np.array(
            [],
            dtype=np.float32,
        )

        emitted_speech_chunk = False

        # Buffer used ONLY for VAD framing
        vad_frame_buffer = np.array(
            [],
            dtype=np.float32
        )

        print(
            f"[{client_id}] "
            f"Decode thread started"
        )

        try:

            container = av.open(
                pipe,
                mode='r',
                format='webm',
                options={
                    'fflags': 'nobuffer',
                    'flags': 'low_delay',
                    'analyzeduration': '0',
                    'probesize': '32',
                },
            )

        except Exception as e:

            print(
                f"[{client_id}] "
                f"av.open failed: {repr(e)}"
            )

            return

        print(
            f"[{client_id}] "
            f"Container opened"
        )

        resampler = av.AudioResampler(
            format='flt',
            layout='mono',
            rate=SAMPLE_RATE,
        )

        try:

            for packet in container.demux():

                if packet.stream.type != 'audio':
                    continue

                try:

                    for frame in packet.decode():

                        for rf in resampler.resample(frame):

                            pcm = (
                                rf.to_ndarray()
                                .flatten()
                                .astype(np.float32)
                            )

                            # =============================
                            # accumulate arbitrary PCM
                            # into exact 512 sample frames
                            # =============================

                            vad_frame_buffer = np.concatenate(
                                [vad_frame_buffer, pcm]
                            )

                            while (
                                len(vad_frame_buffer)
                                >= VAD_FRAME_SIZE
                            ):

                                # =====================
                                # exact 512 samples
                                # =====================

                                vad_frame = (
                                    vad_frame_buffer[
                                        :VAD_FRAME_SIZE
                                    ]
                                )

                                vad_frame_buffer = (
                                    vad_frame_buffer[
                                        VAD_FRAME_SIZE:
                                    ]
                                )

                                # =====================
                                # VAD inference
                                # =====================

                                pcm_tensor = torch.from_numpy(
                                    vad_frame
                                )

                                vad_result = run_vad(
                                    vad_iterator,
                                    pcm_tensor,
                                )

                                # =====================
                                # speech start
                                # =====================

                                if (
                                    vad_result is not None
                                    and "start" in vad_result
                                ):

                                    speech_active = True

                                    print(
                                        f"[{client_id}] "
                                        f"Speech started"
                                    )

                                if speech_active:

                                    speech_buffer = np.concatenate(
                                        [
                                            speech_buffer,
                                            vad_frame,
                                        ]
                                    )

                                    while (
                                        len(speech_buffer)
                                        >= CHUNK_SAMPLES
                                    ):

                                        audio_np = (
                                            speech_buffer[
                                                :CHUNK_SAMPLES
                                            ].copy()
                                        )

                                        wav = torch.tensor(
                                            audio_np,
                                            dtype=torch.float32,
                                        ).unsqueeze(0)

                                        job = (
                                            TranscriptionJob(
                                                request_id=str(
                                                    uuid.uuid4()
                                                ),
                                                client_id=client_id,
                                                chunk_id=state[
                                                    "chunk_counter"
                                                ],
                                                wav=wav,
                                                is_partial=False,
                                                language=language,
                                            )
                                        )

                                        state[
                                            "chunk_counter"
                                        ] += 1

                                        asyncio.run_coroutine_threadsafe(
                                            enqueue_transcription_job(
                                                job
                                            ),
                                            loop,
                                        ).result()

                                        emitted_speech_chunk = True

                                        if OVERLAP_SAMPLES > 0:
                                            speech_buffer = (
                                                speech_buffer[
                                                    CHUNK_SAMPLES
                                                    - OVERLAP_SAMPLES:
                                                ]
                                            )
                                        else:
                                            speech_buffer = (
                                                speech_buffer[
                                                    CHUNK_SAMPLES:
                                                ]
                                            )

                                        print(
                                            f"[{client_id}] "
                                            f"Final chunk queued "
                                            f"({len(audio_np)} samples)"
                                        )

                                # =====================
                                # speech end
                                # =====================

                                if (
                                    vad_result is not None
                                    and "end" in vad_result
                                ):

                                    print(
                                        f"[{client_id}] "
                                        f"Speech ended"
                                    )

                                    speech_active = False

                                    if emitted_speech_chunk:
                                        flush_new_samples = max(
                                            0,
                                            len(speech_buffer)
                                            - OVERLAP_SAMPLES,
                                        )
                                    else:
                                        flush_new_samples = len(
                                            speech_buffer
                                        )

                                    if (
                                        flush_new_samples
                                        >= MIN_FINAL_FLUSH_SAMPLES
                                    ):

                                        audio_np = (
                                            speech_buffer.copy()
                                        )

                                        wav = torch.tensor(
                                            audio_np,
                                            dtype=torch.float32,
                                        ).unsqueeze(0)

                                        job = (
                                            TranscriptionJob(
                                                request_id=str(
                                                    uuid.uuid4()
                                                ),
                                                client_id=client_id,
                                                chunk_id=state[
                                                    "chunk_counter"
                                                ],
                                                wav=wav,
                                                is_partial=False,
                                                language=language,
                                            )
                                        )

                                        state[
                                            "chunk_counter"
                                        ] += 1

                                        asyncio.run_coroutine_threadsafe(
                                            enqueue_transcription_job(
                                                job
                                            ),
                                            loop,
                                        ).result()

                                        print(
                                            f"[{client_id}] "
                                            f"Final chunk queued "
                                            f"({len(audio_np)} samples)"
                                        )

                                    # =================
                                    # reset utterance
                                    # =================

                                    speech_buffer = np.array(
                                        [],
                                        dtype=np.float32,
                                    )

                                    emitted_speech_chunk = False

                except av.FFmpegError as e:

                    print(
                        f"[{client_id}] "
                        f"Decode error: {repr(e)}"
                    )

                    continue

        except Exception as e:

            print(
                f"[{client_id}] "
                f"Demux loop ended: {repr(e)}"
            )

        finally:

            container.close()

            print(
                f"[{client_id}] "
                f"Decode thread exiting"
            )

    # =====================================================
    # Start decoder thread
    # =====================================================

    decoder = threading.Thread(
        target=decode_thread,
        daemon=True,
    )

    decoder.start()

    # =====================================================
    # WebSocket receive loop
    # =====================================================

    try:

        while True:

            data = await websocket.receive_bytes()

            pipe.feed(data)

    except WebSocketDisconnect:

        print(
            f"[{client_id}] "
            f"Client disconnected"
        )

    except Exception as e:

        print(
            f"[{client_id}] "
            f"WebSocket error: {repr(e)}"
        )

    finally:

        pipe.close_pipe()

        decoder.join(timeout=5)

        manager.disconnect(client_id)

        client_states.pop(client_id, None)

        cleanup_client(client_id)

        print(
            f"[{client_id}] "
            f"Cleanup done"
        )
