import uuid
import threading
import io
import numpy as np
import torch
import av
import asyncio

from fastapi import WebSocket, Query
from starlette.websockets import WebSocketDisconnect

from .queue_manager import transcription_queue
from .schemas import TranscriptionJob
from .connection_manager import manager
from .workers import cleanup_client

# =========================================================
# Audio Configuration
# =========================================================

SAMPLE_RATE             = 16000
BUFFER_DURATION_SECONDS = 2
BUFFER_SIZE_SAMPLES     = SAMPLE_RATE * BUFFER_DURATION_SECONDS  # 32,000 samples
OVERLAP_SECONDS         = 0.5
OVERLAP_SIZE_SAMPLES    = int(SAMPLE_RATE * OVERLAP_SECONDS)     # 8,000 samples


# =========================================================
# Blocking pipe: asyncio producer → PyAV consumer
# =========================================================

class WebMPipe(io.RawIOBase):
    """
    Thread-safe blocking pipe.

    - asyncio side calls feed(bytes)  — never blocks
    - PyAV's IO thread calls readinto() — blocks until data or EOF
    """

    def __init__(self):
        super().__init__()
        self._buf  = bytearray()
        self._eof  = False
        self._cond = threading.Condition(threading.Lock())

    def feed(self, data: bytes) -> None:
        with self._cond:
            self._buf.extend(data)
            self._cond.notify_all()

    def close_pipe(self) -> None:
        with self._cond:
            self._eof = True
            self._cond.notify_all()
        super().close()

    # ── RawIOBase interface ──────────────────────────────

    def readable(self) -> bool:
        return True

    def readinto(self, b: bytearray) -> int:
        with self._cond:
            while not self._buf and not self._eof:
                self._cond.wait()
            if not self._buf:
                return 0                        # EOF → PyAV stops demuxing
            n = min(len(b), len(self._buf))
            b[:n] = self._buf[:n]
            del self._buf[:n]
            return n


# =========================================================
# Per-client PCM sample buffer
# =========================================================

client_buffers: dict[str, list] = {}


# =========================================================
# WebSocket endpoint
# =========================================================
async def websocket_endpoint(websocket: WebSocket):

    client_id = str(uuid.uuid4())
    language = websocket.query_params.get("language", "as")

    await manager.connect(client_id, websocket)
    print(f"[{client_id}] Client connected")

    client_buffers[client_id] = []
    chunk_counter  = 0
    pipe           = WebMPipe()
    loop           = asyncio.get_event_loop()

    # ──────────────────────────────────────────────────────
    # Decode thread
    # av.open() + demux + decode all happen here so they
    # never touch the asyncio event loop thread.
    # ──────────────────────────────────────────────────────
    def decode_thread():
        nonlocal chunk_counter

        print(f"[{client_id}] Decode thread started — opening container...")

        try:
            # av.open() calls readinto() here; it will BLOCK until
            # the asyncio loop feeds the first WebM bytes via pipe.feed().
            # This is fine because we are already inside a daemon thread.
            container = av.open(
                pipe,
                mode='r',
                format='webm',
                options={
                    'fflags':          'nobuffer',
                    'flags':           'low_delay',
                    'analyzeduration': '0',
                    'probesize':       '32',
                },
            )
        except Exception as e:
            print(f"[{client_id}] av.open() failed: {repr(e)}")
            return

        print(f"[{client_id}] Container opened — decoding...")

        resampler = av.AudioResampler(format='flt', layout='mono', rate=SAMPLE_RATE)

        try:
            for packet in container.demux():
                if packet.stream.type != 'audio':
                    continue
                try:
                    for frame in packet.decode():
                        for rf in resampler.resample(frame):
                            pcm = rf.to_ndarray().flatten().astype(np.float32)
                            client_buffers[client_id].extend(pcm)

                        # Slice inference windows as buffer fills
                        while len(client_buffers[client_id]) >= BUFFER_SIZE_SAMPLES:
                            inference_samples = client_buffers[client_id][:BUFFER_SIZE_SAMPLES]

                            if OVERLAP_SIZE_SAMPLES > 0:
                                client_buffers[client_id] = client_buffers[client_id][
                                    BUFFER_SIZE_SAMPLES - OVERLAP_SIZE_SAMPLES:
                                ]
                            else:
                                client_buffers[client_id] = client_buffers[client_id][
                                    BUFFER_SIZE_SAMPLES:
                                ]

                            audio_np = np.array(inference_samples, dtype=np.float32)
                            wav      = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0)

                            job = TranscriptionJob(
                                request_id=str(uuid.uuid4()),
                                client_id=client_id,
                                chunk_id=chunk_counter,
                                wav=wav,
                                language=language,
                            )
                            chunk_counter += 1

                            asyncio.run_coroutine_threadsafe(
                                transcription_queue.put(job),
                                loop,
                            )

                            print(
                                f"[{client_id}] Enqueued chunk {chunk_counter} "
                                f"({len(audio_np)} samples)"
                            )

                except av.FFmpegError as e:
                    print(f"[{client_id}] Decode error (skipping packet): {repr(e)}")
                    continue

        except Exception as e:
            print(f"[{client_id}] Demux loop ended: {repr(e)}")

        finally:
            container.close()
            print(f"[{client_id}] Decode thread exiting")

    # Start the decode thread BEFORE we enter the receive loop,
    # so it is already waiting in readinto() when the first bytes arrive.
    decoder = threading.Thread(target=decode_thread, daemon=True)
    decoder.start()

    # ──────────────────────────────────────────────────────
    # asyncio receive loop — just feed bytes into the pipe
    # ──────────────────────────────────────────────────────
    try:
        while True:
            data = await websocket.receive_bytes()
            pipe.feed(data)

    except WebSocketDisconnect:
        print(f"[{client_id}] Client disconnected")

    except Exception as e:
        print(f"[{client_id}] WebSocket error: {repr(e)}")

    finally:
        pipe.close_pipe()
        decoder.join(timeout=5)
        manager.disconnect(client_id)
        client_buffers.pop(client_id, None)
        cleanup_client(client_id)              # ← clears dedup state
        print(f"[{client_id}] Cleanup done")
