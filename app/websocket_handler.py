import uuid
import threading
import io
import numpy as np
import torch
import av
import asyncio

from fastapi import WebSocket
from starlette.websockets import WebSocketDisconnect

from silero_vad import load_silero_vad, VADIterator

from .queue_manager import transcription_queue
from .schemas import TranscriptionJob
from .connection_manager import manager
from .workers import cleanup_client


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

# Partial realtime transcription interval
PARTIAL_TRANSCRIPTION_INTERVAL_SECONDS = 1.0

PARTIAL_SAMPLES = int(
    SAMPLE_RATE * PARTIAL_TRANSCRIPTION_INTERVAL_SECONDS
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

    # =====================================================
    # Load Silero VAD
    # =====================================================

    vad_model = load_silero_vad()

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

        speech_buffer = []

        partial_sent_samples = 0

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

                                vad_result = vad_iterator(
                                    pcm_tensor
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

                                # =====================
                                # accumulate speech
                                # =====================

                                if speech_active:

                                    speech_buffer.extend(
                                        vad_frame
                                    )

                                    current_samples = len(
                                        speech_buffer
                                    )

                                    # ================
                                    # partial realtime
                                    # transcription
                                    # ================

                                    if (
                                        current_samples
                                        - partial_sent_samples
                                        >= PARTIAL_SAMPLES
                                    ):

                                        audio_np = np.array(
                                            speech_buffer,
                                            dtype=np.float32,
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
                                                is_partial=True,
                                                language=language,
                                            )
                                        )

                                        state[
                                            "chunk_counter"
                                        ] += 1

                                        partial_sent_samples = (
                                            current_samples
                                        )

                                        asyncio.run_coroutine_threadsafe(
                                            transcription_queue.put(
                                                job
                                            ),
                                            loop,
                                        )

                                        print(
                                            f"[{client_id}] "
                                            f"Partial chunk queued "
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

                                    if len(speech_buffer) > 0:

                                        audio_np = np.array(
                                            speech_buffer,
                                            dtype=np.float32,
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
                                            transcription_queue.put(
                                                job
                                            ),
                                            loop,
                                        )

                                        print(
                                            f"[{client_id}] "
                                            f"Final chunk queued "
                                            f"({len(audio_np)} samples)"
                                        )

                                    # =================
                                    # reset utterance
                                    # =================

                                    speech_buffer = []

                                    partial_sent_samples = 0

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
