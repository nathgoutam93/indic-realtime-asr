import asyncio
from rapidfuzz import fuzz

from .connection_manager import manager
from .queue_manager import transcription_queue
from .model import transcribe


# =========================================================
# Postprocessor
# =========================================================

def post_process_transcript(
    text: str,
    min_repeat_len: int = 4,
    threshold: int = 95,
) -> str:

    words = text.split()

    result = list(words)

    changed = True

    while changed:

        changed = False

        i = 0

        while i < len(result):

            for seq_len in range(
                6,
                min_repeat_len - 1,
                -1,
            ):

                if i + seq_len > len(result):
                    continue

                candidate_str = " ".join(
                    result[i:i + seq_len]
                )

                search_limit = min(
                    i + seq_len + seq_len,
                    len(result) - seq_len + 1,
                )

                for j in range(
                    i + seq_len,
                    search_limit,
                ):

                    window_str = " ".join(
                        result[j:j + seq_len]
                    )

                    if (
                        fuzz.ratio(
                            candidate_str,
                            window_str,
                        )
                        >= threshold
                    ):

                        result = (
                            result[:j]
                            + result[j + seq_len:]
                        )

                        changed = True

                        break

                if changed:
                    break

            i += 1

    return " ".join(result)


def _build_result_payload(job, text: str) -> dict:
    return {
        "client_id": job.client_id,
        "type": "transcription",
        "request_id": job.request_id,
        "chunk_id": job.chunk_id,
        "text": text,
    }


async def _deliver_result(job, text: str):
    payload = _build_result_payload(
        job,
        text,
    )

    manager.resolve(job.request_id, payload)


# =========================================================
# Worker
# =========================================================

async def inference_worker(worker_id: int):

    print(f"Worker {worker_id} started")

    while True:

        job = await transcription_queue.get()

        try:

            # =============================================
            # 1. ASR inference
            # =============================================

            raw_text: str = await asyncio.to_thread(
                transcribe,
                job.wav,
                job.language,
            )

            if not raw_text or not raw_text.strip():
                await _deliver_result(job, "")
                continue

            final_text = post_process_transcript(
                raw_text.strip()
            )

            await _deliver_result(
                job,
                final_text,
            )

        except Exception as e:
            manager.reject(job.request_id, str(e))

        finally:

            transcription_queue.task_done()


# =========================================================
# Cleanup
# =========================================================

def cleanup_client(client_id: str):
    return None


# =========================================================
# Start workers
# =========================================================

async def start_workers(
    num_workers: int = 4
):

    for i in range(num_workers):

        asyncio.create_task(
            inference_worker(i)
        )

