import asyncio
from rapidfuzz import fuzz

from .queue_manager import (
    cleanup_client_queue_state,
    dequeue_transcription_job,
    transcription_queue,
)
from .connection_manager import manager
from .model import transcribe


# =========================================================
# SmartDeduplicator
# =========================================================

class SmartDeduplicator:

    def __init__(
        self,
        debug=False,
        fuzzy_threshold=70,
    ):
        self.debug = debug
        self.fuzzy_threshold = fuzzy_threshold

    def deduplicate(
        self,
        prev_text: str,
        curr_text: str,
    ):

        if not prev_text or not curr_text:
            return curr_text, {
                "overlap_words": 0,
            }

        prev_words = prev_text.split()

        curr_words = curr_text.split()

        best_prev_len = 0
        best_curr_len = 0
        best_score = 0

        max_overlap = min(
            8,
            len(prev_words),
            len(curr_words),
        )

        for prev_len in range(1, max_overlap + 1):

            for curr_len in range(
                max(1, prev_len - 1),
                min(prev_len + 2, len(curr_words)) + 1,
            ):

                prev_span = " ".join(
                    prev_words[-prev_len:]
                )

                curr_span = " ".join(
                    curr_words[:curr_len]
                )

                score = fuzz.ratio(
                    prev_span,
                    curr_span,
                )

                if (
                    score >= self.fuzzy_threshold
                    and score > best_score
                ):

                    best_score = score
                    best_prev_len = prev_len
                    best_curr_len = curr_len

        if best_curr_len == 0:

            merged_text = (
                prev_text + " " + curr_text
            ).strip()

        else:

            merged_text = " ".join(
                prev_words
                + curr_words[best_curr_len:]
            )

        return merged_text, {
            "overlap_words": best_curr_len,
            "score": best_score,
        }


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


# =========================================================
# Per-client finalized state
# =========================================================

# ONLY finalized transcript lives here
# partials NEVER update this

client_final_text: dict[str, str] = {}


# =========================================================
# Worker
# =========================================================

async def inference_worker(worker_id: int):

    print(f"Worker {worker_id} started")

    dedup = SmartDeduplicator(
        fuzzy_threshold=70
    )

    while True:

        job = await dequeue_transcription_job()

        try:

            # =============================================
            # 1. ASR inference
            # =============================================

            raw_text: str = await asyncio.to_thread(
                transcribe,
                job.wav,
                job.language,
            )

            if (
                not raw_text
                or not raw_text.strip()
            ):
                continue

            raw_text = raw_text.strip()

            connection = manager.get(job.client_id)

            if not connection:
                continue

            # =============================================
            # FINALIZED CHUNK
            # =============================================

            prev_final = client_final_text.get(
                job.client_id,
                "",
            )

            merged_text, dedup_stats = (
                dedup.deduplicate(
                    prev_final,
                    raw_text,
                )
            )

            # =============================================
            # extract ONLY newly added portion
            # =============================================

            if (
                prev_final
                and merged_text.startswith(
                    prev_final
                )
            ):

                new_portion = merged_text[
                    len(prev_final):
                ].strip()

            else:

                curr_words = raw_text.split()

                skip = dedup_stats.get(
                    "overlap_words",
                    0,
                )

                new_portion = " ".join(
                    curr_words[skip:]
                ).strip()

            if not new_portion:
                continue

            # =============================================
            # postprocess
            # =============================================

            final_text = post_process_transcript(
                new_portion
            )

            if not final_text.strip():
                continue

            # =============================================
            # update ONLY finalized state
            # =============================================

            client_final_text[
                job.client_id
            ] = merged_text

            # =============================================
            # send finalized transcript
            # =============================================

            await connection.send_json({
                "client_id": job.client_id,
                "type": "transcription",
                "request_id": job.request_id,
                "chunk_id": job.chunk_id,
                "text": final_text,
                "is_partial": False,
            })

        except Exception as e:

            connection = manager.get(
                job.client_id
            )

            if connection:

                await connection.send_json({
                    "type": "error",
                    "request_id": job.request_id,
                    "message": str(e),
                })

        finally:

            transcription_queue.task_done()


# =========================================================
# Cleanup
# =========================================================

def cleanup_client(client_id: str):
    cleanup_client_queue_state(client_id)

    client_final_text.pop(
        client_id,
        None,
    )


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
