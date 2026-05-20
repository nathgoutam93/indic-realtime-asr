import asyncio
from .queue_manager import transcription_queue
from .connection_manager import manager
from .model import transcribe
from rapidfuzz import fuzz
import re


# =========================================================
# SmartDeduplicator
# =========================================================

class SmartDeduplicator:
    def __init__(self, debug=False, fuzzy_threshold=70):
        self.debug = debug
        self.fuzzy_threshold = fuzzy_threshold

    def _words_match(self, w1: str, w2: str) -> bool:
        if w1 == w2:
            return True
        return fuzz.ratio(w1, w2) >= self.fuzzy_threshold

    def deduplicate(self, prev_text: str, curr_text: str, chunk_idx=0):
        if not prev_text or not curr_text:
            return (prev_text + " " + curr_text).strip(), {'overlap_words': 0}

        prev_words = prev_text.split()
        curr_words = curr_text.split()

        best_prev_len = 0
        best_curr_len = 0
        best_score    = 0

        for prev_len in range(1, min(8, len(prev_words)) + 1):
            for curr_len in range(max(1, prev_len - 1), min(prev_len + 2, len(curr_words)) + 1):
                prev_span = " ".join(prev_words[-prev_len:])
                curr_span = " ".join(curr_words[:curr_len])
                score = fuzz.ratio(prev_span, curr_span)
                if score >= self.fuzzy_threshold and score > best_score:
                    best_score    = score
                    best_prev_len = prev_len
                    best_curr_len = curr_len

        if best_curr_len == 0:
            merged_text = (prev_text + " " + curr_text).strip()
        else:
            merged_text = " ".join(prev_words + curr_words[best_curr_len:])

        return merged_text, {'overlap_words': best_curr_len, 'score': best_score}


# =========================================================
# Postprocessor
# =========================================================

def post_process_transcript(text: str, min_repeat_len: int = 4, threshold: int = 95) -> str:
    words  = text.split()
    result = list(words)

    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(result):
            for seq_len in range(6, min_repeat_len - 1, -1):
                if i + seq_len > len(result):
                    continue

                candidate_str = " ".join(result[i:i + seq_len])
                search_limit  = min(i + seq_len + seq_len, len(result) - seq_len + 1)

                for j in range(i + seq_len, search_limit):
                    window_str = " ".join(result[j:j + seq_len])
                    if fuzz.ratio(candidate_str, window_str) >= threshold:
                        result  = result[:j] + result[j + seq_len:]
                        changed = True
                        break

                if changed:
                    break
            i += 1

    return " ".join(result)


# =========================================================
# Per-client dedup state
# =========================================================

# Stores the last confirmed transcript text per client
# so SmartDeduplicator can compare across consecutive chunks
client_last_text: dict[str, str] = {}


def get_or_create_dedup(client_id: str) -> SmartDeduplicator:
    # One shared instance is fine — dedup is stateless per-call.
    # State lives in client_last_text, not inside the class.
    return SmartDeduplicator(debug=False, fuzzy_threshold=70)


# =========================================================
# Worker
# =========================================================

async def inference_worker(worker_id: int):
    print(f"Worker {worker_id} started")

    while True:
        job = await transcription_queue.get()

        try:
            # ── 1. ASR inference ──────────────────────────────
            raw_text: str = await asyncio.to_thread(
                transcribe,
                job.wav,
                job.language,
            )

            if not raw_text or not raw_text.strip():
                continue                        # silence / empty — skip

            # ── 2. Deduplicate against previous chunk ─────────
            prev_text = client_last_text.get(job.client_id, "")
            dedup     = SmartDeduplicator(fuzzy_threshold=70)

            merged_text, dedup_stats = dedup.deduplicate(
                prev_text,
                raw_text,
                chunk_idx=job.chunk_id,
            )

            # The new tail becomes the reference for the next chunk
            client_last_text[job.client_id] = merged_text

            # Strip the already-sent prefix so we only send the NEW portion
            # i.e. everything in merged_text beyond what was in prev_text
            if prev_text and merged_text.startswith(prev_text):
                new_portion = merged_text[len(prev_text):].strip()
            else:
                # Fallback: send the deduplicated curr portion only
                curr_words   = raw_text.split()
                skip         = dedup_stats.get('overlap_words', 0)
                new_portion  = " ".join(curr_words[skip:]).strip()

            if not new_portion:
                continue                        # was entirely overlap — discard

            # ── 3. Postprocess ────────────────────────────────
            final_text = post_process_transcript(new_portion)

            if not final_text.strip():
                continue

            # ── 4. Send to client ─────────────────────────────
            websocket = manager.get(job.client_id)
            if websocket:
                await websocket.send_json({
                    "client_id":  job.client_id,
                    "type":       "transcription",
                    "request_id": job.request_id,
                    "chunk_id":   job.chunk_id,
                    "text":       final_text,
                    # Optional debug fields — remove in production
                    "_raw":       raw_text,
                    "_dedup":     dedup_stats,
                })

        except Exception as e:
            websocket = manager.get(job.client_id)
            if websocket:
                await websocket.send_json({
                    "type":       "error",
                    "request_id": job.request_id,
                    "message":    str(e),
                })

        finally:
            transcription_queue.task_done()


# =========================================================
# Cleanup hook — call this from websocket_handler.py finally
# =========================================================

def cleanup_client(client_id: str) -> None:
    """Remove all per-client state when a client disconnects."""
    client_last_text.pop(client_id, None)


# =========================================================
# Start workers
# =========================================================

async def start_workers(num_workers: int = 4):
    for i in range(num_workers):
        asyncio.create_task(inference_worker(i))
