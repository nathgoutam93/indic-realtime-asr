import asyncio

from .schemas import TranscriptionJob

transcription_queue = asyncio.Queue(maxsize=1000)


async def enqueue_transcription_job(
    job: TranscriptionJob,
) -> bool:
    await transcription_queue.put(job)
    return True


async def dequeue_transcription_job() -> TranscriptionJob:
    return await transcription_queue.get()


def cleanup_client_queue_state(
    client_id: str,
) -> None:
    return None
