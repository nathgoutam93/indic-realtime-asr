import asyncio

from .jobs import TranscriptionJob

transcription_queue: asyncio.Queue[TranscriptionJob] = asyncio.Queue(maxsize=1000)

