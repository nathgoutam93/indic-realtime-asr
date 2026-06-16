from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class TranscriptionJob:
    client_id: str
    request_id: str
    chunk_id: int
    wav: Any
    language: str = "as"
    is_partial: bool = False
