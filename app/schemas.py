from dataclasses import dataclass
import torch


@dataclass
class TranscriptionJob:
    request_id: str
    client_id: str
    chunk_id: int
    wav: torch.Tensor
    language: str
