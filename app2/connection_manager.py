import asyncio
from typing import Any


class ConnectionManager:
    def __init__(self) -> None:
        self._pending: dict[str, asyncio.Future] = {}

    def create_pending(self, request_id: str) -> asyncio.Future:
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._pending[request_id] = future
        return future

    def resolve(self, request_id: str, payload: Any) -> None:
        future = self._pending.get(request_id)
        if future and not future.done():
            future.set_result(payload)

    def reject(self, request_id: str, error: str) -> None:
        future = self._pending.get(request_id)
        if future and not future.done():
            future.set_exception(RuntimeError(error))

    def discard(self, request_id: str) -> None:
        self._pending.pop(request_id, None)

    async def wait_for(self, request_id: str, timeout: float | None = None) -> Any:
        future = self._pending.get(request_id)
        if future is None:
            raise RuntimeError(f"No pending request found for {request_id}")
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        finally:
            pending = self._pending.get(request_id)
            if pending is future:
                self._pending.pop(request_id, None)


manager = ConnectionManager()
