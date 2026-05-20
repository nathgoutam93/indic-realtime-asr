from fastapi import WebSocket


class ConnectionManager:
    def __init__(self):
        self.active_connections = {}

    async def connect(
        self,
        client_id: str,
        websocket: WebSocket
    ):
        await websocket.accept()

        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        self.active_connections.pop(client_id, None)

    def get(self, client_id: str):
        return self.active_connections.get(client_id)


manager = ConnectionManager()
