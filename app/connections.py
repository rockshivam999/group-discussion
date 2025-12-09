from typing import Dict, List

from fastapi import WebSocket


class ConnectionManager:
    def __init__(self) -> None:
        self.teacher_sockets: List[WebSocket] = []

    async def connect_teacher(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.teacher_sockets.append(websocket)

    def disconnect_teacher(self, websocket: WebSocket) -> None:
        if websocket in self.teacher_sockets:
            self.teacher_sockets.remove(websocket)

    async def broadcast_alert(self, data: Dict) -> None:
        stale: List[WebSocket] = []
        for connection in self.teacher_sockets:
            try:
                await connection.send_json(data)
            except Exception:
                stale.append(connection)
        for conn in stale:
            self.disconnect_teacher(conn)
