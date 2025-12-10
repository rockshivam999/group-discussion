from typing import Dict, List

from fastapi import WebSocket


class ConnectionManager:
    def __init__(self) -> None:
        self.teacher_sockets: List[WebSocket] = []
        self.group_subscribers: Dict[str, List[WebSocket]] = {}

    async def connect_teacher(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.teacher_sockets.append(websocket)

    def disconnect_teacher(self, websocket: WebSocket) -> None:
        if websocket in self.teacher_sockets:
            self.teacher_sockets.remove(websocket)

    async def connect_group(self, group_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self.group_subscribers.setdefault(group_id, []).append(websocket)

    def disconnect_group(self, group_id: str, websocket: WebSocket) -> None:
        if group_id in self.group_subscribers and websocket in self.group_subscribers[group_id]:
            self.group_subscribers[group_id].remove(websocket)
            if not self.group_subscribers[group_id]:
                self.group_subscribers.pop(group_id, None)

    async def broadcast_alert(self, data: Dict) -> None:
        stale: List[WebSocket] = []
        for connection in self.teacher_sockets:
            try:
                await connection.send_json(data)
            except Exception:
                stale.append(connection)
        for conn in stale:
            self.disconnect_teacher(conn)

        # Targeted broadcast to group subscribers
        group_id = data.get("group_id")
        if not group_id:
            return
        stale_group: List[WebSocket] = []
        for conn in self.group_subscribers.get(group_id, []):
            try:
                await conn.send_json(data)
            except Exception:
                stale_group.append(conn)
        for conn in stale_group:
            self.disconnect_group(group_id, conn)
