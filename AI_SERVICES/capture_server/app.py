import asyncio
from pathlib import Path
from typing import Set

import httpx
from fastapi import FastAPI, File, Form, Header, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


class NodeHub:
    """Quan ly cac ket noi WebSocket giua Dien thoai va Desktop Host."""
    def __init__(self):
        self.mobile_sockets: Set[WebSocket] = set()
        self.host_sockets: Set[WebSocket] = set()
        self.last_result: dict | None = None
        self.active_nodes: dict = {}

    async def connect_mobile(self, ws: WebSocket, node_id: str, label: str):
        await ws.accept()
        self.mobile_sockets.add(ws)
        self.active_nodes[node_id] = label
        await self.broadcast_to_hosts({
            "type": "node_status",
            "nodes_count": len(self.active_nodes),
            "nodes": list(self.active_nodes.values()),
        })

    def disconnect_mobile(self, ws: WebSocket, node_id: str):
        self.mobile_sockets.discard(ws)
        self.active_nodes.pop(node_id, None)

    async def connect_host(self, ws: WebSocket):
        await ws.accept()
        self.host_sockets.add(ws)
        # Gui ngay trang thai cac dien thoai dang ket noi
        await ws.send_json({
            "type": "node_status",
            "nodes_count": len(self.active_nodes),
            "nodes": list(self.active_nodes.values()),
        })
        if self.last_result:
            await ws.send_json({"type": "result", "data": self.last_result})

    def disconnect_host(self, ws: WebSocket):
        self.host_sockets.discard(ws)

    async def broadcast_result(self, result_data: dict):
        self.last_result = result_data
        payload = {"type": "result", "data": result_data}
        
        # Gui cho Desktop Host
        for ws in list(self.host_sockets):
            try:
                await ws.send_json(payload)
            except Exception:
                self.host_sockets.discard(ws)

        # Gui cho Dien thoai
        for ws in list(self.mobile_sockets):
            try:
                await ws.send_json(payload)
            except Exception:
                self.mobile_sockets.discard(ws)

    async def broadcast_to_hosts(self, payload: dict):
        for ws in list(self.host_sockets):
            try:
                await ws.send_json(payload)
            except Exception:
                self.host_sockets.discard(ws)


def create_capture_app(token: str, web_root: Path, gateway_url: str = "http://localhost:3000") -> FastAPI:
    app = FastAPI(title="Rice Vision AI - Phone Camera Host", docs_url=None, redoc_url=None)
    hub = NodeHub()

    static_dir = web_root / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    def require_token(value: str | None) -> None:
        if not value or value != token:
            raise HTTPException(status_code=401, detail="Phien ket noi khong hop le hoac da het han.")

    @app.get("/capture", include_in_schema=False)
    async def capture_page(token_value: str = Query(alias="token")):
        require_token(token_value)
        return FileResponse(web_root / "templates" / "capture.html")

    @app.get("/api/health")
    async def health():
        return {
            "status": "ok",
            "connected_phones": len(hub.active_nodes),
            "gateway_url": gateway_url,
        }

    @app.post("/api/capture")
    async def handle_capture(
        file: UploadFile = File(...),
        diam: str = Form("0"),
        height: str = Form("0"),
        empty: str = Form("0"),
        wall_thickness: str = Form("0.1"),
        weight_total: str = Form("0"),
        sample_count: str = Form("0"),
        sample_weight: str = Form("0"),
        node_id: str = Form("mobile-node"),
        x_capture_token: str | None = Header(default=None),
    ):
        require_token(x_capture_token)

        try:
            image_bytes = await file.read()
            filename = file.filename or "capture.jpg"
            content_type = file.content_type or "image/jpeg"

            # Chuyen tiep request sang Gateway Express (hoac thang AI backend)
            predict_endpoint = f"{gateway_url.rstrip('/')}/api/predict"
            
            form_fields = {
                "diam": diam,
                "height": height,
                "empty": empty,
                "wall_thickness": wall_thickness,
                "weight_total": weight_total,
                "sample_count": sample_count,
                "sample_weight": sample_weight,
            }
            files = {
                "file": (filename, image_bytes, content_type)
            }

            async with httpx.AsyncClient(timeout=120.0) as client:
                res = await client.post(predict_endpoint, data=form_fields, files=files)
                
            if res.status_code != 200:
                error_msg = res.text
                try:
                    error_msg = res.json().get("error", res.text)
                except Exception:
                    pass
                err_payload = {"type": "error", "status": "error", "error": f"Loi tu AI Server: {error_msg}"}
                await hub.broadcast_result(err_payload)
                raise HTTPException(status_code=res.status_code, detail=f"Loi tu AI Server: {error_msg}")

            result_data = res.json()

            # Phat song ket qua dong thoi den ca Dien thoai va Desktop Host
            await hub.broadcast_result(result_data)

            return result_data

        except httpx.TimeoutException:
            err_payload = {"type": "error", "status": "error", "error": "AI Server timeout (>120s)"}
            await hub.broadcast_result(err_payload)
            raise HTTPException(status_code=504, detail="AI Server timeout (>120s). Vui long kiem tra ket noi Colab/Local.")
        except httpx.RequestError as exc:
            err_payload = {"type": "error", "status": "error", "error": f"Gateway connection error: {exc}"}
            await hub.broadcast_result(err_payload)
            raise HTTPException(status_code=502, detail=f"Khong the ket noi Gateway tai {gateway_url}: {exc}")
        finally:
            await file.close()

    @app.websocket("/ws/node")
    async def node_socket(websocket: WebSocket):
        token_value = websocket.query_params.get("token")
        if token_value != token:
            await websocket.close(code=1008)
            return

        node_id = (websocket.query_params.get("node_id") or "mobile-node")[:120]
        label = (websocket.query_params.get("label") or "Dien thoai")[:120]

        await hub.connect_mobile(websocket, node_id, label)
        try:
            while True:
                data = await websocket.receive_text()
                if data == "ping":
                    await websocket.send_text("pong")
        except WebSocketDisconnect:
            pass
        finally:
            hub.disconnect_mobile(websocket, node_id)
            await hub.broadcast_to_hosts({
                "type": "node_status",
                "nodes_count": len(hub.active_nodes),
                "nodes": list(hub.active_nodes.values()),
            })

    @app.websocket("/ws/host")
    async def host_socket(websocket: WebSocket):
        await hub.connect_host(websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            hub.disconnect_host(websocket)

    return app
