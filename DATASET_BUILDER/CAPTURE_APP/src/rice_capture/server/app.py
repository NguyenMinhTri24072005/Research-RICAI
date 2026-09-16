import asyncio
from pathlib import Path

from rice_capture.services.sample_service import CaptureCoordinator


def create_mobile_app(coordinator: CaptureCoordinator, token: str, web_root: Path):
    try:
        from fastapi import FastAPI, File, Form, Header, HTTPException, Query, UploadFile, WebSocket
        from fastapi.responses import FileResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:
        raise RuntimeError("Thiếu FastAPI. Hãy chạy setup_capture_app.bat.") from exc

    app = FastAPI(title="Rice Capture Node API", docs_url=None, redoc_url=None)
    static_dir = web_root / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    def require_token(value: str | None) -> None:
        if not value or value != token:
            raise HTTPException(status_code=401, detail="Phiên kết nối không hợp lệ hoặc đã hết hạn.")

    @app.get("/capture", include_in_schema=False)
    async def capture_page(token_value: str = Query(alias="token")):
        require_token(token_value)
        return FileResponse(web_root / "templates" / "capture.html")

    @app.get("/api/health")
    async def health():
        return {"status": "ok"}

    @app.get("/api/session")
    async def session(x_capture_token: str | None = Header(default=None)):
        require_token(x_capture_token)
        return coordinator.session_payload()

    @app.post("/api/samples")
    async def save_sample(
        image: UploadFile = File(...),
        Weight_g: str = Form(...),
        Container_Height_mm: str = Form(...),
        Inner_Diameter_mm: str = Form(...),
        Empty_Height_mm: str = Form(...),
        Actual_Count: str = Form(...),
        node_id: str = Form(...),
        request_id: str = Form(...),
        captured_at: str = Form(default=""),
        x_capture_token: str | None = Header(default=None),
    ):
        require_token(x_capture_token)
        try:
            payload = await image.read(30 * 1024 * 1024 + 1)
            result = coordinator.save_jpeg(
                payload,
                {
                    "Weight_g": Weight_g,
                    "Container_Height_mm": Container_Height_mm,
                    "Inner_Diameter_mm": Inner_Diameter_mm,
                    "Empty_Height_mm": Empty_Height_mm,
                    "Actual_Count": Actual_Count,
                },
                source_type="mobile",
                node_id=node_id[:120],
                request_id=request_id[:120],
                captured_at=captured_at[:80],
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Không thể lưu mẫu: {exc}") from exc
        finally:
            await image.close()
        return {
            "sample_id": result.sample_id,
            "next_sample_id": result.next_sample_id,
            "duplicate_request": result.duplicate_request,
        }

    @app.websocket("/ws/node")
    async def node_socket(websocket: WebSocket):
        token_value = websocket.query_params.get("token")
        if token_value != token:
            await websocket.close(code=1008)
            return
        node_id = (websocket.query_params.get("node_id") or "mobile-node")[:120]
        label = (websocket.query_params.get("label") or "Điện thoại")[:120]
        await websocket.accept()
        coordinator.notify_node("node_connected", node_id, label)
        try:
            await websocket.send_json({"type": "session", **coordinator.session_payload()})
            while True:
                try:
                    await asyncio.wait_for(websocket.receive_text(), timeout=10)
                except asyncio.TimeoutError:
                    await websocket.send_json({"type": "session", **coordinator.session_payload()})
        except Exception:
            pass
        finally:
            coordinator.notify_node("node_disconnected", node_id, label)

    return app
