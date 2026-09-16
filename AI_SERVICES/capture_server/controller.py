from __future__ import annotations

import base64
import io
import secrets
import threading
import time
from pathlib import Path

from capture_server.app import create_capture_app
from capture_server.discovery import discover_lan_addresses
from capture_server.tls import TlsMaterial, ensure_local_certificates


class CaptureServerController:
    """Dieu khien vong doi cua Capture Server: tao TLS, QR code, start/stop uvicorn."""
    def __init__(
        self,
        web_root: Path | None = None,
        port: int = 8765,
        cert_dir: Path | None = None,
        enable_https: bool = True,
        gateway_url: str = "http://localhost:3000",
    ):
        self.web_root = web_root or Path(__file__).resolve().parent / "web"
        self.port = port
        self.cert_dir = cert_dir or Path(__file__).resolve().parent / "certificates"
        self.enable_https = enable_https
        self.gateway_url = gateway_url

        self.token = ""
        self.server = None
        self.thread: threading.Thread | None = None
        self.tls_material: TlsMaterial | None = None
        self.bound_address = ""

    @property
    def running(self) -> bool:
        return bool(self.thread and self.thread.is_alive() and self.server and self.server.started)

    def prepare_certificates(self, address: str | None = None) -> TlsMaterial:
        hosts = discover_lan_addresses()
        if address and address not in hosts:
            hosts.append(address)
        self.tls_material = ensure_local_certificates(self.cert_dir, hosts)
        return self.tls_material

    def start(self, address: str | None = None) -> str:
        if self.running:
            return self.connection_url(address)

        import uvicorn

        self.token = secrets.token_urlsafe(24)
        app = create_capture_app(self.token, self.web_root, self.gateway_url)

        config_values = {
            "app": app,
            "host": "0.0.0.0",
            "port": self.port,
            "log_level": "warning",
            "access_log": False,
        }

        if self.enable_https:
            material = self.prepare_certificates(address)
            config_values.update(
                ssl_certfile=str(material.server_cert_path),
                ssl_keyfile=str(material.server_key_path),
            )

        self.bound_address = address or ""
        config = uvicorn.Config(**config_values)
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, name="rice-capture-server", daemon=True)
        self.thread.start()

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if self.server.started:
                url = self.connection_url(address)
                print(f"[*] [CAPTURE SERVER] Da khoi dong thanh cong tai: {url}")
                return url
            if not self.thread.is_alive():
                break
            time.sleep(0.05)

        self.stop()
        raise RuntimeError(f"Khong the khoi dong Capture Server tai cong {self.port}.")

    def connection_url(self, address: str | None = None) -> str:
        if not self.token:
            raise RuntimeError("Capture Server chua duoc khoi dong.")
        host = address or discover_lan_addresses()[0]
        scheme = "https" if self.enable_https else "http"
        return f"{scheme}://{host}:{self.port}/capture?token={self.token}"

    def make_qr_base64(self, address: str | None = None) -> str:
        """Tao anh QR code dang base64 de truyen thang ve frontend web."""
        import qrcode

        url = self.connection_url(address)
        qr_img = qrcode.make(url)
        buffer = io.BytesIO()
        qr_img.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def stop(self) -> None:
        if self.server is not None:
            self.server.should_exit = True
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3)
        self.server = None
        self.thread = None
        self.token = ""
        self.bound_address = ""
