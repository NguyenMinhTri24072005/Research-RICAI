from __future__ import annotations

import secrets
import socket
import threading
import time
from pathlib import Path

from rice_capture.server.app import create_mobile_app
from rice_capture.server.tls import TlsMaterial, ensure_local_certificates
from rice_capture.services.sample_service import CaptureCoordinator


def discover_lan_addresses() -> list[str]:
    addresses: list[str] = []
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            probe.connect(("8.8.8.8", 80))
            addresses.append(probe.getsockname()[0])
        finally:
            probe.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            address = info[4][0]
            if not address.startswith("127.") and not address.startswith("169.254."):
                addresses.append(address)
    except OSError:
        pass
    unique: list[str] = []
    for address in addresses:
        if address not in unique:
            unique.append(address)
    return unique or ["127.0.0.1"]


class MobileServerController:
    def __init__(
        self,
        coordinator: CaptureCoordinator,
        web_root: Path,
        port: int = 8765,
        cert_dir: Path | None = None,
        enable_https: bool = True,
    ):
        self.coordinator = coordinator
        self.web_root = web_root
        self.port = port
        self.cert_dir = cert_dir or web_root.parents[2] / "runtime" / "certificates"
        self.enable_https = enable_https
        self.token = ""
        self.server = None
        self.thread: threading.Thread | None = None
        self.tls_material: TlsMaterial | None = None
        self.bound_address = ""

    @property
    def running(self) -> bool:
        return bool(self.thread and self.thread.is_alive() and self.server and self.server.started)

    @property
    def ca_certificate_path(self) -> Path | None:
        return self.tls_material.ca_install_path if self.tls_material else None

    def prepare_certificates(self, address: str | None = None) -> TlsMaterial:
        hosts = discover_lan_addresses()
        if address and address not in hosts:
            hosts.append(address)
        self.tls_material = ensure_local_certificates(self.cert_dir, hosts)
        return self.tls_material

    def start(self, address: str | None = None) -> None:
        if self.thread and self.thread.is_alive():
            return
        try:
            import uvicorn
        except ImportError as exc:
            raise RuntimeError("Thiếu Uvicorn. Hãy chạy setup_capture_app.bat.") from exc
        self.token = secrets.token_urlsafe(24)
        app = create_mobile_app(self.coordinator, self.token, self.web_root)
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
        self.thread = threading.Thread(target=self.server.run, name="mobile-capture-server", daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            if self.server.started:
                return
            if not self.thread.is_alive():
                break
            time.sleep(0.05)
        self.stop()
        raise RuntimeError(f"Không thể mở máy chủ tại cổng {self.port}. Cổng có thể đang được sử dụng.")

    def connection_url(self, address: str | None = None) -> str:
        if not self.token:
            raise RuntimeError("Máy chủ chưa được khởi động.")
        host = address or discover_lan_addresses()[0]
        scheme = "https" if self.enable_https else "http"
        return f"{scheme}://{host}:{self.port}/capture?token={self.token}"

    def make_qr_image(self, address: str | None = None):
        try:
            import qrcode
        except ImportError as exc:
            raise RuntimeError("Thiếu thư viện qrcode. Hãy chạy setup_capture_app.bat.") from exc
        return qrcode.make(self.connection_url(address))

    def stop(self) -> None:
        if self.server is not None:
            self.server.should_exit = True
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=4)
        self.server = None
        self.thread = None
        self.token = ""
        self.bound_address = ""
