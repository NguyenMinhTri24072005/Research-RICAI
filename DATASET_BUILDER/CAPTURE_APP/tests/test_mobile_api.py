from __future__ import annotations

import io
import json
import queue
import socket
import ssl
import sys
import tempfile
import unittest
import urllib.request
import uuid
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR / "src"))

from PIL import Image

from rice_capture.server import MobileServerController
from rice_capture.services import CaptureCoordinator


def available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def make_jpeg() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (960, 720), "#cab785").save(output, "JPEG", quality=92)
    return output.getvalue()


def multipart_body(fields: dict[str, str], image: bytes) -> tuple[bytes, str]:
    boundary = f"----RiceCapture{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                str(value).encode(),
                b"\r\n",
            ]
        )
    chunks.extend(
        [
            f"--{boundary}\r\n".encode(),
            b'Content-Disposition: form-data; name="image"; filename="capture.jpg"\r\n',
            b"Content-Type: image/jpeg\r\n\r\n",
            image,
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    return b"".join(chunks), boundary


class MobileApiTests(unittest.TestCase):
    def test_health_session_page_and_upload(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            events: queue.Queue = queue.Queue()
            coordinator = CaptureCoordinator(events)
            coordinator.configure(
                image_dir=root / "images",
                database_path=root / "capture.sqlite3",
                sample_number=1,
                sample_digits=3,
                manual_defaults={"Weight_g": "100"},
            )
            server = MobileServerController(
                coordinator,
                APP_DIR / "src" / "rice_capture" / "web",
                port=available_port(),
                cert_dir=root / "certificates",
            )
            server.start("127.0.0.1")
            try:
                base = f"https://127.0.0.1:{server.port}"
                context = ssl.create_default_context(cafile=str(server.tls_material.ca_cert_path))
                with urllib.request.urlopen(f"{base}/api/health", timeout=5, context=context) as response:
                    self.assertEqual(json.load(response)["status"], "ok")

                page_url = f"{base}/capture?token={server.token}"
                with urllib.request.urlopen(page_url, timeout=5, context=context) as response:
                    page = response.read()
                    self.assertIn(b"Rice Dataset Capture", page)
                    self.assertIn(b'id="cameraVideo"', page)
                    self.assertIn(b'id="aspectRatio"', page)
                    self.assertIn(b'id="flashToggle"', page)

                session_request = urllib.request.Request(
                    f"{base}/api/session",
                    headers={"X-Capture-Token": server.token},
                )
                with urllib.request.urlopen(session_request, timeout=5, context=context) as response:
                    self.assertEqual(json.load(response)["next_sample_id"], "M001")

                body, boundary = multipart_body(
                    {
                        "Weight_g": "120",
                        "Container_Height_mm": "100",
                        "Inner_Diameter_mm": "60",
                        "Empty_Height_mm": "20",
                        "Actual_Count": "500",
                        "node_id": "test-phone",
                        "request_id": "api-request-1",
                        "captured_at": "2026-09-11T10:00:00",
                    },
                    make_jpeg(),
                )
                upload_request = urllib.request.Request(
                    f"{base}/api/samples",
                    data=body,
                    method="POST",
                    headers={
                        "X-Capture-Token": server.token,
                        "Content-Type": f"multipart/form-data; boundary={boundary}",
                    },
                )
                with urllib.request.urlopen(upload_request, timeout=10, context=context) as response:
                    result = json.load(response)
                self.assertEqual(result["sample_id"], "M001")
                self.assertEqual(result["next_sample_id"], "M002")
                self.assertTrue((root / "images" / "M001.jpg").exists())
                self.assertTrue(server.ca_certificate_path.exists())
            finally:
                server.stop()


if __name__ == "__main__":
    unittest.main()
