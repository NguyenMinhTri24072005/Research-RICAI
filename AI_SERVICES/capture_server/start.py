import json
import sys
import time
from pathlib import Path

# Them thu muc cha vao sys.path de import duoc capture_server
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capture_server.controller import CaptureServerController


def main():
    port = 8765
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass

    controller = CaptureServerController(port=port)
    try:
        url = controller.start()
        qr_data = controller.make_qr_base64()

        # In JSON ra stdout de Express doc duoc
        output = {
            "status": "running",
            "port": port,
            "url": url,
            "qr": qr_data,
        }
        print("READY:" + json.dumps(output), flush=True)

        # Giu tien trinh song cho den khi nhan tin hieu dung
        while controller.running:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[CAPTURE SERVER] Dang tat server...", flush=True)
        controller.stop()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
