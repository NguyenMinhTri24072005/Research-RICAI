from __future__ import annotations

import socket


def discover_lan_addresses() -> list[str]:
    """Phat hien danh sach dia chi IP mang LAN cua may tinh."""
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
