from __future__ import annotations

import ipaddress
import os
import socket
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class TlsMaterial:
    ca_cert_path: Path
    ca_install_path: Path
    server_cert_path: Path
    server_key_path: Path


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _certificate_expiry(certificate) -> datetime:
    value = getattr(certificate, "not_valid_after_utc", None)
    if value is not None:
        return value
    return certificate.not_valid_after.replace(tzinfo=timezone.utc)


def _write_file(path: Path, payload: bytes, private: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    if private:
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass


def _load_or_create_ca(cert_dir: Path):
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
    except ImportError as exc:
        raise RuntimeError(
            "Thiếu thư viện cryptography để tạo HTTPS. Hãy chạy lại setup_capture_app.bat."
        ) from exc

    key_path = cert_dir / "rice-capture-ca-key.pem"
    cert_path = cert_dir / "rice-capture-ca.pem"
    install_path = cert_dir / "RiceCapture-CA.cer"
    if key_path.exists() and cert_path.exists():
        try:
            key = serialization.load_pem_private_key(key_path.read_bytes(), password=None)
            certificate = x509.load_pem_x509_certificate(cert_path.read_bytes())
            if _certificate_expiry(certificate) > _utc_now() + timedelta(days=60):
                if not install_path.exists():
                    _write_file(install_path, certificate.public_bytes(serialization.Encoding.DER))
                return key, certificate, cert_path, install_path
        except (OSError, ValueError, TypeError):
            pass

    key = rsa.generate_private_key(public_exponent=65537, key_size=3072)
    subject = x509.Name(
        [
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Rice Dataset Capture"),
            x509.NameAttribute(NameOID.COMMON_NAME, "Rice Dataset Capture Local CA"),
        ]
    )
    now = _utc_now()
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=3650))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        .sign(key, hashes.SHA256())
    )
    _write_file(
        key_path,
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ),
        private=True,
    )
    _write_file(cert_path, certificate.public_bytes(serialization.Encoding.PEM))
    _write_file(install_path, certificate.public_bytes(serialization.Encoding.DER))
    return key, certificate, cert_path, install_path


def ensure_local_certificates(cert_dir: Path, addresses: Iterable[str]) -> TlsMaterial:
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
    except ImportError as exc:
        raise RuntimeError(
            "Thiếu thư viện cryptography để tạo HTTPS. Hãy chạy lại setup_capture_app.bat."
        ) from exc

    cert_dir.mkdir(parents=True, exist_ok=True)
    ca_key, ca_certificate, ca_cert_path, ca_install_path = _load_or_create_ca(cert_dir)
    ip_addresses = {ipaddress.ip_address("127.0.0.1")}
    dns_names = {"localhost", socket.gethostname()}
    for address in addresses:
        value = str(address).strip()
        if not value:
            continue
        try:
            ip_addresses.add(ipaddress.ip_address(value))
        except ValueError:
            dns_names.add(value)

    server_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name(
        [
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Rice Dataset Capture"),
            x509.NameAttribute(NameOID.COMMON_NAME, "Rice Capture Local Server"),
        ]
    )
    alternative_names = [x509.IPAddress(value) for value in sorted(ip_addresses, key=str)]
    alternative_names.extend(x509.DNSName(value) for value in sorted(dns_names))
    now = _utc_now()
    server_certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(ca_certificate.subject)
        .public_key(server_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=397))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.SubjectAlternativeName(alternative_names), critical=False)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=True,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .sign(ca_key, hashes.SHA256())
    )

    server_key_path = cert_dir / "rice-capture-server-key.pem"
    server_cert_path = cert_dir / "rice-capture-server.pem"
    _write_file(
        server_key_path,
        server_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ),
        private=True,
    )
    _write_file(server_cert_path, server_certificate.public_bytes(serialization.Encoding.PEM))
    return TlsMaterial(
        ca_cert_path=ca_cert_path,
        ca_install_path=ca_install_path,
        server_cert_path=server_cert_path,
        server_key_path=server_key_path,
    )
