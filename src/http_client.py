"""Small standard-library HTTP helper with a macOS-friendly CA fallback."""
from __future__ import annotations

import os
import ssl
from pathlib import Path


def tls_context() -> ssl.SSLContext:
    configured = os.getenv("SSL_CERT_FILE")
    if configured:
        return ssl.create_default_context(cafile=configured)
    # The python.org macOS installer can omit this from Python's default trust
    # store even though the system bundle is present. Linux hosts normally use
    # their default context instead.
    system_bundle = Path("/etc/ssl/cert.pem")
    if system_bundle.is_file():
        return ssl.create_default_context(cafile=str(system_bundle))
    return ssl.create_default_context()
