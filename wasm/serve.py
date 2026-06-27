#!/usr/bin/env python3
"""
Local HTTPS development server for WebGPU WebAssembly examples.

WebGPU requires a secure context (HTTPS or localhost). This script
generates a self-signed certificate and starts an HTTPS server so
examples can be tested locally without deploying to a real server.

Usage:
    python3 wasm/serve.py                   # serve dist/wasm/ on port 8443
    python3 wasm/serve.py --port 9000       # custom port
    python3 wasm/serve.py --dir dist/wasm   # custom directory
"""

import argparse
import os
import ssl
import sys
import tempfile
from http.server import HTTPServer, SimpleHTTPRequestHandler

# ---------------------------------------------------------------------------
# Self-signed certificate generation
# ---------------------------------------------------------------------------

def _generate_self_signed_cert(cert_path: str, key_path: str) -> None:
    """Generate a minimal self-signed certificate using the cryptography library."""
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        import datetime

        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, u"localhost"),
        ])
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.datetime.utcnow())
            .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=365))
            .add_extension(
                x509.SubjectAlternativeName([x509.DNSName(u"localhost")]),
                critical=False,
            )
            .sign(key, hashes.SHA256())
        )

        with open(key_path, "wb") as f:
            f.write(key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            ))
        with open(cert_path, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))

    except ImportError:
        print("ERROR: 'cryptography' package is required.")
        print("       Install it with: pip install cryptography")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Request handler — add required headers for SharedArrayBuffer + WebGPU
# ---------------------------------------------------------------------------

class _COEPHandler(SimpleHTTPRequestHandler):
    """Serve files with Cross-Origin headers required by WebGPU / SAB."""

    def end_headers(self) -> None:
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()

    def log_message(self, fmt, *args):  # noqa: N802
        # Suppress the per-request noise; only errors are interesting.
        if args and str(args[1]) not in ("200", "304"):
            super().log_message(fmt, *args)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_dir = os.path.join(os.path.dirname(script_dir), "dist", "wasm")

    parser = argparse.ArgumentParser(
        description="HTTPS dev server for WebGPU WASM examples"
    )
    parser.add_argument("--port", type=int, default=8443, help="TCP port (default 8443)")
    parser.add_argument("--dir", default=default_dir,
                        help="Directory to serve (default: dist/wasm/)")
    args = parser.parse_args()

    serve_dir = os.path.abspath(args.dir)
    if not os.path.isdir(serve_dir):
        print(f"ERROR: serve directory does not exist: {serve_dir}")
        print(f"       Build the WebAssembly examples first:")
        print(f"         cmake -S wasm -B build/wasm")
        print(f"         cmake --build build/wasm")
        sys.exit(1)

    os.chdir(serve_dir)

    # Use a persistent cert so the browser only needs to be accepted once.
    cert_dir = os.path.join(script_dir, ".cert")
    os.makedirs(cert_dir, exist_ok=True)
    cert_path = os.path.join(cert_dir, "cert.pem")
    key_path  = os.path.join(cert_dir, "key.pem")

    if not (os.path.exists(cert_path) and os.path.exists(key_path)):
        print("Generating self-signed certificate …")
        _generate_self_signed_cert(cert_path, key_path)

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert_path, key_path)

    server = HTTPServer(("", args.port), _COEPHandler)
    server.socket = ctx.wrap_socket(server.socket, server_side=True)

    url = f"https://localhost:{args.port}/"
    print(f"Serving {serve_dir}")
    print(f"Open  {url}  in a WebGPU-capable browser.")
    print("Accept the self-signed certificate warning once (Advanced → Proceed).")
    print("Press Ctrl-C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
