import socket
import subprocess
import sys
import time

import pytest

from surface.playwright_surface import PlaywrightSurface


def _wait_for_port(host: str, port: int, timeout: float = 15.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.2)
    raise RuntimeError(f"Server on {host}:{port} did not start in time")


@pytest.fixture(scope="session")
def fake_bank_url():
    port = 8011
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for_port("127.0.0.1", port)
        yield f"http://127.0.0.1:{port}"
    finally:
        proc.terminate()
        proc.wait(timeout=5)


@pytest.fixture()
def surface(fake_bank_url):
    s = PlaywrightSurface(base_url=fake_bank_url, headless=True)
    try:
        yield s
    finally:
        s.close()
