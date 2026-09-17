"""The image side of the worker contract: /health, /metadata, /job, /shutdown.

An adapter supplies the engine-specific parts (readiness, running one job); the provider's
entrypoint supplies the decommission command. Standard library only, so it runs on whatever
python the engine's base image ships.

    python3 contract/server.py --adapter llm --engine-url http://127.0.0.1:30000 \
        --engine-pid 123 --capability llm --cli-params "..." --decommission "kill -TERM 1"
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import llm  # noqa: E402

logger = logging.getLogger("image")

ADAPTERS = {"llm": llm.LlmAdapter}
DECOMMISSION_DELAY_SECONDS = 1.0


class Image:
    def __init__(self, adapter, metadata: dict, decommission: str, engine_pid: int | None):
        self.adapter = adapter
        self.metadata = metadata
        self.decommission = decommission
        self.engine_pid = engine_pid
        self.leaving = False
        self._lock = threading.Lock()

    def health(self) -> tuple:
        if self.leaving:
            return 410, "decommissioning"
        if self.engine_pid is not None and not _alive(self.engine_pid):
            return 500, f"engine process {self.engine_pid} exited"
        ready, reason = self.adapter.ready()
        return (200, reason or "ready") if ready else (503, reason)

    def job(self, body: dict) -> tuple:
        job_type = body.get("type")
        if job_type not in self.metadata["capabilities"]:
            return 500, {"status": "failed", "error": f"this image does not run {job_type!r} jobs"}
        started = time.perf_counter()
        status, out = self.adapter.run(job_type, body.get("payload") or {})
        logger.info("job %s (%s) -> %s in %.1fs", body.get("jobId"), job_type, status,
                    time.perf_counter() - started)
        return status, out

    def shutdown(self) -> None:
        with self._lock:
            if self.leaving:
                return
            self.leaving = True
        logger.info("decommission requested — running %r", self.decommission)

        def later():
            time.sleep(DECOMMISSION_DELAY_SECONDS)
            subprocess.run(self.decommission, shell=True, check=False)

        threading.Thread(target=later, daemon=True).start()


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def make_handler(image: Image):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, status: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                status, reason = image.health()
                self._send(status, {"reason": reason})
            elif self.path == "/metadata":
                self._send(200, image.metadata)
            else:
                self._send(404, {"error": f"no route {self.path}"})

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            if self.path == "/job":
                try:
                    body = json.loads(raw or b"{}")
                except json.JSONDecodeError as e:
                    self._send(500, {"status": "failed", "error": f"unreadable job: {e}"})
                    return
                status, out = image.job(body)
                self._send(status, out)
            elif self.path == "/shutdown":
                image.shutdown()
                self._send(202, {"reason": "decommissioning"})
            else:
                self._send(404, {"error": f"no route {self.path}"})

        def log_message(self, fmt, *args):
            if self.path != "/health":
                logger.info("%s %s", self.address_string(), fmt % args)

    return Handler


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="GameSummoner image contract server")
    parser.add_argument("--adapter", required=True, choices=sorted(ADAPTERS))
    parser.add_argument("--engine-url", required=True)
    parser.add_argument("--engine-pid", type=int, default=None)
    parser.add_argument("--capability", action="append", required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--cli-params", default="")
    parser.add_argument("--decommission", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8700)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
    metadata = {"imageId": args.image_id, "cliParams": args.cli_params,
                "capabilities": args.capability}
    image = Image(ADAPTERS[args.adapter](args.engine_url), metadata, args.decommission,
                  args.engine_pid)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(image))
    logger.info("contract server on %s:%d, %s", args.host, args.port, metadata)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
