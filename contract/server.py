import argparse
import json
import logging
import os
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

logger = logging.getLogger("image")

DECOMMISSION_DELAY_SECONDS = 1.0
PROBE_INTERVAL_SECONDS = 2.0


class Server:
    capabilities: tuple[str, ...] = ()
    kinds: tuple[str, ...] = ()

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--image-id", required=True,
                            help="the build's id, reported in /metadata")
        parser.add_argument("--cli-params", default="",
                            help="the engine's command line, reported in /metadata")
        parser.add_argument("--engine-pid", type=int, default=None,
                            help="engine process to watch: /health answers 500 once it exits")
        parser.add_argument("--entrypoint-pid", type=int, default=None,
                            help="the entrypoint to SIGTERM on decommission; it ends the box")
        parser.add_argument("--host", default="127.0.0.1")
        parser.add_argument("--port", type=int, default=8700)

    @classmethod
    def from_argv(cls, argv=None):
        parser = argparse.ArgumentParser(description=cls.__doc__)
        cls.add_arguments(parser)
        return cls(parser.parse_args(argv))

    def __init__(self, args: argparse.Namespace):
        self.metadata = {"imageId": args.image_id, "cliParams": args.cli_params,
                         "capabilities": list(self.capabilities)}
        self.engine_pid = args.engine_pid
        self.entrypoint_pid = args.entrypoint_pid
        self.bind = (args.host, args.port)
        self.leaving = False
        self._lock = threading.Lock()
        self._verdict = (503, "not probed yet")


    def ready(self) -> tuple[int, str]:
        raise NotImplementedError

    def run(self, job_type: str, payload: dict) -> tuple[int, dict]:
        raise NotImplementedError

    def decommission(self) -> None:
        if self.entrypoint_pid is None:
            logger.warning("no --entrypoint-pid: nothing ends the box")
            return
        os.kill(self.entrypoint_pid, signal.SIGTERM)


    def start(self) -> ThreadingHTTPServer:
        threading.Thread(target=self._probe_loop, daemon=True).start()
        return ThreadingHTTPServer(self.bind, make_handler(self))

    def _probe_loop(self) -> None:
        while not self.leaving:
            try:
                self._verdict = self.ready()
            except Exception as e:  # noqa: BLE001
                self._verdict = (503, f"readiness probe failed ({e.__class__.__name__})")
            time.sleep(PROBE_INTERVAL_SECONDS)

    def health(self) -> tuple[int, str]:
        if self.leaving:
            return 410, "decommissioning"
        if self.engine_pid is not None and not _alive(self.engine_pid):
            return 500, f"engine process {self.engine_pid} exited"
        return self._verdict

    def job(self, body: dict) -> tuple[int, dict]:
        job_type = body.get("type")
        if job_type not in self.kinds:
            return 500, {"status": "failed", "error": f"this image does not run {job_type!r} jobs"}
        started = time.perf_counter()
        status, out = self.run(job_type, body.get("payload") or {})
        logger.info("job %s (%s) -> %s in %.1fs", body.get("jobId"), job_type, status,
                    time.perf_counter() - started)
        return status, out

    def shutdown(self) -> None:
        with self._lock:
            if self.leaving:
                return
            self.leaving = True
        logger.info("decommission requested")

        def later():
            time.sleep(DECOMMISSION_DELAY_SECONDS)
            try:
                self.decommission()
            except Exception:  # noqa: BLE001
                logger.exception("decommission failed")

        threading.Thread(target=later, daemon=True).start()


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def make_handler(image: Server):
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


def main(server_cls: type[Server], argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
    image = server_cls.from_argv(argv)
    httpd = image.start()
    host, port = httpd.server_address[:2]
    logger.info("contract server on %s:%d, %s", host, port, image.metadata)
    httpd.serve_forever()
    return 0
