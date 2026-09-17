from contract.video import VideoServer
from contract.server import main
from local.server import LocalServer


class LocalVideoServer(LocalServer, VideoServer):
    pass


if __name__ == "__main__":
    raise SystemExit(main(LocalVideoServer))
