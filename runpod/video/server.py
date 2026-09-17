from contract.video import VideoServer
from contract.server import main
from runpod.server import RunpodServer


class RunpodVideoServer(RunpodServer, VideoServer):
    pass


if __name__ == "__main__":
    raise SystemExit(main(RunpodVideoServer))
