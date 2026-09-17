from contract.image import ImageServer
from contract.server import main
from runpod.server import RunpodServer


class RunpodImageServer(RunpodServer, ImageServer):
    pass


if __name__ == "__main__":
    raise SystemExit(main(RunpodImageServer))
