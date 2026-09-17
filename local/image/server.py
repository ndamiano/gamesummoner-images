from contract.image import ImageServer
from contract.server import main
from local.server import LocalServer


class LocalImageServer(LocalServer, ImageServer):
    pass


if __name__ == "__main__":
    raise SystemExit(main(LocalImageServer))
