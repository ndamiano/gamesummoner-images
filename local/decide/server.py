from contract.decide import DecideServer
from contract.server import main
from local.server import LocalServer


class LocalDecideServer(LocalServer, DecideServer):
    pass


if __name__ == "__main__":
    raise SystemExit(main(LocalDecideServer))
