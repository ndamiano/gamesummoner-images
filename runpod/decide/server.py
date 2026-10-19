from contract.decide import DecideServer
from contract.server import main
from runpod.server import RunpodServer


class RunpodDecideServer(RunpodServer, DecideServer):
    pass


if __name__ == "__main__":
    raise SystemExit(main(RunpodDecideServer))
