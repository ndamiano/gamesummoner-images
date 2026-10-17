from aws.server import AwsServer
from contract.decide import DecideServer
from contract.server import main


class AwsDecideServer(AwsServer, DecideServer):
    pass


if __name__ == "__main__":
    raise SystemExit(main(AwsDecideServer))
