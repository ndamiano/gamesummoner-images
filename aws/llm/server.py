from aws.server import AwsServer
from contract.llm import LlmServer
from contract.server import main


class AwsLlmServer(AwsServer, LlmServer):
    pass


if __name__ == "__main__":
    raise SystemExit(main(AwsLlmServer))
