from contract.llm import LlmServer
from contract.server import main
from runpod.server import RunpodServer


class RunpodLlmServer(RunpodServer, LlmServer):
    pass


if __name__ == "__main__":
    raise SystemExit(main(RunpodLlmServer))
