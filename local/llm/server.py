from contract.llm import LlmServer
from contract.server import main
from local.server import LocalServer


class LocalLlmServer(LocalServer, LlmServer):
    pass


if __name__ == "__main__":
    raise SystemExit(main(LocalLlmServer))
