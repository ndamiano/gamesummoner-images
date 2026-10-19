from contract.comfy import ComfyServer
from contract.server import main
from local.server import LocalServer


class LocalComfyServer(LocalServer, ComfyServer):
    pass


if __name__ == "__main__":
    raise SystemExit(main(LocalComfyServer))
