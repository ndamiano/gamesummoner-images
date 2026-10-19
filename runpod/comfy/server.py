from contract.comfy import ComfyServer
from contract.server import main
from runpod.server import RunpodServer


class RunpodComfyServer(RunpodServer, ComfyServer):
    pass


if __name__ == "__main__":
    raise SystemExit(main(RunpodComfyServer))
