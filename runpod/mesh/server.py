from contract.mesh import MeshServer
from contract.server import main
from runpod.server import RunpodServer


class RunpodMeshServer(RunpodServer, MeshServer):
    pass


if __name__ == "__main__":
    raise SystemExit(main(RunpodMeshServer))
