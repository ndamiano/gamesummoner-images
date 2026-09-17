from contract.mesh import MeshServer
from contract.server import main
from local.server import LocalServer


class LocalMeshServer(LocalServer, MeshServer):
    pass


if __name__ == "__main__":
    raise SystemExit(main(LocalMeshServer))
