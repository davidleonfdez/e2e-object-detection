import argparse
from pathlib import Path
import subprocess


PORTS = [8080, 8081, 8082, 7070, 7071]
# 0.7.0 doesn't work with ONNX because of a bug
TS_IMG_VERSION = "0.6.1-cpu" #"latest-cpu"


def _get_base_path():
    return Path(__file__).resolve().parent.parent


def run_with_subprocess(model_store_path):
    base_path = _get_base_path()
    ports_binding = ' '.join(f"-p {p}:{p}" for p in PORTS)
    cmd = (
        #"docker run --rm -it -p 8080:8080 -p 8081:8081 -p 8082:8082 -p 7070:7070 -p 7071:7071 " +
        f"docker run --rm -it {ports_binding} " +
        f"-v {model_store_path}:/home/model-server/model-store " +
        f"-v {str(base_path/'config.properties')}:/home/model-server/config.properties " +
        f"pytorch/torchserve:{TS_IMG_VERSION} torchserve --model-store model-store --start --models all --ncs "
    )
    print(cmd)
    subprocess.run(cmd.split(' '))


def run_with_docker_py(model_store_path):
    import docker
    client = docker.from_env()
    base_path = _get_base_path()

    container = client.containers.run(
        f"pytorch/torchserve:{TS_IMG_VERSION}",
        command="torchserve --model-store model-store --start --models all --ncs",
        volumes=[
            f"{model_store_path}:/home/model-server/model-store",
            f"{str(base_path/'config.properties')}:/home/model-server/config.properties",
        ],
        ports={p:p for p in PORTS},
        remove=True,
        detach=True,
        stdin_open=True,
        tty=True,
    )
    return container


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""Start TorchServe hosted in a local Docker container"""
    )
    parser.add_argument(
        'model_store_path', type=str, help='Path of the directory that contains the .mar model files'
    )
    args = parser.parse_args()
    #run_with_subprocess(args.model_store_path)
    run_with_docker_py(args.model_store_path)
