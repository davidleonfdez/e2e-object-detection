import argparse
from pathlib import Path
import subprocess


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""Start TorchServe hosted in a local Docker container"""
    )
    parser.add_argument(
        'model_store_path', type=str, help='Path of the directory that contains the .mar model files'
    )
    args = parser.parse_args()

    base_path = Path(__file__).resolve().parent.parent
    cmd = (
        "docker run --rm -it -p 8080:8080 -p 8081:8081 -p 8082:8082 -p 7070:7070 -p 7071:7071 " +
        f"-v {args.model_store_path}:/home/model-server/model-store " +
        f"-v {str(base_path/'config.properties')}:/home/model-server/config.properties " +
        "pytorch/torchserve torchserve --model-store model-store --start --models all --ncs "
    )
    print(cmd)

    subprocess.run(cmd.split(' '))
