# Object detection model server

Object detection YOLO v7 model server built with TorchServe and deployable to AWS.

## Installation

`pip install --extra-index-url https://download.pytorch.org/whl/cpu -e .[dev]`

## Running the server locally

1. Generate the TorchServe model archive file (.mar) by running the script [generate_mar.py](objdetserver/scripts/generate_mar.py)
2. Start a Docker instance of the server by running the script the script [run_local_docker.py](objdetserver/scripts/run_local_docker.py)

Example:
```
python objdetserver/scripts/generate_mar.py /home/random_user/my_models/best.onnx
mkdir /home/random_user/model_store
mv object_detector.mar /home/random_user/model_store
python objdetserver/scripts/run_local_docker.py /home/random_user/model_store
```

# Deploying to AWS

For scripts and instructions on how to deploy to AWS, see the README files located at 
[objdetserver/deploy/ec2](objdetserver/deploy/ec2) and 
[objdetserver/deploy/ecs](objdetserver/deploy/ecs).
