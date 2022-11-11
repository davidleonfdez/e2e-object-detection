#!/bin/bash
s3_models_bucket="obj-det"

sudo yum update -y
# Needed for TorchServe
sudo yum install -y java-17-amazon-corretto-headless

pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
pip3 install torchserve captum

# Download model file (.mar) from S3 to EC2 instance
mkdir models
aws s3api get-object --bucket $s3_models_bucket --key object_detector.mar models/object_detector.mar

# Create TorchServe configuration file and start server
echo -e "inference_address=http://0.0.0.0:8080\ninstall_py_dep_per_model=true" > config.properties
torchserve --model-store models --models all --ncs --start
