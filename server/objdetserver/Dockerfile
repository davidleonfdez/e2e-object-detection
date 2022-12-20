# syntax=docker/dockerfile:1
# 0.7.0 doesn't work with ONNX because of a bug
#FROM pytorch/torchserve:latest-cpu
FROM pytorch/torchserve:0.6.1-cpu

WORKDIR /home/model-server

# Needed for OpenCV
USER root
RUN apt-get update && apt-get install -y libgl1
USER model-server

COPY config.properties .

ARG mar_path
COPY $mar_path model-store

CMD ["torchserve", "--model-store model-store", "--start", "--models all", "--ncs"]
