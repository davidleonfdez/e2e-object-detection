# syntax=docker/dockerfile:1
FROM pytorch/torchserve:latest-cpu

WORKDIR /home/model-server

# Needed for OpenCV
USER root
RUN apt-get update && apt-get install -y libgl1
USER model-server

COPY config.properties .

ARG mar_path
COPY $mar_path model-store

CMD ["torchserve", "--model-store model-store", "--start", "--models all", "--ncs"]
