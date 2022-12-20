# End to end object detection

Single object detection using YOLO v7 tiny:
- Training notebook [train.ipynb](notebooks/train.ipynb)
- Client project [client/](client/) built with Streamlit
- Server project [server/](server/) built with TorchServe and AWS

1. Train a model on your preferred platform using the notebook and download the exported TorchScript or ONNX model.
2. Start the server locally or deploy to AWS EC2/ALB+ASG/ECS by running just one script (see [server/](server/)).
3. Start the client locally or deploy to Streamlit (see [client/](client/))

For more information and installation instructions, see the README.md in each subfolder.
