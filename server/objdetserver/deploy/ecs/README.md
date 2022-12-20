# Deploy to ECS cluster with EC2 launch type

It includes:
- ECS Cluster with autoscaling of ECS tasks, each one containing a running TorchServe model server.
- User managed infrastructure: autoscaling group of EC2 instances fronted by an application load balancer and distributed
  across two availability zones.
  - Provisioned by an ECS capacity provider with 100% target capacity.
- Custom VPC.

gRPC is not supported for now.

## Requirements

- AWS CLI should be installed and configured on the computer where you execute the deployment script.
- Install the development environment from the project root (server/):

    `pip install --extra-index-url https://download.pytorch.org/whl/cpu -e .[dev]`

  If you have any issues regarding compatibility of library versions, install the locked development requirements (the 
  other requirements.txt contains exclusively the packages needed for the EC2 server), from the project root (server/):
  ```
  pip install -r dev-requirements.txt
  ```

## Commands

```
deploy.py [-h] [--stack-name STACK_NAME] [--model-name MODEL_NAME]
          [--image-name IMAGE_NAME] [--instance-type INSTANCE_TYPE]
          model_path
```

- To deploy for the first time:

  `python deploy.py <model-path(.torchscript.pt or .onnx)>`

- To deploy modifying an existing infrastructure and avoid creating a new stack, you must pass the name of the existing CloudFormation stack:

  `python deploy.py --stack-name <stack-name> <model-path(.torchscript.pt or .onnx)>`
