# Requirements

- AWS CLI should be installed and configured on the computer where you execute the deployment script

# Commands

To deploy for the first time:

`python deploy.py <model-path(.pt or .onnx)>`

To deploy an existing infrastructure and avoid creating a new stack, you must pass the name of the existing CloudFormation stack:

`python deploy.py --stack-name <stack-name> <model-path(.pt or .onnx)>`
