import argparse
import base64
import boto3
from botocore.exceptions import ClientError
import docker
import os
from pathlib import Path
import random
import shutil
import subprocess
import time
from objdetserver.handlers import constants
from objdetserver.deploy.constants import ecs as deploy_constants
from objdetserver.deploy.deploy_utils import stack_exists, wait_for_stack_creation, wait_for_stack_update


class AWSClientManager:
    def __init__(self):
        self._clients = dict()
        self._account_id = None
    
    def get(self, client_name:str):
        # Not using setdefault to avoid creating a client when not needed
        if client_name not in self._clients:
            self._clients[client_name] = boto3.client(client_name)
        return self._clients[client_name]

    @property
    def region_id(self) -> str:
        return self.get('cloudformation').meta.region_name

    @property
    def account_id(self) -> str:
        if self._account_id is None:
            sts = self.get('sts')
            self._account_id = sts.get_caller_identity()["Account"]
        return self._account_id


def generate_mar(model_name, model_path):
    scripts_path = Path(__file__).resolve().parent.parent.parent/constants.SCRIPTS_DIRNAME
    cmd = f"python {scripts_path/constants.GENERATE_MAR_SCRIPT_FILENAME} --model-name {model_name} {model_path}"
    subprocess.run(cmd.split())

    mar_path = Path(f'./{model_name}.mar')
    return mar_path


def build_image(docker_client:docker.DockerClient, image_name, mar_path):
    root_path = Path(__file__).resolve().parent.parent.parent
    mar_path = mar_path.resolve()
    cwd = os.getcwd()

    # Ensure .mar is inside docker build context
    if mar_path.parent != root_path:
        shutil.move(str(mar_path), str(root_path))
    mar_path = mar_path.name

    try:
        # Change directory to the one that contains the Dockerfile.
        # The Dockerfile isn't located in the same dir as this script because it can't exec
        # COPY from parent directories (outside of the build context) (like "COPY ../a.txt .")
        os.chdir(root_path)

        docker_client.images.build(
            path=str(root_path),
            tag=image_name,
            buildargs = {deploy_constants.DOCKERFILE_MAR_PATH_ARG: str(mar_path)},
            rm=True,
        )
    finally:
        os.chdir(cwd)


def get_ecr_registry(aws_client:AWSClientManager) -> str:
    return f"{aws_client.account_id}.dkr.ecr.{aws_client.region_id}.amazonaws.com"


def get_ecr_image_name(aws_client:AWSClientManager, image_name:str) -> str:
    return f"{get_ecr_registry(aws_client)}/{image_name}"


def log_into_ecr_registry(aws_client:AWSClientManager, docker_client:docker.DockerClient):
    ecr_client = aws_client.get('ecr')
    token = ecr_client.get_authorization_token()
    username, password = base64.b64decode(token['authorizationData'][0]['authorizationToken']).decode('utf-8').split(":")

    docker_client.login(username=username, password=password, registry=get_ecr_registry(aws_client))


def maybe_create_ecr_repo(ecr_client, repo_name:str):
    should_create = False
    try:
        repositories = ecr_client.describe_repositories(repositoryNames=[repo_name])
    except ecr_client.exceptions.RepositoryNotFoundException:
        should_create = True
    # Second predicate is useless at the time of writing: it should never happen when (should_create == False), 
    # because the exception is raised when the repo doesn't exist; still, this is more robust in case boto3 behaviour 
    # changes.
    should_create = should_create or (len(repositories) == 0)
    if should_create:
        ecr_client.create_repository(repositoryName=repo_name)


def tag_image_for_ecr(aws_client:AWSClientManager, docker_client:docker.DockerClient, image_name:str):
    target_image_name = get_ecr_image_name(aws_client, image_name)

    docker_client.api.tag(image_name, target_image_name)
    docker_client.images.push(target_image_name)


def deploy_stack(client, stack_name:str, ecr_image_name:str, instance_type:str):
    if not stack_name:
        stack_name = f'objdet-ecs-stack-{random.randint(1e6, 9_999_999)}'
        is_new_stack = True
    else:
        is_new_stack = not stack_exists(client, stack_name)
    
    template_path = Path(__file__).resolve().with_name(deploy_constants.TEMPLATE_FILENAME)
    
    # boto3 cloudformation client doesn't have a deploy method, so we use CLI.
    # The boto3 alternative would be to:
    # 1. create change set
    # 2. wait for change set creation
    # 3. execute change set
    # 4. wait for stack creation or update
    cmd = [
        "aws", "cloudformation", "deploy", "--template-file", str(template_path), "--stack-name", stack_name,
        "--capabilities", "CAPABILITY_IAM",  "--parameter-overrides", f"ImageName={ecr_image_name}",
        f"InstanceType={instance_type}", "--no-fail-on-empty-changeset"
    ]

    subprocess.run(cmd)

    if not is_new_stack:
        # Wait a bit to ensure deploy has updated the state before waiting for stack update
        time.sleep(1.0)
        wait_for_stack_update(client, stack_name)
    else:
        wait_for_stack_creation(client, stack_name)

    return stack_name


def deploy(args):
    docker_client = docker.from_env()
    aws_client = AWSClientManager()
    ecr_client = aws_client.get('ecr')

    print('Generating .mar file...')
    mar_path = generate_mar(args.model_name, args.model_path)
    if not mar_path.exists():
        raise RuntimeError('Failed to generate .mar file')
    print('...Generated .mar file')

    print('Building Docker image...')
    build_image(docker_client, args.image_name, mar_path)
    print('...Built Docker image')

    print('Setting up ECR repository...')
    log_into_ecr_registry(aws_client, docker_client)
    maybe_create_ecr_repo(ecr_client, args.image_name)
    print('...Finished setting up ECR repository')

    print('Pushing image to ECR repository...')
    tag_image_for_ecr(aws_client, docker_client, args.image_name)
    print('...Pushed image to ECR repository')

    print('Deploying CloudFormation stack...')
    stack_name = deploy_stack(
        aws_client.get('cloudformation'), 
        args.stack_name, 
        get_ecr_image_name(aws_client, args.image_name),
        args.instance_type
    )
    print(f'...Deployed CloudFormation stack {stack_name}')


if __name__ == '__main__':    
    parser = argparse.ArgumentParser(
        description="""Deploy the model `model_path` to TorchServe instance(s) hosted in an ECS cluster with EC2 launch type"""
    )
    parser.add_argument(
        'model_path', type=str, help='Path of the input model file (.torchscript.pt or .onnx)'
    )
    parser.add_argument(
        '--stack-name', type=str, default='', help='Name of the CloudFormation stack',
    )
    parser.add_argument(
        '--model-name', type=str, default='object_detector', help='Name of the model endpoint',
    )
    parser.add_argument(
        '--image-name', type=str, default='objdet-server', help='Name of the Docker image (repository) deployed to ECR',
    )
    parser.add_argument(
        '--instance-type', type=str, default='t2.micro', help='EC2 instance type',
    )
    args = parser.parse_args()

    deploy(args)
