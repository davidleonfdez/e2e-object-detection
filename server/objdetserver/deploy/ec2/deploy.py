import argparse
import boto3
import logging
from pathlib import Path
import random
import subprocess
import time
from objdetserver.deploy.constants import ec2 as deploy_constants
from objdetserver.deploy.deploy_utils import stack_exists, wait_for_stack_creation, wait_for_stack_update
from objdetserver.handlers import constants
from objdetserver.logging import configure_deploy_ec2_logger


configure_deploy_ec2_logger()
logger = logging.getLogger(__name__)


def generate_user_data(model_name, include_s3_model_download:bool):
    cts = deploy_constants
    out_path = Path(__file__).with_name(cts.USER_DATA_OUT_FILENAME)
    script_path = Path(__file__).with_name(cts.GENERATE_USER_DATA_SCRIPT_FILENAME)
    bucket_name_placeholder = cts.BUCKET_NAME_PLACEHOLDER_IN_USER_DATA if include_s3_model_download else ''
    cmd = [
        "python", str(script_path), "--bucket-name", bucket_name_placeholder, "--model-name", model_name,
        "--output-path", str(out_path)
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)

    with open(out_path) as f:
        user_data = f.read()
    return user_data


def deploy_stack(client, stack_name:str, test:bool, user_data:str, asg:bool, instance_type:str):
    if not stack_name:
        stack_name = f'objdet-stack-{random.randint(1e6, 9_999_999)}'
        is_new_stack = True
    else:
        is_new_stack = not stack_exists(client, stack_name)
    
    template_filename = deploy_constants.TEMPLATE_FILENAME_ASG if asg else deploy_constants.TEMPLATE_FILENAME
    template_path = Path(__file__).resolve().with_name(template_filename)
    
    # boto3 cloudformation client doesn't have a deploy method, so we use CLI.
    # The boto3 alternative would be to:
    # 1. create change set
    # 2. wait for change set creation
    # 3. execute change set
    # 4. wait for stack creation or update
    cmd = [
        "aws", "cloudformation", "deploy", "--template-file", str(template_path), "--stack-name", stack_name,
        "--capabilities", "CAPABILITY_IAM", "--parameter-overrides", f"UserDataParam={user_data}",
        f"InstanceTypeParameter={instance_type}", "--no-fail-on-empty-changeset"
    ]
    
    if test:
        cmd.append("--no-execute-changeset")

    subprocess.run(cmd)

    if not test:
        if not is_new_stack:
            # Wait a bit to ensure deploy has updated the state before waiting for stack update
            time.sleep(1.0)
            wait_for_stack_update(client, stack_name)
        else:
            wait_for_stack_creation(client, stack_name)

    return stack_name


def get_stack_output(client, stack_name):
    outputs_list = client.describe_stacks(StackName=stack_name)['Stacks'][0]['Outputs']
    return {output['OutputKey']: output['OutputValue'] for output in outputs_list}


def set_bucket_notification(bucket_name, lambda_arn):
    """Make the models S3 bucket invoke a Lambda after a new model is created/updated.
    
    The Lambda function is defined inside the CloudFormation stack and it's in
    charge of running an SSM document that copies the model from the S3 bucket to
    the instance and restarts TorchServe.
    The notification configuration can't be defined in the CloudFormation stack because
    it must be defined inside the S3 bucket resource, which would result in a circular 
    reference between the bucket and the Lambda function.
    """
    s3_client = boto3.client('s3')
    response = s3_client.put_bucket_notification_configuration(
        Bucket = bucket_name,
        NotificationConfiguration = {
            'LambdaFunctionConfigurations': [{
                'LambdaFunctionArn': lambda_arn,
                'Events': ['s3:ObjectCreated:*'],
            }]
        }
    )


def copy_mar_to_s3(local_mar_path, bucket_name, model_name):
    cmd = f"aws s3 cp {local_mar_path} s3://{bucket_name}/{model_name}.mar"
    subprocess.run(cmd.split())


def generate_mar(model_name, model_path):
    scripts_path = Path(__file__).resolve().parent.parent.parent/constants.SCRIPTS_DIRNAME
    generate_mar_script_path = scripts_path/constants.GENERATE_MAR_SCRIPT_FILENAME
    cmd = f"python {generate_mar_script_path} --model-name {model_name} {model_path}"
    subprocess.run(cmd.split())

    mar_path = Path(f'./{model_name}.mar')
    return mar_path


def deploy(args):
    logger.info(f'Running EC2 deployment script with args = {args}')

    client = boto3.client('cloudformation')

    logger.info('Generating UserData script...')
    # For a single EC2 instance, the user data script doesn't need to include code to download
    # the model from S3, as the copy will be triggered when we copy the model to S3 after the
    # stack has been deployed.
    user_data = generate_user_data(args.model_name, include_s3_model_download=args.asg)
    logger.info('...Generated UserData script')

    logger.info('Deploying CloudFormation stack...')
    stack_name = deploy_stack(client, args.stack_name, args.test, user_data, args.asg, args.instance_type)
    logger.info(f'...Deployed CloudFormation stack {stack_name}')

    if not args.test:
        logger.info('Setting S3 notifications...')
        stack_out = get_stack_output(client, stack_name)
        bucket_name = stack_out[deploy_constants.BUCKET_NAME_STACK_OUT]
        set_bucket_notification(bucket_name, stack_out[deploy_constants.LAMBDA_ARN_STACK_OUT])
        logger.info('...Finished setting S3 notifications')

        logger.info('Generating .mar file...')
        mar_path = generate_mar(args.model_name, args.model_path)
        logger.info('...Generated .mar file')

        logger.info('Copying .mar file to S3...')
        copy_mar_to_s3(mar_path, bucket_name, args.model_name)
        logger.info('...Copied .mar file to S3')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""Deploy the model `model_path` to a TorchServe instance hosted in a single EC2 instance"""
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
        '--test', 
        action='store_true', 
        default=False, 
        help=(
            'Only test (not execute) the infrastructure changes. This option causes the creation of a CloudFormation '
            + 'changeset that you should delete or execute manually.'
        )
    )
    parser.add_argument(
        '--asg', 
        action='store_true', 
        default=False, 
        help='Deploy an EC2 autoscaling group fronted by a load balancer (ALB), instead of a single EC2 instance',
    )
    parser.add_argument(
        '--instance-type', type=str, default='t2.micro', help='EC2 instance type',
    )
    args = parser.parse_args()

    deploy(args)
