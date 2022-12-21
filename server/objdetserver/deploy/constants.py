from dataclasses import dataclass


@dataclass(frozen=True)
class EC2:
    TEMPLATE_FILENAME = 'cfn_stack.yaml'
    TEMPLATE_FILENAME_ASG = 'asg_stack.yaml'
    BUCKET_NAME_STACK_OUT = 'S3BucketName'
    LAMBDA_ARN_STACK_OUT = 'LambdaArn'
    # If you edit this value, you should edit the same in ec2/asg_stack.yaml
    BUCKET_NAME_PLACEHOLDER_IN_USER_DATA = '__BUCKET_NAME__'
    GENERATE_USER_DATA_SCRIPT_FILENAME = 'generate_user_data.py'
    USER_DATA_OUT_FILENAME = 'user_data_generated.sh'


@dataclass(frozen=True)
class ECS:
    TEMPLATE_FILENAME = 'ecs_stack.yaml'
    DOCKERFILE_MAR_PATH_ARG = 'mar_path'


ec2 = EC2()
ecs = ECS()
