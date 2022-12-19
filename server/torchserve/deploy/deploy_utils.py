from botocore.exceptions import ClientError
import random


CFN_UPDATE_IN_PROGRESS_STATUS = 'UPDATE_IN_PROGRESS'


def stack_exists(cfn_client, stack_name:str):
    try:
        data = cfn_client.describe_stacks(StackName=stack_name)
    except ClientError:
        return False
    return True


def create_stack(cfn_client, stack_name):
    if not stack_name:
        stack_name = f'objdet-stack-{random.randint(1e6, 9_999_999)}'

    cfn_client.create_stack(StackName=stack_name)

    return stack_name


def update_stack(cfn_client, stack_name):
    cfn_client.update_stack(StackName=stack_name)


def wait_for_stack_creation(cfn_client, stack_name):
    waiter = cfn_client.get_waiter('stack_create_complete')
    waiter.wait(StackName=stack_name)


def wait_for_stack_update(cfn_client, stack_name):
    # Avoid waiting when an update is not in place; for instance, when CLI `deploy` 
    # produced an empty changset. Without this check, we could get blocked.
    status = cfn_client.describe_stacks(StackName=stack_name)['Stacks'][0]['StackStatus']
    if status != CFN_UPDATE_IN_PROGRESS_STATUS:
        return
    # If status changes between these two sentences we could get blocked anyway but it's highly unlikely
    waiter = cfn_client.get_waiter('stack_update_complete')
    waiter.wait(StackName=stack_name)
