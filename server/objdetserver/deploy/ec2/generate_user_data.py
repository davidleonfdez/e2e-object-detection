import argparse
from pathlib import Path
import textwrap


S3_BUCKET_VAR_NAME = "s3_models_bucket"


def _get_shebang():
    return "#!/bin/bash"


def _get_torchserve_prereqs():
    return textwrap.dedent("""
        sudo yum update -y
        # Needed for TorchServe
        sudo yum install -y java-17-amazon-corretto-headless

        """
    )


def _get_requirements(requirements_path):
    with open(requirements_path) as f:
        requirements_str = f.read()
    return ( 
        f'echo "{requirements_str}" > requirements.txt\n' +
        'pip3 install -r requirements.txt\n\n'
    )


def _get_create_models_dir():
    return "mkdir /models\n\n"

    
def _get_s3_model_download(bucket_name, model_name):
    return "\n".join([
        f'# Download model file (.mar) from S3 to EC2 instance',
        f'{S3_BUCKET_VAR_NAME}="{bucket_name}"',
        f'aws s3api head-object --bucket ${S3_BUCKET_VAR_NAME} --key {model_name}.mar',
        f's3_query_ret_code=$?',
        f'# If object exists, download it',
        f'if [ "$s3_query_ret_code" = "0" ]; then',
        f'        aws s3api get-object --bucket ${S3_BUCKET_VAR_NAME} --key {model_name}.mar /models/{model_name}.mar',
        'fi\n\n'
    ])


def _get_torchserve_config(conf_path):
    with open(conf_path) as f:
        conf_str = f.read()
    return ( 
        "# Create TorchServe configuration file\n" +
        f'echo "{conf_str}" > config.properties\n'
    )


def _get_torchserve_start():
    return textwrap.dedent(f"""
        # Start server
        torchserve --model-store /models --models all --ncs --start
    """)


def generate_user_data(args):
    script_str = _get_shebang()
    script_str += _get_torchserve_prereqs()
    script_str += _get_requirements(args.req_path)
    script_str += _get_create_models_dir()
    if len(args.bucket_name) > 0:
        script_str += _get_s3_model_download(args.bucket_name, args.model_name)
    script_str += _get_torchserve_config(args.conf_path)
    script_str += _get_torchserve_start()
    return script_str


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(
            """Generate a EC2 user data bash script that setups an EC2 instance and starts a TorchServe
            instance that (optionally) serves the model object_detector.mar stored in the S3 bucket given by the 
            param 'bucket_name'.
            """
        )
    )
    parser.add_argument(
        '--bucket-name', 
        type=str, 
        help=textwrap.dedent(
            """Name of S3 bucket that contains the model .mar file.
            If this param is empty (default value), it means that the generated script shouldn't include code to 
            download the model from S3.
            """
        ), 
        default='',
    )
    parser.add_argument(
        '--model-name', type=str, default='object_detector', help='Name of the model (S3 key without the extension)',
    )
    parser.add_argument(
        '--output-path', type=str, default='user_data.sh', help='Path where the generated script should be written.'
    )
    default_req_path = Path(__file__).resolve().parent.parent.parent.parent / 'requirements.txt'
    parser.add_argument(
        '--req-path', type=str, default=default_req_path, 
        help='Local path of the requirements.txt that must be installed on the server.'
    )
    default_ts_conf_path = Path(__file__).resolve().parent.parent.parent / 'config.properties'
    parser.add_argument(
        '--conf-path', type=str, default=default_ts_conf_path, 
        help='Local path of the TorchServe config file (config.properties) that must be copied to the server.'
    )

    args = parser.parse_args()

    script_str = generate_user_data(args)
    print(script_str)

    with open(args.output_path, 'w') as f:
        f.write(script_str)
