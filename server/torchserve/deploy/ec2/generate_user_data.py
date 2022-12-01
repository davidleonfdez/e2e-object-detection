import argparse
from pathlib import Path
import textwrap


S3_BUCKET_VAR_NAME = "s3_models_bucket"


def _add_shebang(script_str):
    return script_str + "#!/bin/bash"


def _add_bucket_name_var(script_str, bucket_name):
    return script_str + f'{S3_BUCKET_VAR_NAME}="{bucket_name}"\n'


def _add_torchserve_prereqs(script_str):
    return script_str + textwrap.dedent("""
        sudo yum update -y
        # Needed for TorchServe
        sudo yum install -y java-17-amazon-corretto-headless

        """
    )


def _add_requirements(script_str, requirements_path):
    with open(requirements_path) as f:
        requirements_str = f.read()
    return (script_str + 
        f'echo "{requirements_str}" > requirements.txt\n' +
        'pip3 install -r requirements.txt\n\n'
    )


def _add_create_models_dir(script_str):
    return script_str + "mkdir /models\n\n"

    
def _add_s3_model_download(script_str):
    return script_str + textwrap.dedent (f"""
        # Download model file (.mar) from S3 to EC2 instance
        aws s3api get-object --bucket ${S3_BUCKET_VAR_NAME} --key object_detector.mar /models/object_detector.mar
        """
    )


def _add_torchserve_config(script_str, conf_path):
    with open(conf_path) as f:
        conf_str = f.read()
    return (script_str + 
        "# Create TorchServe configuration file\n" +
        f'echo "{conf_str}" > config.properties\n'
    )


def _add_torchserve_start(script_str):
    return script_str + textwrap.dedent(f"""
        # Start server
        torchserve --model-store models --models all --ncs --start
    """)


def generate_user_data(args):
    script_str = _add_shebang("")
    #script_str = _add_bucket_name_var(script_str, args.bucket_name)
    script_str = _add_torchserve_prereqs(script_str)
    script_str = _add_requirements(script_str, args.req_path)
    script_str = _add_create_models_dir(script_str)
    #script_str = _add_s3_model_download(script_str)
    script_str = _add_torchserve_config(script_str, args.conf_path)
    #script_str = _add_torchserve_start(script_str)
    return script_str


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate a EC2 user data bash script that setups an EC2 instance ready to run TorchServe"
        ##description="""Generate a EC2 user data bash script that setups an EC2 instance and starts a TorchServe
        #instance that serves the model object_detector.mar stored in the S3 bucket given by the param 'bucket_name'"""
    )
    #parser.add_argument(
    #    '--bucket-name', type=str, help='name of S3 bucket that contains the model .mar file', required=True
    #)
    parser.add_argument(
        '--output-path', type=str, default='user_data.sh', help='path where the generated script should be written'
    )
    default_req_path = Path(__file__).resolve().parent.parent.parent.parent / 'requirements.txt'
    parser.add_argument(
        '--req-path', type=str, default=default_req_path, 
        help='local path of the requirements.txt that must be installed on the server'
    )
    default_ts_conf_path = Path(__file__).resolve().parent.parent.parent / 'config.properties'
    parser.add_argument(
        '--conf-path', type=str, default=default_ts_conf_path, 
        help='local path of the TorchServe config file (config.properties) that must be copied to the server'
    )

    args = parser.parse_args()

    script_str = generate_user_data(args)
    print(script_str)

    with open(args.output_path, 'w') as f:
        f.write(script_str)
