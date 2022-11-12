import argparse
from pathlib import Path
import textwrap


S3_BUCKET_VAR_NAME = "s3_models_bucket"


def _add_shebang(script_str):
    return script_str + "#!/bin/bash\n"


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
    #print('Path(__file__).parent =' , Path(__file__).resolve().parent.parent)
    #requirements_path = Path(__file__).resolve().parent.parent.parent.parent / 'requirements.txt'
    #print('requirements_path = ', requirements_path.resolve())
    with open(requirements_path) as f:
        requirements_str = f.read()
    return (script_str + 
        f'echo "{requirements_str}" > requirements.txt\n' +
        'pip install -r requirements.txt\n'
    )


def _add_s3_model_download(script_str):
    return script_str + textwrap.dedent (f"""
        # Download model file (.mar) from S3 to EC2 instance
        mkdir models
        aws s3api get-object --bucket ${S3_BUCKET_VAR_NAME} --key object_detector.mar models/object_detector.mar
        """
    )


def _add_torchserve_start(script_str):
    return script_str + textwrap.dedent(f"""
        # Create TorchServe configuration file and start server
        echo -e "inference_address=http://0.0.0.0:8080\\ninstall_py_dep_per_model=true" > config.properties
        torchserve --model-store models --models all --ncs --start
    """)


def generate_user_data(args):
    script_str = _add_shebang("")
    script_str = _add_bucket_name_var(script_str, args.bucket_name)
    script_str = _add_torchserve_prereqs(script_str)
    script_str = _add_requirements(script_str, args.req_path)
    script_str = _add_s3_model_download(script_str)
    script_str = _add_torchserve_start(script_str)
    return script_str


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""Generate a EC2 user data bash script that setups an EC2 instance and starts a TorchServe
        instance that serves the model object_detector.mar stored in the S3 bucket given by the param 'bucket_name'"""
    )
    parser.add_argument('--bucket-name', type=str, help='name of S3 bucket that contains the model .mar file', required=True)
    parser.add_argument('--output-path', type=str, default='user_data.sh', help='path where the generated script should be written')
    default_req_path = Path(__file__).resolve().parent.parent.parent.parent / 'requirements.txt'
    parser.add_argument(
        '--req-path', type=str, default=default_req_path, 
        help='path of the requirements.txt that must be installed on the server'
    )

    args = parser.parse_args()

    script_str = generate_user_data(args)
    print(script_str)

    with open(args.output_path, 'w') as f:
        f.write(script_str)
