from dataclasses import dataclass
import os
import tempfile
from torchserve.deploy.ec2.generate_user_data import generate_user_data


BUCKET_NAME_PLACEHOLDER = "#$%BUCKET_NAME%$#"
REQUIREMENTS_PLACEHOLDER = "#$%REQUIREMENTS%$#"


template = f"""#!/bin/bash
s3_models_bucket="{BUCKET_NAME_PLACEHOLDER}"

sudo yum update -y
# Needed for TorchServe
sudo yum install -y java-17-amazon-corretto-headless

echo "{REQUIREMENTS_PLACEHOLDER}" > requirements.txt
pip install -r requirements.txt

# Download model file (.mar) from S3 to EC2 instance
mkdir models
aws s3api get-object --bucket $s3_models_bucket --key object_detector.mar models/object_detector.mar

# Create TorchServe configuration file and start server
echo -e "inference_address=http://0.0.0.0:8080\\ninstall_py_dep_per_model=true" > config.properties
torchserve --model-store models --models all --ncs --start
"""


@dataclass
class FakeArgs:
    bucket_name:str=""
    output_path:str=""
    req_path:str=""


def test_generate_user_data():
    requirements = """
    --extra-index-url https://download.pytorch.org/whl/cpu
    torch==1.8.0+cpu
    torchserve
    """
    # Delete=False needed for Windows; otherwise, a permission error is raised when trying
    # to open the file a second time (inside `generate_user_data`)
    #with tempfile.NamedTemporaryFile(mode='w', delete=False) as fake_reqs_f:
    fake_reqs_f = tempfile.NamedTemporaryFile(mode='w', delete=False)
    print('file name = ', fake_reqs_f.name)
    try:
        fake_reqs_f.write(requirements)
        fake_reqs_f.close()
        
        bucket_name = "random-bucket-name"
        expected_script_str = (
            template.replace(BUCKET_NAME_PLACEHOLDER, bucket_name).replace(REQUIREMENTS_PLACEHOLDER, requirements)
        )
        generated_script_str = generate_user_data(FakeArgs(
            bucket_name,
            "",
            fake_reqs_f.name
        ))
        print(expected_script_str, generated_script_str)
        assert expected_script_str == generated_script_str

    finally:
        os.unlink(fake_reqs_f.name)
