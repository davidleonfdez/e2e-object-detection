from dataclasses import dataclass
import os
import tempfile
from objdetserver.deploy.ec2.generate_user_data import generate_user_data


S3_SUBTEMPLATE_PLACEHOLDER = "#$%S3_SUBTEMPLATE%$#"

REQUIREMENTS_PLACEHOLDER = "#$%REQUIREMENTS%$#"
BUCKET_NAME_PLACEHOLDER = "#$%BUCKET_NAME%$#"
MODEL_NAME_PLACEHOLDER = "#$%MODEL_NAME%$#"
CONFIG_PLACEHOLDER = "#$%CONFIG%$#"


s3_download_template = f"""
# Download model file (.mar) from S3 to EC2 instance
s3_models_bucket="{BUCKET_NAME_PLACEHOLDER}"
aws s3api head-object --bucket $s3_models_bucket --key {MODEL_NAME_PLACEHOLDER}.mar
s3_query_ret_code=$?
# If object exists, download it
if [ "$s3_query_ret_code" = "0" ]; then
        aws s3api get-object --bucket $s3_models_bucket --key {MODEL_NAME_PLACEHOLDER}.mar /models/{MODEL_NAME_PLACEHOLDER}.mar
fi
"""

template = f"""#!/bin/bash
sudo yum update -y
# Needed for TorchServe
sudo yum install -y java-17-amazon-corretto-headless

echo "{REQUIREMENTS_PLACEHOLDER}" > requirements.txt
pip3 install -r requirements.txt

mkdir /models
{S3_SUBTEMPLATE_PLACEHOLDER}
# Create TorchServe configuration file
echo "{CONFIG_PLACEHOLDER}" > config.properties

# Start server
torchserve --model-store /models --models all --ncs --start
"""


@dataclass
class FakeArgs:
    bucket_name:str=""
    model_name:str=""
    output_path:str=""
    req_path:str=""
    conf_path:str=""


def test_generate_user_data():
    requirements = """
    --extra-index-url https://download.pytorch.org/whl/cpu
    torch==1.8.0+cpu
    torchserve
    """
    ts_config = """
    inference_address=http://0.0.0.0:8080
    management_address=http://0.0.0.0:8081
    install_py_dep_per_model=true
    """
    # Delete=False needed for Windows; otherwise, a permission error is raised when trying
    # to open the file a second time (inside `generate_user_data`)
    #with tempfile.NamedTemporaryFile(mode='w', delete=False) as fake_reqs_f:
    fake_reqs_f = tempfile.NamedTemporaryFile(mode='w', delete=False)
    fake_conf_f = tempfile.NamedTemporaryFile(mode='w', delete=False)

    try:
        fake_reqs_f.write(requirements)
        fake_reqs_f.close()
        fake_conf_f.write(ts_config)
        fake_conf_f.close()
        
        bucket_name = "random-bucket-name"
        model_name = "random-model-name"
        expected_script_str = (
            template.replace(REQUIREMENTS_PLACEHOLDER, requirements)
                    .replace(CONFIG_PLACEHOLDER, ts_config)
                    .replace(
                        S3_SUBTEMPLATE_PLACEHOLDER, 
                        (s3_download_template.replace(BUCKET_NAME_PLACEHOLDER, bucket_name)
                                             .replace(MODEL_NAME_PLACEHOLDER, model_name))
                    )
        )
        generated_script_str = generate_user_data(FakeArgs(
            bucket_name,
            model_name,
            "",
            fake_reqs_f.name,
            fake_conf_f.name,
        ))
        assert expected_script_str == generated_script_str

        expected_script_nobucket_str = (
            template.replace(REQUIREMENTS_PLACEHOLDER, requirements)
                    .replace(CONFIG_PLACEHOLDER, ts_config)
                    .replace(S3_SUBTEMPLATE_PLACEHOLDER, "")
        )
        generated_script_nobucket_str = generate_user_data(FakeArgs(
            "",
            model_name,
            "",
            fake_reqs_f.name,
            fake_conf_f.name,
        ))
        assert expected_script_nobucket_str == generated_script_nobucket_str

    finally:
        os.unlink(fake_reqs_f.name)
        os.unlink(fake_conf_f.name)
