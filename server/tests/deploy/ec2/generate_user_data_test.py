from dataclasses import dataclass
import os
import tempfile
from torchserve.deploy.ec2.generate_user_data import generate_user_data


REQUIREMENTS_PLACEHOLDER = "#$%REQUIREMENTS%$#"
CONFIG_PLACEHOLDER = "#$%CONFIG%$#"


template = f"""#!/bin/bash
sudo yum update -y
# Needed for TorchServe
sudo yum install -y java-17-amazon-corretto-headless

echo "{REQUIREMENTS_PLACEHOLDER}" > requirements.txt
pip3 install -r requirements.txt

mkdir /models

# Create TorchServe configuration file
echo "{CONFIG_PLACEHOLDER}" > config.properties
"""


@dataclass
class FakeArgs:
    bucket_name:str=""
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
    print('file name = ', fake_reqs_f.name)
    try:
        fake_reqs_f.write(requirements)
        fake_reqs_f.close()
        fake_conf_f.write(ts_config)
        fake_conf_f.close()
        
        bucket_name = "random-bucket-name"
        expected_script_str = (
            template.replace(REQUIREMENTS_PLACEHOLDER, requirements).replace(CONFIG_PLACEHOLDER, ts_config)
        )
        generated_script_str = generate_user_data(FakeArgs(
            bucket_name,
            "",
            fake_reqs_f.name,
            fake_conf_f.name,
        ))
        print(expected_script_str, generated_script_str)
        assert expected_script_str == generated_script_str

    finally:
        os.unlink(fake_reqs_f.name)
        os.unlink(fake_conf_f.name)
