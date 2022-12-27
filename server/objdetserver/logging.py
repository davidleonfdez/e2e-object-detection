import logging
from pathlib import Path
from objdetserver.deploy.constants import ec2 as ec2_constants
from objdetserver.deploy.constants import ecs as ecs_constants
from objdetserver.handlers import constants
import sys


def _configure_deploy_logger(filename:str):
    root_pkg_path = Path(__file__).resolve().parent
    filepath = root_pkg_path/constants.LOG_DIRNAME/filename
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout), 
            logging.FileHandler(filename=filepath)
        ],
        level=logging.INFO,
    )


def configure_deploy_ec2_logger():
    _configure_deploy_logger(ec2_constants.LOG_FILENAME)


def configure_deploy_ecs_logger():
    _configure_deploy_logger(ecs_constants.LOG_FILENAME)
