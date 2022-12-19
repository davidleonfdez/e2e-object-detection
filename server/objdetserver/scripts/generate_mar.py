import argparse
from enum import auto, Enum
import os
from pathlib import Path
import shutil
import subprocess


class ModelType(Enum):
    TORCHSCRIPT = auto()
    ONNX = auto()


def _get_model_type(args):
    model_path = args.model_path.strip().lower()
    if model_path.endswith('onnx'):
        return ModelType.ONNX
    elif model_path.endswith('torchscript.pt'):
        return ModelType.TORCHSCRIPT
    return None


def _get_model_files(model_type):
    root_ts_path = Path(__file__).parent.parent
    if model_type == ModelType.ONNX:
        return {
            'handler': 'yolo_onnx_handler.py',
            'req': 'yolo_onnx_handler_requirements.txt',
            'extra_files': ['preprocess.py', 'yolo_utils.py']
            #'handler': str(root_ts_path/'yolo_onnx_handler.py'),
            #'req': str(root_ts_path/'yolo_onnx_handler_requirements.txt'),
            #'extra_files': [str(root_ts_path/p) for p in ['preprocess.py', 'yolo_utils.py']]
        }
    if model_type == ModelType.TORCHSCRIPT:
        return {
            #'handler': 'yolo_handler.py',
            'req': 'yolo_handler_requirements.txt',
            'extra_files': ['detect_ops.py', 'preprocess.py', 'yolo_utils.py'],
            'handler': str(root_ts_path/'yolo_handler.py'),
            #'req': str(root_ts_path/'yolo_handler_requirements.txt'),
            #'extra_files': [str(root_ts_path/p) for p in ['detect_ops.py', 'preprocess.py', 'yolo_utils.py']]
        }


def _get_base_handlers_path():
    return Path(__file__).resolve().parent.parent


def build_command(args):
    model_type = _get_model_type(args)
    if model_type is None:
        raise ValueError('Unsupported serialized model type')

    model_files = _get_model_files(model_type)
    base_path = _get_base_handlers_path()
    extra_files = [str(base_path/fn) for fn in model_files['extra_files']]
    
    cmd = [
        "torch-model-archiver", "-f", "--model-name", args.model_name, "--version", "1.0", 
        "--serialized-file",  args.model_path,  "--handler", model_files['handler'], 
        "--requirements-file", model_files['req'], "--extra-files", ','.join(extra_files)
    ]
    if len(args.out_path.strip()) > 0:
        cmd.extend(["--export-path", args.out_path])
    return cmd


def exec_command(script_arr, args):
    # We need to `cd` to be able to use relative paths. When we use absolute paths, and we copy the .mar
    # file to a different machine, TorchServe gets confused looking for the requirements file at the
    # exact path passed to torch-model-archiver
    cwd = os.getcwd()
    base_path = _get_base_handlers_path()
    os.chdir(base_path)
    proc_out = subprocess.run(script_arr)
    output_is_cwd = len(args.out_path.strip()) == 0
    if output_is_cwd:
        out_path = str(base_path/(args.model_name + '.mar'))
        shutil.copy2(out_path, cwd)
        os.remove(out_path)
    os.chdir(cwd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""Generate a model archive file (.mar) that packages all the artifacts needed to serve
        the model with TorchServe"""
    )
    parser.add_argument(
        'model_path', type=str, help='Path of the input model file (.torchscript.pt or .onnx)'
    )
    parser.add_argument(
        '--model-name', type=str, default='object_detector', help='Name of the output model .mar file',
    )
    parser.add_argument(
        '--out-path', 
        type=str, 
        default='', 
        help='Path where the exported .mar file will be saved. If --export-path is not specified, the file will be saved in the current working directory.',
    )
    args = parser.parse_args()

    script_arr = build_command(args)
    print(' '.join([str(cmd_item) for cmd_item in script_arr]))

    exec_command(script_arr, args)
