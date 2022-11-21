import argparse
from enum import auto, Enum
from pathlib import Path
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
    if model_type == ModelType.ONNX:
        return {
            'handler': 'yolo_onnx_handler.py',
            'req': 'yolo_onnx_handler_requirements.txt',
            'extra_files': ['preprocess.py', 'yolo_utils.py']
        }
    if model_type == ModelType.TORCHSCRIPT:
        return {
            'handler': 'yolo_handler.py',
            'req': 'yolo_handler_requirements.txt',
            'extra_files': ['detect_ops.py', 'preprocess.py', 'yolo_utils.py']
        }


def build_command(args):
    model_type = _get_model_type(args)
    if model_type is None:
        raise ValueError('Unsupported serialized model type')

    model_files = _get_model_files(model_type)
    base_path = Path(__file__).resolve().parent.parent
    extra_files = [str(base_path/fn) for fn in model_files['extra_files']]
    cmd = [
        "torch-model-archiver", "-f", "--model-name", args.model_name, "--version", "1.0", 
        "--serialized-file",  args.model_path,  "--handler", model_files['handler'], 
        "--requirements-file", model_files['req'], "--extra-files", ','.join(extra_files)
    ]
    return cmd


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
    args = parser.parse_args()

    script_arr = build_command(args)
    print(' '.join(script_arr))

    proc_out = subprocess.run(script_arr)
