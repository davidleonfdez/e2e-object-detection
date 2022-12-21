NUM_CLASSES = 1

GENERATE_MAR_SCRIPT_FILENAME = 'generate_mar.py'
SCRIPTS_DIRNAME = 'scripts'

HANDLERS_DIRNAME = 'handlers'
ONNX_HANDLER_FILENAME = 'yolo_onnx_handler.py'
ONNX_HANDLER_REQS_FILENAME = 'yolo_onnx_handler_requirements.txt'
ONNX_HANDLER_EXTRA_FILES = ['preprocess.py', 'yolo_utils.py']

TORCHSCRIPT_HANDLER_FILENAME = 'yolo_handler.py'
TORCHSCRIPT_HANDLER_REQS_FILENAME = 'yolo_handler_requirements.txt'
TORCHSCRIPT_HANDLER_EXTRA_FILES = ['detect_ops.py', 'preprocess.py', 'yolo_utils.py', 'constants.py']
