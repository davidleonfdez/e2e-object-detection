import base64
import cv2
import numpy as np


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    # (Copied from https://github.com/WongKinYiu/yolov7/utils/datasets.py)
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def preprocess_images(data):
    """Extract and preprocess the input images from an incoming request `data`.

    Adapted from ts.handler.vision_handler
    Args:
        data (List): Input data from the request is in the form of a Tensor
    Returns:
        Tuple[List, List]: List of input images and list of preprocessed images.
    """
    orig_images = []
    preprocessed_images = []

    for row in data:
        # Compat layer: normally the envelope should just return the data
        # directly, but older versions of Torchserve didn't have envelope.
        image = row.get("data") or row.get("body")
        if isinstance(image, str):
            # if the image is a string of bytesarray.
            image = base64.b64decode(image)

        # If the image is sent as bytesarray
        if isinstance(image, (bytearray, bytes)):
            image = np.frombuffer(image, dtype=np.uint8)

            # openCV places the data in hwc format
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        else:
            # if the image is a list
            image = np.array(image)

        preprocessed_image, ratio, (dw, dh) = letterbox(image)#, self.img_size, stride=self.stride)[0]

        orig_images.append(image)
        preprocessed_images.append(preprocessed_image)

    return orig_images, preprocessed_images
