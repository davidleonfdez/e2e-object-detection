import base64
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchserve.yolo_handler import CONFIDENCE_IDX, YoloObjectDetector


class StubModule(nn.Module):
    def __init__(self, return_value): 
        super().__init__()
        self.return_value = return_value

    def forward(self, x):
        return self.return_value

def test_yolo_object_detector():
    handler = YoloObjectDetector()

    # chw = 3x2x4
    input_img = np.array([
        [
            [0., 0., 0., 1.],
            [0., 0., 1., 1.],
        ],
        [
            [1., 0., 0., 0.],
            [1., 1., 0., 0.],
        ],
        [
            [0., 0., 1., 0.],
            [0., 1., 1., 0.],
        ]
    ]).transpose(1, 2, 0)
    h, w, nch = input_img.shape
    jpg_img = cv2.imencode('.jpg', input_img)[1]
    base64_jpg_img = base64.b64encode(jpg_img).decode('utf-8')
    png_img = cv2.imencode('.png', input_img)[1]

    data = [
        {'data': jpg_img.tobytes()},
        {'data': base64_jpg_img},
        {'data': png_img.tobytes()}
    ]
    bs = len(data)
    MODEL_IMG_SIZE = YoloObjectDetector.IMG_SIZE
    expected_size = (MODEL_IMG_SIZE * h/w, MODEL_IMG_SIZE) if w > h else (MODEL_IMG_SIZE, MODEL_IMG_SIZE * w/h)

    tensors, orig_images, preprocessed_images = handler.preprocess(data)
    assert isinstance(tensors, torch.Tensor)
    assert np.allclose(tensors.shape, [bs, nch, *expected_size])
    assert len(orig_images) == len(preprocessed_images) == bs        
    assert all(orig_image.shape == input_img.shape for orig_image in orig_images)
    assert all(
        np.allclose(preprocessed_image.shape, [*expected_size, nch]) for preprocessed_image in preprocessed_images
    )

    # 4th dim is 6 because it should contain (x1,y1,x2,y2,confidence,class score)
    expected_model_output = [
        torch.zeros((bs, nch, int(expected_size[0]//stride), int(expected_size[1]//stride), 6)) 
        for stride in (handler.detect_ops.stride)
    ]
    for expected_grid_model_output in expected_model_output:
        # Confidence should go through sigmoid, so 0 is 0.5, -inf is 0 and inf is 1
        # Here we set random values between -2 and 2 (sigmoid(2) ~ 0.88, enough to be chosen)
        expected_grid_model_output[..., CONFIDENCE_IDX] = (torch.rand(expected_grid_model_output.shape[:-1]) - 0.5) * 4
    handler.model = StubModule(expected_model_output)
    preds, orig_images, preprocessed_images = handler.inference((tensors, orig_images, preprocessed_images))
    assert all(
        grid_preds.shape == expected_grid_model_output.shape 
        for grid_preds, expected_grid_model_output in zip(preds, expected_model_output)
    )
    assert all(
        torch.allclose(grid_preds, expected_grid_model_output)
        for grid_preds, expected_grid_model_output in zip(preds, expected_model_output)
    )

    result = handler.postprocess((preds, orig_images, preprocessed_images))
    assert all(len(det['coords']) == 4 for img_result in result for det in img_result)
    assert all(handler.CONF_THRESH <= det['conf'] <= 1 for img_result in result for det in img_result)
