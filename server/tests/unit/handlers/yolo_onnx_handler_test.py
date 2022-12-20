import base64
import cv2
from dataclasses import dataclass
import math
import numpy as np
import torch
import torch.nn as nn
from objdetserver.handlers.yolo_onnx_handler import CONFIDENCE_IDX, YoloONNXObjectDetector


@dataclass
class FakeONNXInput:
    name:str='InputName'


class StubONNXModel():
    def __init__(self, return_value): 
        super().__init__()
        self.return_value = return_value

    def get_inputs(self):
        return [FakeONNXInput()]

    def run(self, _, inputs):
        return self.return_value


@torch.no_grad()
def test_yolo_onnx_object_detector():
    # Declare handler in outer scope to be in a situation similar to production,
    # where the same instance of the handler receives multiple requests
    handler = YoloONNXObjectDetector()
    #max_stride = max(handler.detect_ops.stride)

    # chw = 3x2x4
    input_img_a = np.array([
        [
            [0., 1., 0., 1.],
            [1., 0., 1., 0.],
        ],
        [
            [1., 0., 0., 0.],
            [0., 0., 1., 1.],
        ],
        [
            [0., 0., 1., 0.],
            [0., 1., 1., 0.],
        ]
    ]).transpose(1, 2, 0)
    # chw = 3x2x3
    input_img_b = np.array([
        [
            [0., 0., 1.],
            [0., 1., 0.],
        ],
        [
            [1., 0., 1.],
            [1., 0., 0.],
        ],
        [
            [0., 1., 0.],
            [0., 1., 1.],
        ]
    ]).transpose(1, 2, 0)

    def _test_img(input_img):
        h, w, nch = input_img.shape
        jpg_img = cv2.imencode('.jpg', input_img)[1]
        base64_jpg_img = base64.b64encode(jpg_img).decode('utf-8')
        png_img = cv2.imencode('.png', input_img)[1]

        data = [
            {'data': png_img.tobytes()},        
            {'data': jpg_img.tobytes()},
            {'data': base64_jpg_img},
        ]
        bs = len(data)
        MODEL_IMG_SIZE = YoloONNXObjectDetector.IMG_SIZE
        expected_size = (MODEL_IMG_SIZE, MODEL_IMG_SIZE)

        np_preprocessed_imgs, orig_images, preprocessed_images = handler.preprocess(data)
        assert np.allclose(np_preprocessed_imgs.shape, [bs, nch, *expected_size])
        assert len(orig_images) == len(preprocessed_images) == bs
        assert all(orig_image.shape == input_img.shape for orig_image in orig_images)
        assert all(
            np.allclose(preprocessed_image.shape, [*expected_size, nch]) for preprocessed_image in preprocessed_images
        )

        # n x n_detections x 7
        # 4th dim is 7 because it should contain (0,x1,y1,x2,y2,class,confidence)
        expected_model_output = [
            torch.zeros((torch.randint(0, 5, (1,)), 7)) 
            for _ in range(bs)
        ]
        expected_model_output = [
            torch.Tensor([
                [0., 0,0, MODEL_IMG_SIZE-1,MODEL_IMG_SIZE-1, 0, 0.5],
                [0., MODEL_IMG_SIZE//2,MODEL_IMG_SIZE//2, MODEL_IMG_SIZE-1,MODEL_IMG_SIZE-1, 0, 0.3],
                [0., 0,0, MODEL_IMG_SIZE//2,MODEL_IMG_SIZE//2, 0, 0.6],
            ]),
            torch.Tensor([
                [0., 0,0, MODEL_IMG_SIZE-1,MODEL_IMG_SIZE//2, 0, 0.9],
                [0., 0,0, MODEL_IMG_SIZE//2,MODEL_IMG_SIZE-1, 0, 0.1],
                [0., 0,0, MODEL_IMG_SIZE-1,MODEL_IMG_SIZE-1, 0, 0.6],
            ]),
            torch.Tensor([
                [0., 0,0, MODEL_IMG_SIZE-1,MODEL_IMG_SIZE//2, 0, 0.2],
                [0., 0,0, MODEL_IMG_SIZE//2,MODEL_IMG_SIZE-1, 0, 0.4],
            ]),
        ]

        handler.model = StubONNXModel(expected_model_output)
        preds, orig_images, preprocessed_images = handler.inference(
            (np_preprocessed_imgs, orig_images, preprocessed_images)
        )
        assert all(
            img_preds.shape == expected_img_model_output.shape
            for img_preds, expected_img_model_output in zip(preds, expected_model_output)
        )
        assert all(
            torch.allclose(img_preds, expected_img_model_output)
            for img_preds, expected_img_model_output in zip(preds, expected_model_output)
        )


        expected_scale = w / MODEL_IMG_SIZE, h / MODEL_IMG_SIZE
        expected_postprocess_output = []
        for expected_img_model_output in expected_model_output:
            expected_img_postprocess_output = (
                expected_img_model_output * np.array([1, *expected_scale, *expected_scale, 1, 1])
            )
            expected_img_postprocess_output[..., 1:5] = expected_img_postprocess_output[..., 1:5].round()
            expected_postprocess_output.append(expected_img_postprocess_output)
        result = handler.postprocess((preds, orig_images, preprocessed_images))

        def _detection_matches_expectation(det, expected_det_arr):
            return (
                np.allclose(det['coords'], expected_det_arr[1:5], rtol=0, atol=0.01) 
                and (det['conf'] == expected_det_arr[CONFIDENCE_IDX])
            )
        assert all(
            _detection_matches_expectation(det, expected_det_arr)
            for img_result, img_expected_result in zip(result, expected_postprocess_output)
            for det, expected_det_arr in zip(img_result, img_expected_result)
        )

    _test_img(input_img_a)
    _test_img(input_img_b)
