from captum.attr import IntegratedGradients
from PIL import Image
import numpy as np
import time
import torch
from ts.torch_handler.base_handler import BaseHandler


# Make aux files imports work for both tests and TorchServe
is_test = "torchserve." in __name__
if is_test:
    from .preprocess import preprocess_images
    from .yolo_utils import scale_coords
else:
    # Inside TorchServe server
    from preprocess import preprocess_images
    from yolo_utils import scale_coords


CONFIDENCE_IDX = 6
NUM_CLASSES = 1


class YoloONNXObjectDetector(BaseHandler):
    CONF_THRESH = 0.25
    IOU_THRESH = 0.65
    IMG_SIZE = 640
    
    def initialize(self, context):
        super().initialize(context)
        self.ig = IntegratedGradients(self.model)
        self.initialized = True
        properties = context.system_properties
        if not properties.get("limit_max_image_pixels"):
            Image.MAX_IMAGE_PIXELS = None
            
    # Adapted from VisionHandler
    def preprocess(self, data):
        """The preprocess function of MNIST program converts the input data to a float tensor
        Args:
            data (List): Input data from the request is in the form of a Tensor
        Returns:
            list : The preprocess function returns the input image as a list of float tensors.
        """
        t1 = time.time()
        orig_images, preprocessed_images = preprocess_images(data, pad=True)
        model_inputs = (np.stack(preprocessed_images).astype(np.float32) / 255).transpose(0, 3, 1, 2)
        print('preprocess time = ', time.time() - t1)

        return model_inputs, orig_images, preprocessed_images
    
    def inference(self, data, *args, **kwargs):
        """
        The Inference Function is used to make a prediction call on the given input request.
        The user needs to override the inference function to customize it.
        Args:
            data (Torch Tensor): A Torch Tensor is passed to make the Inference Request.
            The shape should match the model input shape.
        Returns:
            Torch Tensor : The Predicted Torch Tensor is returned in this function.
        """
        t1 = time.time()
        arrays, orig_images, preprocessed_images = data

        ort_inputs = {self.model.get_inputs()[0].name: arrays}
        preds = self.model.run(None, ort_inputs)
        print('Inference time: ', time.time() - t1)

        return preds, orig_images, preprocessed_images

    def postprocess(self, data):
        t1 = time.time()
        preds, orig_images, preprocessed_images = data

        result = []

        # We don't apply NMS because it's already embedded in the ONNX model exported by YOLOv7 export script

        # Process detections
        for i, (det, orig_img, preprocessed_image) in enumerate(zip(preds, orig_images, preprocessed_images)):  # detections per image
            if len(det):
                det = torch.tensor(det)
                # Rescale boxes from img_size to orig_img size
                det[:, 1:5] = scale_coords(preprocessed_image.shape, det[:, 1:5], orig_img.shape).round()
                result_cur_img = [
                    {'coords': np_det[1:5], 'conf': np_det[CONFIDENCE_IDX]} 
                    for np_det in reversed(det.tolist())
                ]
            else:
                result_cur_img = []
            result.append(result_cur_img)

        print('Postprocess time: ', time.time() - t1)
        return result

    def get_insights(self, tensor_data, _, target=0):
        print("input shape", tensor_data.shape)
        return self.ig.attribute(tensor_data, target=target, n_steps=15).tolist()
