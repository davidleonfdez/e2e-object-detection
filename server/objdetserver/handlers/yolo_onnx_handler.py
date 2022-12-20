from captum.attr import IntegratedGradients
from PIL import Image
import numpy as np
import time
import torch
from ts.torch_handler.base_handler import BaseHandler


# Make aux files imports work for both tests and TorchServe
is_test = "objdetserver." in __name__
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
    "TorchServe object detection handler for a YOLO v7 model with ONNX format and NMS included"
    # These thresholds are just informative. The actual thresholds are embedded inside the ONNX model
    # and depend only on the parameters passed to the YOLO v7 export.py script.
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

    def preprocess(self, data):
        """The preprocess function converts the data from a request to a NumPy array.

        The images are resized preserving the aspect ratio and then padded to meet the size expected by the model
        `(self.IMG_SIZE, self.IMG_SIZE)`.
        Args:
            data (List): Input data, every item should be an image base64 encoded and/or as a bytearray or just a 
              plain list.
        Returns:
            Tuple:
            - np.array: float array of preprocessed images, ready to be passed as input to a YOLOv7 ONNX model. 
              Shape: nchw.
            - List[np.array]: original images before preprocessing with OpenCV dimension ordering. 
                Item shape: hwc.
            - List[np.array]: preprocessed images with OpenCV dimension ordering. 
                Item shape: hwc.
        """
        t1 = time.time()
        orig_images, preprocessed_images = preprocess_images(data, pad=True)
        model_inputs = (np.stack(preprocessed_images).astype(np.float32) / 255).transpose(0, 3, 1, 2)
        print('preprocess time = ', time.time() - t1)

        return model_inputs, orig_images, preprocessed_images
    
    def inference(self, data, *args, **kwargs):
        """
        The Inference Function is used to make a prediction call on the given input request.

        The second and third components of `data` are just returned as they are (forwarded to the `postprocess` 
          method).
        This method assumes that a NMS layer is part of the ONNX model.

        Args:
            data (Tuple): same as the output of `preprocess` method.
            - np.array: float array of preprocessed input images, ready to be passed as input to a YOLOv7 ONNX model. 
              Shape: nchw.
            - List[np.array]: input images before preprocessing with OpenCV dimension ordering. 
                Item shape: hwc.
            - List[np.array]: input images after preprocessing with OpenCV dimension ordering. 
                Item shape: hwc.
        Returns:
            [Tuple]:
            - List[np.array]: float arrays of predictions (one entry per image). Shape: n x [number of detections, 7]
              Last dimension contains (0,x1,y1,x2,y2,class,confidence).
            - List[np.array]: input images before preprocessing with OpenCV dimension ordering. 
                Item shape: hwc.
            - List[np.array]: input images after preprocessing with OpenCV dimension ordering. 
                Item shape: hwc.
        """
        t1 = time.time()
        arrays, orig_images, preprocessed_images = data

        ort_inputs = {self.model.get_inputs()[0].name: arrays}
        preds = self.model.run(None, ort_inputs)
        print('Inference time: ', time.time() - t1)

        return preds, orig_images, preprocessed_images

    def postprocess(self, data):
        """
        Transform the output of the detection+NMS layers of YOLOv7 into a list of bounding boxes and confidence scores.

        This method assumes that a NMS layer with IOU threshold `self.IOU_THRESH` is part of the ONNX model.
        Any predicted box with a confidence score lower than `self.CONF_THRESH` is also discarded.
        Args:
            data (Tuple): same as the output of `inference` method.
            - List[np.array]: float arrays of predictions (one entry per image). Shape: n x [number of detections, 7]
              Last dimension contains (0,x1,y1,x2,y2,class,confidence).
            - List[np.array]: input images before preprocessing with OpenCV dimension ordering. 
                Item shape: hwc.
            - List[np.array]: input images after preprocessing with OpenCV dimension ordering. 
                Item shape: hwc.

        Returns:
            List[List[dict]]: number of images x number of detections. Every dict contains the keys:
            - 'coords': bounding box of the detection, represented by an array of length 4 (x1, y1, x2, y2).
              (x1, y1) -> Upper left corner.
              (x2, y2) -> Bottom right corner.
              (0, 0) is the upper left corner of the image.
            - 'conf': confidence score
        """
        t1 = time.time()
        preds, orig_images, preprocessed_images = data

        result = []

        # We don't apply NMS because it's already embedded in the ONNX model exported by YOLOv7 export script

        # Process detections
        # `det = preds[i]` -> detections in image `i`
        for i, (det, orig_img, preprocessed_image) in enumerate(zip(preds, orig_images, preprocessed_images)): 
            if len(det):
                det = torch.tensor(det)
                # Rescale boxes from img_size to orig_img size
                det[:, 1:5] = scale_coords(preprocessed_image.shape, det[:, 1:5], orig_img.shape).round()
                result_cur_img = [
                    {'coords': np_det[1:5], 'conf': np_det[CONFIDENCE_IDX]} 
                    for np_det in det.tolist()
                ]
            else:
                result_cur_img = []
            result.append(result_cur_img)

        print('Postprocess time: ', time.time() - t1)
        return result

    def get_insights(self, tensor_data, _, target=0):
        print("input shape", tensor_data.shape)
        return self.ig.attribute(tensor_data, target=target, n_steps=15).tolist()
