from captum.attr import IntegratedGradients
from PIL import Image
import time
import torch
from torchvision import transforms
from ts.torch_handler.base_handler import BaseHandler


ROOT_PACKAGE_NAME = "objdetserver"


# Make aux files imports work for both tests and TorchServe
is_test = f"{ROOT_PACKAGE_NAME}." in __name__
if is_test:
    from .detect_ops import IDetectOps
    from .preprocess import preprocess_images
    from .yolo_utils import non_max_suppression, scale_coords
else:
    # Inside TorchServe server
    from detect_ops import IDetectOps
    from preprocess import preprocess_images
    from yolo_utils import non_max_suppression, scale_coords


class YoloObjectDetector(BaseHandler):
    "TorchServe object detection handler for a YOLO v7 model compiled with TorchScript"
    image_processing = transforms.Compose([transforms.ToTensor()])
    CONF_THRESH = 0.25
    IOU_THRESH = 0.65
    IMG_SIZE = 640
    CONFIDENCE_IDX = 4
    detect_ops = IDetectOps()
    
    def initialize(self, context):
        super().initialize(context)
        self.ig = IntegratedGradients(self.model)
        self.initialized = True
        properties = context.system_properties
        if not properties.get("limit_max_image_pixels"):
            Image.MAX_IMAGE_PIXELS = None
        self.detect_ops = IDetectOps()
            
    def preprocess(self, data):
        """The preprocess function converts the data from a request to a float tensor.

        The images are resized preserving the aspect ratio and then padded to meet the size expected by the model
        `(self.IMG_SIZE, self.IMG_SIZE)`.
        Args:
            data (List): Input data, every item should be an image base64 encoded and/or as a bytearray or just a 
              plain list.
        Returns:
            Tuple:
            - torch.Tensor: float tensor of preprocessed images, ready to be passed as input to a YOLOv7 
                PyTorch/TorchScript model. Shape: nchw.
            - List[np.array]: original images before preprocessing with OpenCV dimension ordering. 
                Item shape: hwc.
            - List[np.array]: preprocessed images with OpenCV dimension ordering. 
                Item shape: hwc.
        """
        t1 = time.time()
        orig_images, preprocessed_images = preprocess_images(data)
        tensors = torch.stack([self.image_processing(prepro_img) for prepro_img in preprocessed_images])
        print('preprocess time = ', time.time() - t1)

        return tensors.to(self.device), orig_images, preprocessed_images
    
    def inference(self, data, *args, **kwargs):
        """
        The Inference Function is used to make a prediction call on the given input request.

        The second and third components of `data` are just returned as they are (forwarded to the `postprocess` 
          method).
        Args:
            data (Tuple): same as the output of `preprocess` method.
            - torch.Tensor: float tensor of preprocessed input images, ready to be passed as input to a YOLOv7 
                PyTorch/TorchScript model. Shape: nchw.
            - List[np.array]: input images before preprocessing with OpenCV dimension ordering. 
                Item shape: hwc.
            - List[np.array]: input images after preprocessing with OpenCV dimension ordering. 
                Item shape: hwc.
        Returns:
            Tuple:
            - torch.Tensor: float tensor of predictions. Shape: [number of grids, n, c, grid height, grid width, 6]
              Last dimension contains (x1,y1,x2,y2,confidence,class).
            - List[np.array]: input images before preprocessing with OpenCV dimension ordering. 
                Item shape: hwc.
            - List[np.array]: input images after preprocessing with OpenCV dimension ordering. 
                Item shape: hwc.
        """
        t1 = time.time()
        tensors, orig_images, preprocessed_images = data
        preds = super().inference(tensors, *args, **kwargs)
        print('Inference time: ', time.time() - t1)
        return preds, orig_images, preprocessed_images

    def postprocess(self, data):
        """Transform the output of the detection layers of YOLOv7 into a list of bounding boxes and confidence scores.

        The detection boxes are filtered using non maximum supression (NMS), with `self.IOU_THRESH` being the 
        intersection over union threshold employed to discard the boxes whose IOU with the candidate box (of a given
        NMS iteration) exceeds that value.
        Any predicted box with a confidence score lower than `self.CONF_THRESH` is also discarded.
        Args:
            data (Tuple): same as the output of `inference` method.
            - torch.Tensor: float tensor of predictions. Shape: [number of grids, n, c, grid height, grid width, 6]
              Last dimension contains (x1,y1,x2,y2,confidence,class).
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
        preds = self.detect_ops(preds)

        pred = non_max_suppression(preds[0], self.CONF_THRESH, self.IOU_THRESH)

        result = []

        # Process detections
        for i, (det, orig_img, preprocessed_image) in enumerate(zip(pred, orig_images, preprocessed_images)):  # detections per image
            if len(det):
                # Rescale boxes from img_size to orig_img size
                det[:, :4] = scale_coords(preprocessed_image.shape, det[:, :4], orig_img.shape).round()
                result_cur_img = [
                    {'coords': np_det[:4], 'conf': np_det[self.CONFIDENCE_IDX]} 
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
