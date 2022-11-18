from captum.attr import IntegratedGradients
from PIL import Image
import time
import torch
from torchvision import transforms
from ts.torch_handler.base_handler import BaseHandler


# Make aux files imports work for both tests and TorchServe
is_test = "torchserve." in __name__
if is_test:
    from .detect_ops import IDetectOps
    from .preprocess import preprocess_images
    from .yolo_utils import non_max_suppression, scale_coords
else:
    # Inside TorchServe server
    from detect_ops import IDetectOps
    from preprocess import preprocess_images
    from yolo_utils import non_max_suppression, scale_coords


CONFIDENCE_IDX = 4
NUM_CLASSES = 1


class YoloObjectDetector(BaseHandler):
    image_processing = transforms.Compose([transforms.ToTensor()])
    CONF_THRESH = 0.25
    IOU_THRESH = 0.65
    IMG_SIZE = 640
    detect_ops = IDetectOps()
    
    def initialize(self, context):
        super().initialize(context)
        self.ig = IntegratedGradients(self.model)
        self.initialized = True
        properties = context.system_properties
        if not properties.get("limit_max_image_pixels"):
            Image.MAX_IMAGE_PIXELS = None
        self.detect_ops = IDetectOps()
            
    # Adapted from VisionHandler
    def preprocess(self, data):
        """The preprocess function of MNIST program converts the input data to a float tensor
        Args:
            data (List): Input data from the request is in the form of a Tensor
        Returns:
            list : The preprocess function returns the input image as a list of float tensors.
        """
        t1 = time.time()
        orig_images, preprocessed_images = preprocess_images(data)
        tensors = torch.stack([self.image_processing(prepro_img) for prepro_img in preprocessed_images])        
        print('preprocess time = ', time.time() - t1)

        return tensors.to(self.device), orig_images, preprocessed_images
    
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
        tensors, orig_images, preprocessed_images = data
        preds = super().inference(tensors, *args, **kwargs)
        print('Inference time: ', time.time() - t1)
        return preds, orig_images, preprocessed_images

    def postprocess(self, data):
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
                    {'coords': np_det[:4], 'conf': np_det[CONFIDENCE_IDX]} 
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
