from abc import ABC, abstractmethod
from config import Config
from dataclasses import dataclass
import grpc
from grpc_aux import inference_pb2, inference_pb2_grpc
import json
import requests
from typing import Dict, List


@dataclass
class DetectionResult:
    detections:List[Dict] = None
    error_msg:str = ''

    @property
    def is_error(self): 
        return (self.error_msg is not None) and (len(self.error_msg.strip()) > 0)


class BaseDetectionService(ABC):
    @abstractmethod
    def detect(self, bytes_img:bytes, config:Config) -> DetectionResult:
        pass


class RestDetectionService(BaseDetectionService):
    def detect(self, bytes_img:bytes, config:Config) -> DetectionResult:
        api_url = f'http://{config.backend_host}:{config.rest_api_port}/predictions/{config.model_name}'
        response = requests.post(api_url, {'data': bytes_img})

        if response.ok:
            return DetectionResult(detections=response.json())
        else:
            error_text = f'Error in request to {api_url}'
            return DetectionResult(error_msg=error_text)


class GRPCDetectionService(BaseDetectionService):
    def __init__(self, config):
        self.api_url = f'{config.backend_host}:{config.grpc_api_port}'
        channel = grpc.insecure_channel(self.api_url)
        self.stub = inference_pb2_grpc.InferenceAPIsServiceStub(channel)
    
    def detect(self, bytes_img:bytes, config:Config) -> DetectionResult:
        # TODO: update stub if config changed
        try:        
            response = self.stub.Predictions(
                inference_pb2.PredictionsRequest(model_name=config.model_name,
                                                input={'data': bytes_img}))

            prediction = json.loads(response.prediction.decode('utf-8'))
            return DetectionResult(detections=prediction)
        except grpc.RpcError as e:
            return DetectionResult(error_msg=str(e))
