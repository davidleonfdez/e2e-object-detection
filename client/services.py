from abc import ABC, abstractmethod
from config import Config
from dataclasses import dataclass
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
        api_url = config.backend_url
        response = requests.post(api_url, {'data': bytes_img})

        if response.ok:
            return DetectionResult(detections=response.json())
        else:
            error_text = f'Error in request to {api_url}'
            return DetectionResult(error_msg=error_text)
