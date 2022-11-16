from config import Config
import io
from PIL import Image, ImageDraw
from services import BaseDetectionService, GRPCDetectionService, RestDetectionService
import streamlit as st
import time
from typing import List, Dict


MAX_SIZE = 640
COORDS_RESULT_KEY = 'coords'


@st.cache(ttl=300)
def get_config():
    return Config()


config = get_config()
#detection_service = RestDetectionService()
detection_service = GRPCDetectionService(config)


def bytes_to_image(bytes_data):
    return Image.open(io.BytesIO(bytes_data))


def image_to_bytes(image:Image.Image):
    bytes_io = io.BytesIO()
    image.save(bytes_io, quality=100, format='jpeg')
    return bytes_io.getvalue()


def scale_coords(coords, scale):
    return [int(coord / scale) for coord in coords]


def scale_back_detections(detections:List[Dict], scale):
    return [
        {k: (scale_coords(v, scale) if k == COORDS_RESULT_KEY else v) for k, v in detection.items()}
        for detection in detections 
    ]


def draw_detections(detections, img):
    if len(detections) == 0: 
        st.text("No objects detected!")
        return

    draw = ImageDraw.ImageDraw(img)    
    
    for detection in detections:
        draw.rectangle(detection[COORDS_RESULT_KEY])
        upper_left_xy = detection[COORDS_RESULT_KEY][:2]
        draw.text(upper_left_xy, str(round(detection['conf'], 3)))

    st.image(img)    


def detect_and_draw(uploaded_file, sent_img_sz, detection_service:BaseDetectionService):
    bytes_img = uploaded_file.getvalue()

    img = bytes_to_image(bytes_img)
    scale = sent_img_sz / max(img.size)
    should_resize = scale < 1

    if should_resize:
        bytes_img = image_to_bytes(img.resize(
            (int(dim_size * scale) for dim_size in img.size),
            resample=Image.Resampling.BILINEAR
        ))

    detection_result = detection_service.detect(bytes_img, config)

    if not detection_result.is_error:
        detections = detection_result.detections
        if should_resize:
            detections = scale_back_detections(detections, scale)
        draw_detections(detections, img)      
    else:
        error_text = detection_result.error_msg
        st.error(error_text)
        print(error_text)


sent_img_sz = st.selectbox(
    label="Speed up by resizing largest image dimension in client (before detection) to: ",
    options=[MAX_SIZE, MAX_SIZE // 2, MAX_SIZE // 4]
)

uploaded_file = st.file_uploader("Upload an image", label_visibility="visible")

if uploaded_file is not None:
    detect_and_draw(uploaded_file, sent_img_sz, detection_service)
