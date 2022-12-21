import av
from config import Config
import io
from PIL import Image, ImageDraw
from services import BaseDetectionService, GRPCDetectionService, RestDetectionService
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import time
from typing import List, Dict


MAX_SIZE = 640
COORDS_RESULT_KEY = 'coords'
REST_API_SELECT_ITEM = "REST"
GRPC_API_SELECT_ITEM = "gRPC"
IMG_SOURCE_SELECT_ITEM = "Image"
CAMERA_SOURCE_SELECT_ITEM = "Camera"
STREAM_SOURCE_SELECT_ITEM = "Camera stream"


@st.cache(ttl=300)
def get_config():
    return Config()


@st.cache(ttl=300, hash_funcs={GRPCDetectionService: lambda serv: serv.api_url})
def get_service(api_type):
    return RestDetectionService() if api_type == REST_API_SELECT_ITEM else GRPCDetectionService(get_config())


config = get_config()


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


def draw_detections(detections, img, display=True):
    if len(detections) == 0: 
        st.text("No objects detected!")
        return

    draw = ImageDraw.ImageDraw(img)    
    
    for detection in detections:
        draw.rectangle(detection[COORDS_RESULT_KEY])
        upper_left_xy = detection[COORDS_RESULT_KEY][:2]
        draw.text(upper_left_xy, str(round(detection['conf'], 3)))

    if display:
        st.image(img)    


def detect_and_draw(bytes_img, sent_img_sz, detection_service:BaseDetectionService, display=True, img:Image=None):
    if img is None:
        img = bytes_to_image(bytes_img)
    scale = sent_img_sz / max(img.size)
    should_resize = scale < 1

    if should_resize:
        bytes_img = image_to_bytes(img.resize(
            (int(dim_size * scale) for dim_size in img.size),
            resample=Image.Resampling.BILINEAR
        ))

    #t1 = time.time()
    detection_result = detection_service.detect(bytes_img, config)
    #st.text(f'Detection time: {time.time() - t1}')

    if not detection_result.is_error:
        detections = detection_result.detections
        if should_resize:
            detections = scale_back_detections(detections, scale)
        draw_detections(detections, img, display=display)      
    else:
        error_text = detection_result.error_msg
        st.error(error_text)
        print(error_text)


sent_img_sz = st.selectbox(
    label="Speed up by resizing largest image dimension in client (before detection) to: ",
    options=[MAX_SIZE, MAX_SIZE // 2, MAX_SIZE // 4]
)

api_type = st.selectbox(
    label="API",
    options=[REST_API_SELECT_ITEM, GRPC_API_SELECT_ITEM]
)

source_type = st.selectbox(
    label="Source",
    options=[IMG_SOURCE_SELECT_ITEM, CAMERA_SOURCE_SELECT_ITEM, STREAM_SOURCE_SELECT_ITEM]
)


def camera_stream_callback(frame):
    img = frame.to_image()
    detect_and_draw(image_to_bytes(img), sent_img_sz, get_service(api_type), display=False, img=img)
    return av.VideoFrame.from_image(img)


uploaded_file = None

if source_type == IMG_SOURCE_SELECT_ITEM:
    uploaded_file = st.file_uploader("Upload an image", label_visibility="visible")
elif source_type == CAMERA_SOURCE_SELECT_ITEM:
    uploaded_file = st.camera_input("Take a picture")
else:
    webrtc_streamer(key="Stream", video_frame_callback=camera_stream_callback)

if uploaded_file is not None:
    detect_and_draw(uploaded_file.getvalue(), sent_img_sz, get_service(api_type))
