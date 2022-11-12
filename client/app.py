from config import Config
import io
from PIL import Image, ImageDraw
import requests
import streamlit as st


@st.cache(ttl=300)
def get_config():
    return Config()


config = get_config()


def bytes_to_image(bytes_data):
    return Image.open(io.BytesIO(bytes_data))


def draw_detections(detections, img):
    if len(detections) == 0: 
        st.text("No objects detected!")
        return

    draw = ImageDraw.ImageDraw(img)    
    
    for detection in detections:
        draw.rectangle(detection['coords'])
        upper_left_xy = detection['coords'][:2]
        draw.text(upper_left_xy, str(round(detection['conf'], 3)))

    st.image(img)    


def detect_and_draw(uploaded_file):
    bytes_img = uploaded_file.getvalue()
    api_url = config.backend_url
    detection_response = requests.post(api_url, {'data': bytes_img})
    if detection_response.ok:
        img = bytes_to_image(bytes_img)
        draw_detections(detection_response.json(), img)      
    else:
        error_text = f'Error in request to {api_url}'
        st.error(error_text)
        print(error_text)


uploaded_file = st.file_uploader("Upload an image", label_visibility="visible")

if uploaded_file is not None:
    detect_and_draw(uploaded_file)