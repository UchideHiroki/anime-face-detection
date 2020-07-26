import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import os

st.set_option('deprecation.showfileUploaderEncoding', False)

from anime_face_detection import AnimeFaceDetection
from nms_wrapper import NMSType, NMSWrapper
from settings import DEMO_IMAGE, MODEL, CHARACTERS, CASCADE

@st.cache
def read_image(img_file_buffer):
    if img_file_buffer is not None:
        image = np.asarray(bytearray(img_file_buffer.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = cv2.imread(DEMO_IMAGE, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
    return image
@st.cache
def face_detection_by_lbpcascade(image):
    cascade = cv2.CascadeClassifier(CASCADE)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    face_frames = cascade.detectMultiScale(
        gray,
        # detector options
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (24, 24)
    )
    face_images = [image[y:(y+h), x:(x+w)] for (x, y, w, h) in face_frames]
    if len(face_images) > 0:
        face_images = [cv2.resize(face, (224, 224)) for face in face_images]
        for (x, y, w, h) in face_frames:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return face_images, image

detector = AnimeFaceDetection()

def face_detection_by_frcnn(image):
    scores, boxes = detector.detect(image)
    
    nms_type = NMSType.PY_NMS
    nms = NMSWrapper(nms_type)
    nms_thresh = 0.3
    conf_thresh = 0.8
    
    boxes = boxes[:, 4:8]
    scores = scores[:, 1]
    keep = nms(np.hstack([boxes, scores[:, np.newaxis]]).astype(np.float32), nms_thresh)
    boxes = boxes[keep, :]
    scores = scores[keep]
    inds = np.where(scores >= conf_thresh)[0]
    scores = scores[inds]
    boxes = boxes[inds, :]
    
    face_images = []
    cropped_image = image.copy()
    
    for box in boxes:
        x1, y1, x2, y2 = box
        face_image = image[int(y1):int(y2), int(x1):int(x2)]
        cropped_image = cv2.rectangle(cropped_image, (x1,y2), (x2,y1), (0,255,0), 3)
        face_images.append(face_image)
    
    return face_images, cropped_image

# @st.cache
# def predict_image(image):
#     model = tf.keras.models.load_model(MODEL, custom_objects={"KerasLayer":hub.KerasLayer})
#     image = image / 255
#     scores = model.predict(image[np.newaxis, :]).flatten()
#     label = CHARACTERS[np.argmax(scores)]
#     return label, scores

st.title("Character Detection with Fast R-CNN")
img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
image = read_image(img_file_buffer)
face_images, detected_image = face_detection_by_frcnn(image)
st.image(detected_image, caption=f"Result", width=200)
if len(face_images) == 0:
    st.text("No faces are detected")
for face in face_images:
    # label, scores = predict_image(face)
    st.image(
        face, caption=f"Detected face", width=100,
    )
    # st.write(f"Predict character: {label}")
    # st.write(pd.DataFrame({
    #   "character": CHARACTERS,
    #  "score": scores
    # }))