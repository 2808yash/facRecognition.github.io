import streamlit as st
import cv2
import numpy as np
from PIL import Image
st.title("Face Recognition using OpenCV")
img=st.file_uploader("Upload any Image here for faces detection", type=("png", "jpg","jpeg"))
facecascade=cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
if img is not None:
    imag = Image.open(img)
    st.header("Original Image : ")
    st.image(imag,use_column_width=True)
    newimg=np.array(imag.convert('RGB'))
    im=cv2.cvtColor(newimg,1)
    img2=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=facecascade.detectMultiScale(img2,1.1,4)
    for (x,y,w,h) in faces:
         cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
    st.header("After Detection : ")
    st.image(im,use_column_width=True)
    count_faces=str(len(faces))
    st.header(count_faces+" Faces Detected")
