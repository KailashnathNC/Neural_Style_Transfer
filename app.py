import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2
import PIL

#harivinodn@gmail.com


from model import generate
from model2 import generate_g
from vgg19 import generate_vgg19

st.title("NST")

content = st.sidebar.file_uploader("Choose a content image")
style = st.sidebar.file_uploader("Choose a Style image")


if st.sidebar.button("Analyze"):
    if content and style:
        #st.write(content.name)
        #st.write(style.name)

        col1 , col2 = st.columns(2)

        with col1:

            content_image = cv2.resize(cv2.imread(content.name),(224,224))
            content_image_tf = tf.image.convert_image_dtype(content_image , tf.float32)


            fig , ax = plt.subplots(figsize = (10,10))

            plt.title("Content Image")

            ax = plt.imshow(cv2.cvtColor(np.array(content_image_tf), cv2.COLOR_BGR2RGB))
            plt.axis(False)

            st.pyplot(fig)

        with col2:

            ###
            style_image = cv2.resize(cv2.imread(style.name),(224,224))
            style_image_tf = tf.image.convert_image_dtype(style_image , tf.float32)


            fig , ax = plt.subplots(figsize = (10,10))

            plt.title("Style Image")


            ax = plt.imshow(cv2.cvtColor(np.array(style_image_tf), cv2.COLOR_BGR2RGB))
            plt.axis(False)

            st.pyplot(fig)

        ecol1, ecol2 = st.columns(2)
        
        with ecol1:
            
            tensor = generate(content_image , style_image)
            #st.write(tensor)

            fig , ax = plt.subplots(figsize = (6,6))

            #plt.title("Style Image")
            st.markdown("#### VGG-16")
            ax = plt.imshow(cv2.cvtColor(np.array(tensor), cv2.COLOR_BGR2RGB))
            plt.axis(False)

            st.pyplot(fig)



        with ecol2:
            tensor2 = generate_vgg19(content_image , style_image)
            #st.write(tensor2)

            fig , ax = plt.subplots(figsize = (6,6))

            #plt.title("Style Image")
            st.markdown("#### VGG-19")
            ax =  plt.imshow(cv2.cvtColor(np.array(tensor2), cv2.COLOR_BGR2RGB))
            plt.axis(False)

            st.pyplot(fig)

        