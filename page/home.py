import streamlit as st
from utils import *


def app():
    image = decode_img("img/boston.jpg")

    st.write("# Boston housing price prediction")

    st.markdown(f"""
        <div class="container">
            <div class="row Viga">
                <div class="inner text-center">
                <img src="data:image/jpg;base64,{image}" width="500" height="300">
                </div>
                <a class="decoration-none" href="https://unsplash.com/photos/sMtKdbJfi_I"><p class="text-center">Image Source</p></a>
                <p align="justify">Boston housing data were collected in 1978 and each of the 506 entries represents aggregated data about 14 features for homes from various suburbs in Boston, Massachusetts. In this project, you will evaluate the performance and predictive power of a model that has been trained and tested on data collected from homes in suburban Boston, Massachusetts. Models trained on this data that are deemed suitable can then be used to make certain predictions about a house — specifically, its monetary value. This model proves invaluable to someone like a real estate agent who can tap into such information on a daily basis. You can see the full project <a href="https://github.com/MuhammadGeza/Boston-house-price-prediction" class="decoration-none">here</p>
            </div>
        </div>""", unsafe_allow_html=True)