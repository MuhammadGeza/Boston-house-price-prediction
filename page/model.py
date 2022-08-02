import streamlit as st
from utils import *

def app():
    model = load_model('model/model.pkl')
    X_train = load_df('data/X_train.csv')
    X_test = load_df('data/X_test.csv')
    y_train = load_df('data/y_train.csv')
    y_test = load_df('data/y_test.csv')

    st.write("# Model")

    st.markdown("""<p class="Viga">You can see the model creation flow <a href="https://github.com/MuhammadGeza" class="decoration-none">here</a></p>""", unsafe_allow_html=True)

    st.markdown("""<h2>What model to use?</h2>""", unsafe_allow_html=True)
    model_overview()
    

    st.markdown("""<h2 style="margin-top: 1.5rem;">Parameters</h2>""", unsafe_allow_html=True)
    model_parameters(model)

    st.markdown("""<h2 style="margin-top: 1.5rem;">Scores</h2>""", unsafe_allow_html=True)
    model_scores(model, X_train, X_test, y_train, y_test)

    st.markdown("""<h2 style="margin-top: 1.5rem;">The Ten best cross validation results</h2>""", unsafe_allow_html=True)
    model_cv_result(model)

    st.markdown("""<h2>Pipeline & Column Transformer</h2>""", unsafe_allow_html=True)
    model_pipeline(model)
    