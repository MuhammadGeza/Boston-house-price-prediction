import streamlit as st
from utils import *

def app():
    df = load_df('data/boston.csv')

    st.write("# About Dataset")

    st.markdown("""<p class="Viga">You can download the data required for this case study <a href="https://www.kaggle.com/datasets/fedesoriano/the-boston-houseprice-data" class="decoration-none">here</a></p>""", unsafe_allow_html=True)
    
    st.markdown("""<h2>Five samples from the dataset</h2>""", unsafe_allow_html=True)
    st.write(sample_data(df, 5))

    st.markdown("""<h2>Attribute Information</h2>""", unsafe_allow_html=True)
    atribut_information()

    st.markdown("""<h2 style="margin-top: 1.5rem;">Dataset Statistics</h2>""", unsafe_allow_html=True)
    dataset_statistics(df)

    st.markdown("""<h2 style="margin-top: 1.5rem;">Variabels overview</h2>""", unsafe_allow_html=True)
    variabels_overview(df)

    st.markdown("""<h2 style="margin-top: 1.5rem;">Interaction</h2>""", unsafe_allow_html=True)
    interactions(df)

    st.markdown("""<h2 style="margin-top: 1.5rem;">Correlations</h2>""", unsafe_allow_html=True)
    correlations(df)