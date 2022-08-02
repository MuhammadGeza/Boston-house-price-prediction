import streamlit as st
from utils import *

def app():   
    model = load_model('model/model.pkl')
    X_test = load_df('data/X_test.csv')
    df = load_df('data/boston.csv')
    st.write("# Predict")

    method = st.selectbox('Select Method', ['File upload', 'Enter value manually'])

    if method=='File upload':
        predict_upload_file(model, X_test)

    if method=='Enter value manually':
        predict_manually(model, df)