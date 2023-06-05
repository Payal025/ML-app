import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib as plt
import streamlit as st
page_image_bg = f"""
<style>
[data-testid="stAppViewContainer"]{{
background-image: url("https://cdn.pixabay.com/photo/2020/04/12/20/37/abstract-5035778_1280.jpg");
background-size: cover;
background-repeat: no repeat;
}}

[data-testid="stHeader"]{{
background-image: url("https://cdn.pixabay.com/photo/2020/04/12/20/37/abstract-5035778_1280.jpg");
background-size: cover;
background-repeat: no repeat;
}}

[data-testid="stSidebar"]{{
background-image: url("https://img.freepik.com/free-vector/network-mesh-wire-digital-technology-background_1017-27428.jpg?w=360");
background-size: cover;
background-repeat: no repeat;
}}
</style>
"""
st.markdown(page_image_bg ,unsafe_allow_html = True)
import os
import pandas as pd
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
import pycaret
from pycaret.regression import setup, compare_models, pull, save_model

with st.sidebar:
    st.subheader("Welcome! This machine learning application is for regression tasks only :). Do try it out.")
    st.write("How to use this app: Please click on 'Upload' button to upload your dataset. Note: only .csv files allowed!")
    st.write("To do Analysis, click on 'EDA' button below.")
    st.write("To perform Regression task on your dataset, please select 'Model' button.")
    st.info("You can also download your file using 'Download' button below. The .pkl file will be downloaded.")
    st.caption("Cheers!")
    choice = st.selectbox("Please Select:",["Upload","EDA","Model","Download"])
    
if os.path.exists("sourcedata.csv"):
    df =  pd.read_csv("sourcedata.csv",index_col=None)

if choice== "Upload":
    st.title("Supervised Machine Learning using Regression")
    file = st.file_uploader("Upload your Data Here. Only .csv files")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("sourcedata.csv", index = None)
        st.dataframe(df)
            
if choice == "EDA":
    st.title("Explore your data in better way using Exploratory data analysis!")
    st.info("I used pandas_profiling here!") 
    profile_report = df.profile_report() 
    st_profile_report(profile_report)

if choice == "Model":
        st.title("Machine Learning algorithm")
        st.info("I am using Regression type Algorithm for my machine learning model.")
        if st.button("Train"):
            setup(df, target=target)
            setup_df = pull()
            st.info("Below will show you the Regression Algorithm result")
            st.dataframe(setup_df)
            best_model = compare_models()
            compare_df = pull()
            st.info("This is machine learning regression model")
            st.dataframe(compare_df)
            best_model
            save_model(best_model, 'MLmodel')

if choice=="Download":
         with open("MLmodel.pkl",'rb') as f:
             st.title("Download your model from here:")
             st.download_button("Download the fire",f, "MLmodel.pkl") 

             
             
            
                    
             