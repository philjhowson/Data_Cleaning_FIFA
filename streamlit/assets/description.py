import streamlit as st
from PIL import Image

def description():

    with Image.open('streamlit/images/FIFA_logo.png') as img:
        st.image(img)

    st.markdown("""<h2 style = "font-size: 1.5em;"><u>Data Cleaning
                & Analysis of FIFA Salaries</u></h2>""",
                unsafe_allow_html = True)

    st.markdown("""The objective of this project is to take a messy
                dataset, clean the data to make it usable, and then
                explore and analyze the data. I used a dataset scraped
                from the internet on FIFA salaries. The dataset was
                downloaded from kaggle:<br>
                <a href = "https://www.kaggle.com/datasets/yagunnersya/fifa-21-messy-raw-dataset-for-cleaning-exploring">FIFA 21 messy, raw dataset for cleaning/exploring</a><br>
                Thus, the project follows three major steps:<br>
                1. Data Cleaning<br>
                2. Data Exploration & Preparation<br>
                3. Model Training and Comparison""", unsafe_allow_html = True)
    
