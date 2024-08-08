import numpy as np
from PIL import Image
import PIL.Image as Image
import csv
from st_on_hover_tabs import on_hover_tabs
import streamlit as st
import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np # linear algebra
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
warnings.filterwarnings('ignore')

centered_style = """
        display: flex;
        justify-content: center;
"""

font = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Kanit:wght@200&display=swap" rel="stylesheet">
"""
st.markdown(font,unsafe_allow_html=True)

css_styles = """
                <style>
                * {
                    font-family: 'Kanit', sans-serif !important;
                }
                .stTextArea {
                    height: auto;
                }
                div[class="css-keje6w e1tzin5v2"] {
                    column-gap: 100px;
                }
                h2 {
                    color: #5ba56e;
                }
                h3 {
                    color: #007a7a;
                }
                label[class="css-16huue1 effi0qh3"] {
                    font-size: 16px;
                }
                p {
                    color: #78701d;
                    font-size: 16px;
                }
                textarea {
                    color: #007a7a;
                }
                    section[data-testid='stSidebar'] 
                {
                    background-color: #111;
                    min-width: unset !important;
                    width: unset !important;
                    flex-shrink: unset !important;
                }

                button[kind="header"] {
                    background-color: transparent;
                    color: rgb(180, 167, 141);
                }

                @media (hover) {
                    /* header element to be removed */
                    header["data"-testid="stHeader"] {
                        display: none;
                    }

                    /* The navigation menu specs and size */
                    section[data-testid='stSidebar'] > div {
                        height: 100%;
                        width: 95px;
                        position: relative;
                        z-index: 1;
                        top: 0;
                        left: 0;
                        background-color: #111;
                        overflow-x: hidden;
                        transition: 0.5s ease;
                        padding-top: 60px;
                        white-space: nowrap;
                    }

                    /* The navigation menu open and close on hover and size */
                    /* section[data-testid='stSidebar'] > div {
                    height: 100%;
                    width: 75px; /* Put some width to hover on. */
                    /* } 

                    /* ON HOVER */
                    section[data-testid='stSidebar'] > div:hover{
                    width: 300px;
                    }

                    /* The button on the streamlit navigation menu - hidden */
                    button[kind="header"] {
                        display: none;
                    }
                }

                @media (max-width: 272px) {
                    section["data"-testid='stSidebar'] > div {
                        width: 15rem;
                    }/.
                }
                </style>
"""
st.markdown(css_styles, unsafe_allow_html=True)

st.markdown(
    """
<div style='border: 2px solid green; border-radius: 5px; padding: 10px; background-color: white; font-family: "Times New Roman", Times, serif;'>
    <h1 style='text-align: center; color: green; font-family: "Times New Roman", Times, serif;'>
    🏥 Lung Cancer Classification with Vision Transformer : จำแนกมะเร็งปอด 🫁
    </h1>
</div>
    """, unsafe_allow_html=True)
with st.sidebar:
    tabs = on_hover_tabs(tabName=['Home','Upload', 'Analytics', 'More Information', 'Reset'], 
    iconName=['home','upload', 'analytics', 'informations', 'refresh'], 
    styles={'navtab': {'background-color': '#111', 'color': '#818181', 'font-size': '18px', 
                    'transition': '.3s', 'white-space': 'nowrap', 'text-transform': 'uppercase'}, 
                    'tabOptionsStyle': 
                    {':hover :hover': {'color': 'red', 'cursor': 'pointer'}}, 'iconStyle': 
                    {'position': 'fixed', 'left': '7.5px', 'text-align': 'left'}, 'tabStyle': 
                    {'list-style-type': 'none', 'margin-bottom': '30px', 'padding-left': '30px'}}, 
                    key="1",default_choice=0)
    st.markdown(
    """
        <div style='border: 2px solid green; padding: 10px; white; margin-top: 5px; margin-buttom: 5px; margin-right: 20px; bottom: 50;'>
            <h1 style='text-align: center; color: yellow; font-size: 100%'> Lung Cancer Identification </h1>
            <h1 style='text-align: center; color: orange; font-size: 100%'> KMUTT & KMITL </h1>
            <h1 style='text-align: center; color: blue; font-size: 100%'> ✨ Thailand Innovation Awards ⚙️ </h1>
        </div>
    """, unsafe_allow_html=True)


if tabs == 'Upload':
    # Load the trained model
    model = joblib.load('./random_forest_lung_cancer_model.pkl')
    input_data = {
        'GENDER': 0,  # Assuming 'F' was encoded as 1
        'AGE': 59,
        'SMOKING': 2,
        'YELLOW_FINGERS': 2,
        'ANXIETY': 2,
        'PEER_PRESSURE': 2,
        'CHRONIC DISEASE': 2,
        'FATIGUE ': 2,  # Note the trailing space
        'ALLERGY ': 2,  # Note the trailing space
        'WHEEZING': 2,
        'ALCOHOL CONSUMING': 2,
        'COUGHING': 2,
        'SHORTNESS OF BREATH': 2,
        'SWALLOWING DIFFICULTY': 2,
        'CHEST PAIN': 2
    }
    
    st.markdown(
        """
        <style>
        .sky-border {
            text-align: center; 
            color: white; 
            font-size: 125%; 
            border: 2px solid #33FFFF; 
            padding: 10px;
            background-color: rgba(255, 255, 255, 0.5);
        }
        </style>
        """, 
        unsafe_allow_html=True
    )
    
    cols = st.columns(4)

    cols[0].markdown("<h1 class='sky-border'>GENDER</h1>", unsafe_allow_html=True)
    GENDER = cols[0].text_input(' ', value=input_data['GENDER'], key='GENDER')

    cols[1].markdown("<h1 class='sky-border'>AGE</h1>", unsafe_allow_html=True)
    AGE = cols[1].number_input(' ', value=input_data['AGE'], key='AGE')

    cols[2].markdown("<h1 class='sky-border'>SMOKING</h1>", unsafe_allow_html=True)
    SMOKING = cols[2].number_input(' ', value=input_data['SMOKING'], key='SMOKING')

    cols[3].markdown("<h1 class='sky-border'>YELLOW_FINGERS</h1>", unsafe_allow_html=True)
    YELLOW_FINGERS = cols[3].number_input(' ', value=input_data['YELLOW_FINGERS'], key='YELLOW_FINGERS')

    cols[0].markdown("<h1 class='sky-border'>ANXIETY</h1>", unsafe_allow_html=True)
    ANXIETY = cols[0].number_input(' ', value=input_data['ANXIETY'], key='ANXIETY')

    cols[1].markdown("<h1 class='sky-border'>PEER_PRESSURE</h1>", unsafe_allow_html=True)
    PEER_PRESSURE = cols[1].number_input(' ', value=input_data['PEER_PRESSURE'], key='PEER_PRESSURE')

    cols[2].markdown("<h1 class='sky-border'>CHRONIC DISEASE</h1>", unsafe_allow_html=True)
    CHRONIC_DISEASE = cols[2].number_input(' ', value=input_data['CHRONIC DISEASE'], key='CHRONIC DISEASE')

    cols[3].markdown("<h1 class='sky-border'>FATIGUE</h1>", unsafe_allow_html=True)
    FATIGUE = cols[3].number_input(' ', value=input_data['FATIGUE'], key='FATIGUE')

    cols[0].markdown("<h1 class='sky-border'>ALLERGY</h1>", unsafe_allow_html=True)
    ALLERGY = cols[0].number_input(' ', value=input_data['ALLERGY'], key='ALLERGY')

    cols[1].markdown("<h1 class='sky-border'>WHEEZING</h1>", unsafe_allow_html=True)
    WHEEZING = cols[1].number_input(' ', value=input_data['WHEEZING'], key='WHEEZING')

    cols[2].markdown("<h1 class='sky-border'>ALCOHOL CONSUMING</h1>", unsafe_allow_html=True)
    ALCOHOL_CONSUMING = cols[2].number_input(' ', value=input_data['ALCOHOL CONSUMING'], key='ALCOHOL CONSUMING')

    cols[3].markdown("<h1 class='sky-border'>COUGHING</h1>", unsafe_allow_html=True)
    COUGHING = cols[3].number_input(' ', value=input_data['COUGHING'], key='COUGHING')

    cols[0].markdown("<h1 class='sky-border'>SHORTNESS OF BREATH</h1>", unsafe_allow_html=True)
    SHORTNESS_OF_BREATH = cols[0].number_input(' ', value=input_data['SHORTNESS OF BREATH'], key='SHORTNESS OF BREATH')

    cols[1].markdown("<h1 class='sky-border'>SWALLOWING DIFFICULTY</h1>", unsafe_allow_html=True)
    SWALLOWING_DIFFICULTY = cols[1].number_input(' ', value=input_data['SWALLOWING DIFFICULTY'], key='SWALLOWING DIFFICULTY')

    cols[2].markdown("<h1 class='sky-border'>CHEST PAIN</h1>", unsafe_allow_html=True)
    CHEST_PAIN = cols[2].number_input(' ', value=input_data['CHEST PAIN'], key='CHEST PAIN')
    
    # Collect input data into a dictionary
    input_data = {
        'GENDER': GENDER,
        'AGE': AGE,
        'SMOKING': SMOKING,
        'YELLOW_FINGERS': YELLOW_FINGERS,
        'ANXIETY': ANXIETY,
        'PEER_PRESSURE': PEER_PRESSURE,
        'CHRONIC DISEASE': CHRONIC_DISEASE,
        'FATIGUE ': FATIGUE,
        'ALLERGY ': ALLERGY,
        'WHEEZING': WHEEZING,
        'ALCOHOL CONSUMING': ALCOHOL_CONSUMING,
        'COUGHING': COUGHING,
        'SHORTNESS OF BREATH': SHORTNESS_OF_BREATH,
        'SWALLOWING DIFFICULTY': SWALLOWING_DIFFICULTY,
        'CHEST PAIN': CHEST_PAIN
    }
    
    input_data['GENDER'] = 0 if input_data['GENDER'] == "M" else 1
    input_df = pd.DataFrame([input_data])
    predicted_label = model.predict(input_df)
    st.write(predicted_label)
