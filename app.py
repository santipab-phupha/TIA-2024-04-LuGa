import numpy as np
from PIL import Image
import PIL.Image as Image
import csv
from st_on_hover_tabs import on_hover_tabs
import streamlit as st
import warnings
import cv2
import os
import pandas as pd

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
            <h1 style='text-align: center; color: blue; font-size: 100%'> ✨ Thailand Innovation Awards  ⚙️ </h1>
        </div>
    """, unsafe_allow_html=True)
